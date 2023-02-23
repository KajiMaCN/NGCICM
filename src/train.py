import time
import numpy as np
import tensorflow as tf

from src.inits import adj_to_bias
from src.inits import test_negative_sample
from src.inits import init_data
from src.inits import generate_mask
from src.metrics import masked_accuracy
from src.metrics import ROC


def train(labels,AM,CF,MF,train_arr, test_arr,args):

    model =args.model
    batch_size=args.batch_size

    interaction, features,matrix_shape, y_train, y_test, train_mask, test_mask, labels = init_data(labels,AM,CF,MF,train_arr, test_arr)
    nb_nodes = features.shape[0]  
    ft_size = features.shape[1]  

    features = features[np.newaxis]
    interaction = interaction[np.newaxis]
    biases = adj_to_bias(interaction, [nb_nodes], nhood=1)

    entry_size = matrix_shape[0] * matrix_shape[1]
    with tf.Graph().as_default():
        with tf.name_scope('input'):
              feature_in = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, nb_nodes, ft_size))
              bias_in = tf.placeholder(dtype=tf.float32, shape=(args.batch_size, nb_nodes, nb_nodes))
              lbl_in = tf.placeholder(dtype=tf.int32, shape=(entry_size, args.batch_size))
              msk_in = tf.placeholder(dtype=tf.int32, shape=(entry_size, args.batch_size))
              neg_msk = tf.placeholder(dtype=tf.int32, shape=(entry_size,args.batch_size))
              attn_drop = tf.placeholder(dtype=tf.float32, shape=())
              ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
              is_train = tf.placeholder(dtype=tf.bool, shape=())
        
        final_embedding, coefs = model.encoder(interaction,feature_in, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                delta=args.delta, k=args.k,
                                activation=args.act)
        scores = model.decoder(final_embedding, matrix_shape[0])
        loss = model.loss_sum(scores, lbl_in, msk_in, neg_msk, args.weight_decay, args.l2_coef)
        accuracy = masked_accuracy(scores, lbl_in, msk_in, neg_msk)
        train_op = model.training(loss, args.lr)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:              
          sess.run(init_op)

          train_loss_avg = 0
          train_acc_avg = 0

          for epoch in range(args.nb_epochs):
              
              t = time.time()
              
              ##########    train     ##############
              
              tr_step = 0
              tr_size = features.shape[0] 
              
              neg_mask, label_neg = generate_mask(labels, len(train_arr),matrix_shape)
              
              while tr_step * batch_size < tr_size:  
                      _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                      feed_dict={
                           feature_in: features[tr_step*batch_size:(tr_step+1)*batch_size],   
                           bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
                           lbl_in: y_train,
                           msk_in: train_mask,
                           neg_msk: neg_mask,
                           is_train: True,
                           attn_drop: 0.1, ffd_drop: 0.1})
                      train_loss_avg += loss_value_tr
                      train_acc_avg += acc_tr
                      tr_step += 1
              print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch+1), loss_value_tr,1-acc_tr, time.time()-t))
          
          print("Finish traing.")
          
          ###########     test      ############
          
          ts_size = features.shape[0]
          ts_step = 0
          ts_loss = 0.0
          ts_acc = 0.0
    
          print("Start to test")
          while ts_step * batch_size < ts_size:
              out_come, emb, coef, loss_value_ts, acc_ts = sess.run([scores, final_embedding, coefs, loss, accuracy],
                      feed_dict={
                          feature_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                          bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                          lbl_in: y_test,
                          msk_in: test_mask,
                          neg_msk: neg_mask,
                          is_train: False,
                          attn_drop: 0.0, ffd_drop: 0.0})
              ts_loss += loss_value_ts
              ts_acc += acc_ts
              ts_step += 1
          acc = 1 - ts_acc / ts_step
          print('Test loss:', ts_loss/ts_step, '; Test accuracy:', acc)

          out_come = out_come.reshape((matrix_shape[0],matrix_shape[1]))
          test_negative_samples = test_negative_sample(labels,len(test_arr),neg_mask.reshape((matrix_shape[0],matrix_shape[1])))
          test_labels, score = ROC(out_come, labels, test_arr, test_negative_samples)
          sess.close()
          return test_labels, score,acc