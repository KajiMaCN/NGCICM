import tensorflow as tf
from src.inits import glorot
from models.crf import crf_layer


conv1d = tf.layers.conv1d

def attn_head(interaction,seq,out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0):
  with tf.name_scope('my_attn'):
    if in_drop != 0.0:
       seq = tf.nn.dropout(seq, 1.0 - in_drop)
    seq_fts = seq
    latent_factor_size = 8  
    
    w_1 = glorot([seq_fts.shape[2].value,latent_factor_size])
    w_2 = glorot([2*seq_fts.shape[2].value,latent_factor_size])

    f_1 = tf.layers.conv1d(seq_fts, out_sz, 1)
    f_1 = tf.layers.conv1d(f_1, 1, 1)
    f_2 = tf.layers.conv1d(seq_fts, out_sz, 1)
    f_2 = tf.layers.conv1d(f_2, 1, 1)
    logits = f_1 + tf.transpose(f_2, [0, 2, 1])
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits[0]) + bias_mat[0])

    if coef_drop != 0.0:
       coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
    if in_drop != 0.0:
       seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
    
    neigh_embs = tf.matmul(coefs, seq_fts[0])
    hidden_crf1 = tf.add(seq_fts[0],neigh_embs)
    hidden_new1 = tf.add(seq_fts[0],neigh_embs)
    hidden_crf2 = tf.concat([seq_fts[0],neigh_embs],axis=-1)
    hidden_new2 = tf.concat([seq_fts[0],neigh_embs],axis=-1)
    for cv in range(0, 1):
      hidden_crf1 = crf_layer(interaction, hidden_crf1, hidden_new1,out_sz)
      hidden_new1 = hidden_crf1
      hidden_crf2 = crf_layer(interaction,hidden_crf2, hidden_new2,out_sz)
      hidden_new2 = hidden_crf2

    neigh_embs_aggre_11 = tf.matmul(hidden_new1, w_1)
    neigh_embs_aggre_22 = tf.matmul(hidden_new2, w_2)
    final_embs = activation(neigh_embs_aggre_11) + activation(neigh_embs_aggre_22)

    return final_embs, coefs