import tensorflow as tf

from src import layers
from models.base_gattn import BaseGAttN
from src.inits import glorot
from src.metrics import masked_accuracy


class GAT(BaseGAttN):
    
    def encoder(interaction,inputs, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, delta, k, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(k[0]):
            attn_temp, coefs = layers.attn_head(interaction, inputs, bias_mat=bias_mat,
                                                out_sz=delta, activation=activation,
                                                in_drop=ffd_drop, coef_drop=attn_drop,)
            inputs = attn_temp[tf.newaxis]
            attns.append(attn_temp)
        h_1 = tf.concat(attns, axis=-1)
        return h_1, coefs
            
    def decoder(embed, nd):
        embed_size = embed.shape[1].value
        with tf.variable_scope("deco"):
             weight3 = glorot([embed_size,embed_size])
        U=embed[0:nd,:]
        V=embed[nd:,:]
        logits=tf.matmul(tf.matmul(U,weight3),tf.transpose(V))
        logits=tf.reshape(logits,[-1,1])

        return tf.nn.relu(logits)
    
    def loss_sum(scores, lbl_in, msk_in, neg_msk, weight_decay,l2_coef):
        loss_basic = masked_accuracy(scores, lbl_in, msk_in, neg_msk)
        loss_basic +=  weight_decay * l2_coef
        return loss_basic