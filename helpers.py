import tensorflow as tf
import numpy as np
import os

def get_exemplary_exps():
    if(os.path.exists('./good_e_bad_e.npz')):
        cached = np.load('./good_e_bad_e.npz') 
        arrays = [cached[f] for f in sorted(cached.files)]
        good_e, bad_e = arrays
        good_e, bad_e = tf.convert_to_tensor(good_e, dtype=tf.float32), tf.convert_to_tensor(bad_e, dtype=tf.float32)
        
        good_e = tf.reshape(good_e, [-1])
        good_e = tf.square(good_e)
        bad_e = tf.reshape(bad_e, [-1])
        bad_e = tf.square(bad_e)
        
        good_e = tf.tile(tf.expand_dims(good_e,0),[batch_size,1])
        bad_e = tf.tile(tf.expand_dims(bad_e,0),[batch_size,1])
        return good_exp, bad_exp
    else:
        print('exemplary explanations not found on disk.')