"""Unit tests for the causal_conv op."""

import numpy as np
import tensorflow as tf

t1=tf.expand_dims(tf.constant([[1,2,3],[1,2,3]]),0)
t2=tf.expand_dims(tf.constant([[4,5,6],[4,5,6]]),0)
concated = tf.concat([t1,t2],axis=2)

with tf.Session() as sess:
    print(sess.run(t1))
    print(sess.run(t2))
    print(sess.run(concated))