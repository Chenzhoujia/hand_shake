# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])
t = tf.constant([[[1,2],
                 [3,4],
                 [5,6],
                 [7,8],
                 [9,10]]])
t2 = time_to_batch(t,2)
with tf.Session() as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 输出卷积值
    print(sess.run(t2))