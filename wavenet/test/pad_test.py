# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
t1 = tf.constant([[1.0,2],
                 [3,4],
                 [5,6],
                 [7,8],
                 [9,10]])

t2 = tf.constant([[[3.0,3],
                 [4,5],
                 [6,7],
                 [8,9],
                 [10,11]]])
t3 = tf.square(t2-t1)/(tf.square(t2)+tf.constant(1e-10))
t4 = tf.reduce_mean(t1)
t5 = tf.pad(t1, [[3, 0], [0, 0]])
t5 = tf.concat([t5,t5,t5],axis=1)

t5 = tf.pad(t1, [[0, 0], [0, 3]])

t6 = t5[(3-1):,:]
with tf.Session() as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 输出卷积值
    print(sess.run(t6))