# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
t = tf.constant([[[1,2],
                 [3,4],
                 [5,6],
                 [7,8],
                 [9,10]]])
t2 = tf.pad(t, [[0, 0], [0, 7], [0, 0]])  # [[[3, 3, 3]]]

with tf.Session() as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 输出卷积值
    print(sess.run(t2))