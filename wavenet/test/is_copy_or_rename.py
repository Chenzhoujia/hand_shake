# -*- coding: UTF-8 -*-
#执行的是拷贝
import tensorflow as tf
import numpy as np
t1 = tf.constant([[1.0,2],
                 [3,4],
                 [5,6],
                 [7,8],
                 [9,10]])
t3 = t1+1
t2 = t1[:3,:]
t2 = t2+1
t_l = []
t_l.append(t1)
t1 = t1+1
t_l.append(t1)
t1 = t1+1
t_l.append(t1)
with tf.Session() as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 输出卷积值
    print(sess.run([t_l[0],t_l[1],t_l[2]]))