# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, input_batch, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        # choose different rnn cell 
        if args["model"] == 'rnn':
            cell_fn = rnn.RNNCell
        elif args["model"] == 'gru':
            cell_fn = rnn.GRUCell
        elif args["model"] == 'lstm':
            cell_fn = rnn.LSTMCell
        elif args["model"] == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args["model"]))

        # warp multi layered rnn cell into one cell with dropout
        cells = []
        for _ in range(args["num_layers"]):
            cell = cell_fn(args["rnn_size"])
            if training and (args["output_keep_prob"] < 1.0 or args["input_keep_prob"] < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=args["input_keep_prob"],
                                          output_keep_prob=args["output_keep_prob"])
            cells.append(cell)
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

        # input/target data (int32 since input is char-level)
        """
        self.input_data = tf.placeholder(
            tf.int32, [args["batch_size"], args["seq_length"]])
        self.targets = tf.placeholder(
            tf.int32, [args["batch_size"], args["seq_length"]])
        """


        # softmax output layer, use softmax to classify

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args["rnn_size"], args["vocab_size"]])
            softmax_b = tf.get_variable("softmax_b", [args["vocab_size"]])


        # transform input to embedding神经网络，一般是对实数的向量起作用。他们最好是在密集向量上训练，向量中所有的值都有助于定义一个对象。然而，对于机器学习的许多重要的输入，例如文字，没有自然的矢量表示。Embedding函数是将这些离散输入对象转换为有用的连续向量的标准和有效的方法。
        #https://vimsky.com/article/3656.html
        """
        embedding = tf.get_variable("embedding", [args["vocab_size"], args["rnn_size"]])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        """
        inputs = input_batch[:, 0:args["seq_length"], 0:9]
        self.original_inputs = inputs
        self.original_labels = input_batch[:, 0:args["seq_length"], 9:12]
        #self.initial_state = cell.zero_state(args["batch_size"], tf.float32)
        self.initial_state = cell.zero_state(1, tf.float32)
        # dropout beta testing: double check which one should affect next line
        if training and args["output_keep_prob"]:
            inputs = tf.nn.dropout(inputs, args["output_keep_prob"])

        # unstack the input to fits in rnn model
        inputs = tf.split(inputs, args["seq_length"], 1)#每个单元包含batch个元素
        self.inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # loop function for rnn_decoder, which take the previous i-th cell's output and generate the (i+1)-th cell's input
        """
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)
        """
        # rnn_decoder to generate the ouputs and final state. When we are not training the model, we use the loop function.
        #outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        outputs, last_state = legacy_seq2seq.rnn_decoder(self.inputs, self.initial_state, cell,
                                                         loop_function=None, scope='rnnlm')
        self.output = tf.reshape(tf.concat(outputs, 1), [-1, args["rnn_size"]])

        # output layer

        self.logits = tf.matmul(self.output, softmax_w) + softmax_b

        loss = tf.square(self.logits - self.original_labels[0,:,:])
        loss = tf.reduce_mean(loss)
        self.loss = loss


    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for _ in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
