# coding:utf-8
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf


class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0, batch_norm=False):
        '''
        sequence_length: 文本的长度，我们把所有的文本都填充成了相同的长度。
        num_classes: 输出层的类别数。
        vocab_size: 词汇表的大小。
        embedding_size: 嵌入的维度(词向量维度)
        filter_sizes: convolutional filters 覆盖的words的个数，对于每个size，我们会有 num_filters 个 filters。比如 [3,4,5] 表示我们有分别滑过3，4，5个 words 的 filters，总共是3 * num_filters 个 filters。
        num_filters: 每一个filter size的filters数量
        l2_reg_lambda: l2正则化基数
        batch_norm: BN正则化
        '''
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # embedded操作的结果是形状为 [None, sequence_length, embedding_size]
            self.weights = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True,
                name='weights')
            # 卷积conv2d操作接收一个4维的张量，维度分别代表batch, width, height 和 channel。
            # 我们的embedding结果不包含channel维度，因此要手动添加，得到形状为[None, sequence_length, embedding_size, 1]的embedding层。
            embedded = tf.nn.embedding_lookup(self.weights, self.input_x)
            embedded = tf.expand_dims(embedded, -1)

        pooled_output = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%d' % filter_size):
                # conv2d的filter形状[filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(embedded, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                y_ = tf.nn.bias_add(conv, b)
                if batch_norm:
                    fc_mean, fc_var = tf.nn.moments(y_, axes=[0])
                    scale = tf.Variable(tf.ones([num_filters]))
                    shift = tf.Variable(tf.zeros([num_filters]))
                    epsilon = 0.001
                    y_ = tf.nn.batch_normalization(y_, fc_mean, fc_var, shift, scale, epsilon)

                # h输出形状:[1, sequence_length - filter_size + 1, 1, 1]
                h = tf.nn.relu(y_, name='relu')

                # 每个filter的max-pool输出形状:[batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_output.append(pooled)

        # 从每个filter size得到了所有的pool了的输出张量积，将它们结合在一起形成一个长的特征向量，
        # 形状是[batch_size, num_filters_total]
        # tf.reshape中使用-1是告诉TensorFlow把维度展平
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_output, axis=3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        l2_loss = tf.constant(0.0)
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, axis=1, name='predictions')
            self.pred_proba = tf.nn.softmax(self.scores, name='pred_proba')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')

if __name__ == '__main__':
    TextCNN(1000, 2, 5000, 50, [2, 3, 4], 8)