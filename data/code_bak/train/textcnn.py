# coding:utf-8
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .textcnn_model import TextCNN
from ..freeze_model import freeze_graph, load_graph, get_graph_op

import os
import time
MODEL_NAME = 'textcnn'
import tempfile

class TextcnnTrain(object):
    def __init__(self, vocab_size, sequence_length, num_classes, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda, batch_norm):
        self.continue_train = True
        self.embedding_size = embedding_size
        self.optimizer = tf.train.AdamOptimizer
        self.learning_rate = 0.001
        self.init_embedding_weights = None
        self.model_name = MODEL_NAME
        self.model = TextCNN(
            sequence_length=sequence_length,
            num_classes=num_classes,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda,
            batch_norm=batch_norm)

        temp_dir = tempfile.gettempdir()
        self.model_dir = os.path.join(temp_dir, 'model')
        self.summaries_dir = os.path.join(temp_dir, 'summaries')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def set_continue_train(self, continue_train):
        self.continue_train = continue_train

    def set_model_dir(self, model_dir, model_name=MODEL_NAME):
        self.model_dir = model_dir
        self.model_name = model_name
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def set_summaries_dir(self, summaries_dir):
        self.summaries_dir = summaries_dir

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def _init_model_train_op(self, sess):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = self.optimizer(learning_rate=self.learning_rate)
        grads_and_vars = opt.compute_gradients(self.model.loss)
        train_op = opt.apply_gradients(
            grads_and_vars=grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.model.loss)
        acc_summary = tf.summary.scalar("accuracy", self.model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(self.summaries_dir, 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # Test Summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        test_summary_dir = os.path.join(self.summaries_dir, 'test')
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
        return train_op, train_summary_op, train_summary_writer, test_summary_op, test_summary_writer

    def _reload_model_params(self, sess):
        saver = tf.train.Saver()
        ckpt_has_restore = False
        if self.continue_train:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                ckpt_has_restore = True

        if not ckpt_has_restore:
            print("Created model with fresh parameters.")
            if self.init_embedding_weights is not None:
                # 采用预训练word2vec初始化embedding权重矩阵
                sess.run(self.model.weights.assign(self.init_embedding_weights))
        return saver

    def init_embedding_weights_with_word2vec(self, init_w, embedding_dim):
        if self.embedding_size != embedding_dim:
            raise 'embedding_size(%d) must == word_vector_size(%d)' % (self.embedding_size, embedding_dim)
        self.init_embedding_weights = init_w

    def train(self, dropout, epoches, train_next_batch_cb, test_next_batch_cb):
        b = time.time()
        print('trainning start...')
        ckpfile = os.path.join(self.model_dir, self.model_name)
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            train_op, train_summary_op, train_summary_writer, test_summary_op, test_summary_writer = self._init_model_train_op(sess)
            sess.run(tf.global_variables_initializer())
            saver = self._reload_model_params(sess)

            next_train_batch_index = 0
            next_test_batch_index = 0
            for epoch in range(epoches):
                x, y, next_train_batch_index = train_next_batch_cb(next_train_batch_index)
                feed_dict = {
                    self.model.input_x: x,
                    self.model.input_y: y,
                    self.model.dropout_keep_prob: dropout
                }
                _, loss, acc, train_summary = sess.run(
                    [train_op, self.model.loss, self.model.accuracy, train_summary_op],
                    feed_dict=feed_dict)
                train_summary_writer.add_summary(train_summary, epoch)
                # print('step:%d -->(batch_idx:%d) loss:%f, acc:%f' % (epoch, next_train_batch_index, loss, acc))
                if epoch % 50 == 0:
                    if test_next_batch_cb:
                        test_x, test_y, next_test_batch_index = test_next_batch_cb(next_test_batch_index)
                        feed_dict = {
                            self.model.input_x: test_x,
                            self.model.input_y: test_y,
                            self.model.dropout_keep_prob: 1.0
                        }
                        test_loss, test_acc, test_num_correct, test_summary = sess.run(
                            [self.model.loss, self.model.accuracy, self.model.num_correct, test_summary_op],
                            feed_dict=feed_dict)
                        test_summary_writer.add_summary(test_summary, epoch)
                        print('        step:%d  loss:%f, acc:%f, test_loss:%f, test_acc:%f, test_num_correct:%d' % (
                            epoch, loss, acc, test_loss, test_acc, test_num_correct))
                    saver.save(sess, ckpfile)
            saver.save(sess, ckpfile)
            mode_file = self.model_name + '.pb'
            freeze_graph(self.model_dir, mode_file, 'output/pred_proba')
        print('trainning finished...(time:%d)' % (time.time() - b))
        return os.path.join(self.model_dir, mode_file)

class TextcnnPredict(object):
    def __init__(self, model_pb_file):
        self.sess_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.graph = load_graph(model_pb_file, is_debug=False)
        self._input_x = get_graph_op(self.graph, 'input_x')
        self._dropout_keep_prob = get_graph_op(self.graph, 'dropout_keep_prob')
        self._pred_proba = get_graph_op(self.graph, 'output/pred_proba')

    def predict(self, x):
        y = self.predict_proba(x)
        return [np.argmax(row, axis=0) for row in y]

    def predict_proba(self, x):
        feed_dict = {
            self._input_x: x,
            self._dropout_keep_prob: 1.0
        }
        sess = tf.Session(graph=self.graph, config=self.sess_conf)
        with sess.as_default():
            y = sess.run(self._pred_proba, feed_dict=feed_dict)
        return y
