# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib import learn
import os
class Dataset(object):
    def __init__(self, max_text_limit_len, train_set, vocab_file):
        '''
        max_text_limit_len: 每篇文章最大限制长度
        train_set: 训练数据集,格式:
                    第一行:类别数量，测试样本数量
                    第2-n行:类别,一篇文章的分词(分词之间空格隔开)
        vocab_file:词汇表文件(train_set中的所有词汇）
        '''
        labels = []
        texts = []
        max_document_length = 0
        with open(train_set, 'r', encoding='utf-8') as rf:
            first = rf.readline().strip().split(',')
            self._num_classes = int(first[0])
            test_size = int(first[1])
            for row in rf.readlines():
                items = row.strip().split(',')
                onehot = [0 for _ in range(self._num_classes)]
                onehot[int(items[0])] = 1
                labels.append(np.array(onehot))
                texts.append(items[1])
                wdlist = items[1].split(' ')
                if max_document_length < len(wdlist):
                    max_document_length = len(wdlist)
        if max_document_length > max_text_limit_len:
            max_document_length = max_text_limit_len
        if os.path.exists(vocab_file):
            self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_file)
            texts = np.array(list(self.vocab_processor.transform(texts)))
        else:
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            texts = np.array(list(self.vocab_processor.fit_transform(texts)))
            self.vocab_processor.save(vocab_file)

        labels = np.array(labels)
        batches = list(zip(texts, labels))
        n = len(batches)
        idx = test_size
        self.test_batches = batches[0:idx]
        self.train_batches = batches[idx:]
        self.train_num = n - idx
        self.test_num = idx
        print('train_num:%d, test_num:%d' % (self.train_num, self.test_num))
        print('vocab_size:%d, sequence_length:%d, max_doc_length:%d' % (
            self.vocab_size(), texts.shape[1], self.vocab_processor.max_document_length))

    def num_classes(self):
        return self._num_classes

    def vocab_size(self):
        return len(self.vocab_processor.vocabulary_)

    def sequence_length(self):
        return self.vocab_processor.max_document_length

    def init_embedding_weights_with_word2vec(self, w2v_file):
        from gensim.models.keyedvectors import KeyedVectors
        w2v = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
        embedding_dim = w2v.vector_size
        init_w = np.random.uniform(-0.25, 0.25, (self.vocab_size(), embedding_dim))
        for idx in range(self.vocab_size()):
            word = self.vocab_processor.vocabulary_.reverse(idx)
            if word in w2v:
                weight = w2v[word]
                init_w[idx] = weight
        return init_w, embedding_dim

    def next_batch(self, idx, batch_size):
        assert batch_size < self.train_num
        if idx + batch_size > self.train_num:
            np.random.shuffle(self.train_batches)
            res_batches = self.train_batches[0: batch_size]
            next_idx = batch_size
        else:
            res_batches = self.train_batches[idx: idx + batch_size]
            next_idx = idx + batch_size
        x_batches, y_batches = zip(*res_batches)
        return x_batches, y_batches, next_idx

    def next_test_batch(self, idx, batch_size):
        assert batch_size < self.test_num
        if idx + batch_size > self.test_num:
            np.random.shuffle(self.test_batches)
            res_batches = self.test_batches[0: batch_size]
            next_idx = batch_size
        else:
            res_batches = self.test_batches[idx: idx + batch_size]
            next_idx = idx + batch_size
        x_batches, y_batches = zip(*res_batches)
        return x_batches, y_batches, next_idx
