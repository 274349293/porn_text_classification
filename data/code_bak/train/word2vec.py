#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
from gensim.models import Word2Vec
from .pathiterator import MongoIterator


class W2V(object):
    def __init__(self, model_file, size=100, window=5, min_count=5, iter=10):
        '''
        size: 词向量维度
        window: 中心词离上下文的最大长度
        '''
        self.model_file = model_file
        self.size = size
        self.window = window
        self.min_count = min_count
        self.iter = iter
        self.model = None
        self.sg = 0
        self.hs = 0
        self.save_only_vector = False
        self.worker = multiprocessing.cpu_count()

    def fit(self, corpus):
        '''
        corpus: iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.
        '''
        self.model = Word2Vec(
            sentences=corpus,
            size=self.size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            hs=self.hs,
            iter=self.iter,
            workers=self.worker)
        self.model.save(self.model_file)
        return self.model

    def fit_by_mongo(self):
        return self.fit(MongoIterator())
