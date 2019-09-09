# coding:utf-8
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import jieba
import jieba.posseg as pseg
import os
import logging
jieba.setLogLevel(logging.WARN)

STOPWORDS_DIR = "/data/code/data/stopwords/"
SEGMENT_DIR = '/data/code/data/userdict/'

class Segment(object):
    def __init__(self, filter_digit=True, filter_stopwords=True):
        self.stopwords = set()
        for fname in os.listdir(STOPWORDS_DIR):
            fpath = os.path.join(STOPWORDS_DIR, fname)
            if os.path.isfile(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    for row in f:
                        self.stopwords.add(row.strip())

        if os.path.exists(SEGMENT_DIR):
            for fname in os.listdir(SEGMENT_DIR):
                user_dict = os.path.join(SEGMENT_DIR, fname)
                if os.path.isfile(user_dict):
                    jieba.load_userdict(user_dict)

        self.filters = []
        self.add_filter(lambda w: w.isspace())
        if filter_digit:
            self.add_filter(lambda w: w.isdigit())
        if filter_stopwords:
            self.add_filter(lambda w: w in self.stopwords)

    def get_userdict(self):
        res = set()
        with jieba.get_dict_file() as rf:
            for w in rf:
                res.add(w.strip())
        return list(res)

    def load_stopwords(self, stopwords_file):
        if os.path.isfile(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for row in f:
                    self.stopwords.add(row.strip())

    def load_userdict(self, user_dict):
        if os.path.exists(user_dict):
            jieba.load_userdict(user_dict)

    def add_filter(self, filter):
        self.filters.append(filter)

    def split(self, text, seg=' '):
        words = text.split(seg)
        return [w for w in words if not self.__fitlter_func(w)]

    def cut(self, text, hmm=True, pos=False):
        pairs = pseg.lcut(text, HMM=hmm)
        if pos:
            return [(pair.word, pair.flag) for pair in pairs]
        else:
            return self._do_filter(pairs)

    def __fitlter_func(self, w):
        for filter in self.filters:
            if filter(w):
                return True
        return False

    def _do_filter(self, pairs):
        res = []
        for pair in pairs:
            w = pair.word
            if not self.__fitlter_func(w):
                res.append(w)
        return res

if __name__ == '__main__':
    seg = Segment()
    cut_doc = seg.cut('''''')
    print(cut_doc)