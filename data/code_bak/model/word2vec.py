#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter
from utils.segment import Segment
from gensim import matutils
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

class W2V(object):
    def __init__(self, word2vec_model_file):
        self.model_file = word2vec_model_file
        self.wv = KeyedVectors.load_word2vec_format(self.model_file, binary=True, encoding='latin1')
        self.segment = Segment()

    def __contains__(self, item):
        return self.wv.__contains__(item)

    def get_model_file(self):
        return self.model_file

    def vector_dim(self):
        return self.wv.vector_size

    def similar_by_word(self, word, topn=10):
        if word not in self.wv:
            print("word '%s' not in vocabulary" % word)
            return []
        return self.wv.similar_by_word(word, topn)

    def similarity(self, word1, word2):
        if word1 not in self.wv or word2 not in self.wv:
            return 0.0
        return self.wv.similarity(word1, word2)

    def infer_similarity_by_text(self, text1, text2):
        v1 = self.infer_vector_by_text(text1)
        v2 = self.infer_vector_by_text(text2)
        assert len(v1) > 0 and len(v2) > 0
        return np.dot(v1, v2)

    def infer_vector_by_text(self, text):
        words = self.segment.cut(text)
        v = [self.wv[word] for word in words if word in self.wv]
        if len(v) == 0:
            v = [self.wv[char] for char in list(text) if char in self.wv]
        return matutils.unitvec(np.array(v).mean(axis=0)) if len(v) > 0 else np.zeros(self.vector_dim())

    def infer_vector_by_textlist(self, text_list):
        return [self.infer_vector_by_text(v) for v in text_list]

    def get_vector(self, word):
        if word not in self.wv:
            print("word '%s' not in vocabulary" % word)
            return []

        return self.wv[word]

    def predict_proba(self, oword, iword):
        iword_vec = self.wv[iword]
        oword = self.wv.wv.vocab[oword]
        oword_l = self.wv.syn1[oword.point].T
        dot = np.dot(iword_vec, oword_l)
        lprob = -sum(np.logaddexp(0, -dot) + oword.code * dot)
        return lprob

    def relation_by_word(self, word, topn=10):
        r = {i:self.predict_proba(i, word) - np.log(j.count) for i, j in self.wv.wv.vocab.items()}
        return Counter(r).most_common(n=topn)

    def get_keywords(self, text, topn=10, with_proba=False):
        words = self.segment.cut(text)
        return self.get_keywords_by_wordslist(words, topn, with_proba)

    def get_keywords_by_wordslist(self, words_list, topn=10, with_proba=False):
        topn = min(topn, len(words_list))
        words = [w for w in words_list if w in self.wv]
        words_dict = {w: sum([self.predict_proba(u, w) for u in words]) for w in words}
        tuple_list = Counter(words_dict).most_common(n=topn)
        return tuple_list if with_proba else [item[0] for item in tuple_list]
