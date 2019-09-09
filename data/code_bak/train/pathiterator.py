#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.mongo_util import MongoUtil

import time

class MongoIterator(object):
    def __init__(self):
        mongoUtil = MongoUtil()
        self.cut_documents = []

        count = 0
        b = time.time()
        for i in mongoUtil.get_all_cut_sex_doc():
            count += 1
            self.cut_documents.append(i['cut_doc'].split(' '))
        print('[MongoIterator] scan file count:%d \r'%(count), end='')
        print('\n[MongoIterator] scan file finished, count:%d time:%d' %(count, time.time() - b))

    # __iter__调用多次(扫描次数 1 + 迭代次数n, n要看word2vec的训练迭代轮次参数，默认是5, 尽量不要放耗时操作)
    def __iter__(self):
        b = time.time()
        total = len(self.cut_documents)
        for i, words in enumerate(self.cut_documents):
            # words = words.encode('utf-8')
            yield words
            print('[MongoIterator] %.2f%% ===> %d / %d \r' % ((i + 1) * 100.0 / total, i + 1, total), end='')
        print('\n[MongoIterator] trainning, count:%d, time:%d' % (total, time.time() - b))
