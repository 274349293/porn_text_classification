# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import numpy as np
from utils.utils import is_alpha, clean_text, dump_json
from utils.segment import Segment
from .train import TextClassifier as Impl
import os
import time
import csv

class _SexClassifier(Impl):
    def __init__(
            self,
            model_dir,
            model_name,
            embeding_size,
            filter_sizes,
            num_filters):
        super(_SexClassifier, self).__init__(
            model_dir,
            model_name,
            embeding_size,
            filter_sizes,
            num_filters)
        self.word2vec_file = None
        self.word2vec_size = 20
        self.word2vec_iter = 10

    def _preprocess(self, data_dir, trainset_file, tokenizer):
        seg = Segment()
        seg.add_filter(lambda w: is_alpha(w) and len(w) > 10)

        b = time.time()
        train_set = []
        test_set = []
        category = dict()
        for fname in os.listdir(data_dir):
            cate_fpath = os.path.join(data_dir, fname)
            if os.path.isdir(cate_fpath):
                print('processing:%s' % fname)
                cate_set = []
                if fname not in category:
                    id = len(category)
                    category[fname] = id
                else:
                    id = category[fname]
                for (root, _, files) in os.walk(cate_fpath):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        with open(fpath, 'r', encoding='utf-8') as rf:
                            text = clean_text(rf.read())
                            words = seg.cut(text)
                            if len(words) == 0:
                                continue
                            cate_set.append([id, ' '.join(words)])
                n = int(len(cate_set) * 0.1)
                test_set.extend(cate_set[0:n])
                train_set.extend(cate_set[n:])
        np.random.shuffle(test_set)
        np.random.shuffle(train_set)
        with open(trainset_file, 'w', encoding='utf-8') as wf:
            writer = csv.writer(wf, delimiter=',', lineterminator='\n')
            writer.writerow([len(category), len(test_set)])
            writer.writerows(test_set)
            writer.writerows(train_set)
        print('finished time:%d' % (time.time() - b))
        return category

    def _preprocess_sample_balanced(self, data_dir, trainset_file, tokenizer):
        seg = Segment()
        seg.add_filter(lambda w: is_alpha(w) and len(w) > 10)

        b = time.time()
        train_set = []
        test_set = []
        category = dict()
        cate_set_list = []
        for fname in os.listdir(data_dir):
            cate_fpath = os.path.join(data_dir, fname)
            if os.path.isdir(cate_fpath):
                print('processing:%s' % fname)
                cate_set = []
                if fname not in category:
                    id = len(category)
                    category[fname] = id
                else:
                    id = category[fname]
                for (root, _, files) in os.walk(cate_fpath):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        with open(fpath, 'r', encoding='utf-8') as rf:
                            text = clean_text(rf.read())
                            words = seg.cut(text)
                            if len(words) == 0:
                                continue
                            cate_set.append([id, ' '.join(words)])
                cate_set_list.append(cate_set)

        min_len = min([len(x) for x in cate_set_list])
        max_len = min_len * 3
        for cate_set in cate_set_list:
            size = len(cate_set)
            train_n = int(size * 0.9)
            if train_n > max_len:
                train_n = max_len
            test_n = size - train_n
            test_set.extend(cate_set[0:test_n])
            train_set.extend(cate_set[test_n:])

        np.random.shuffle(test_set)
        np.random.shuffle(train_set)
        with open(trainset_file, 'w', encoding='utf-8') as wf:
            writer = csv.writer(wf, delimiter=',', lineterminator='\n')
            writer.writerow([len(category), len(test_set)])
            writer.writerows(test_set)
            writer.writerows(train_set)
        print('finished time:%d' % (time.time() - b))
        return category

    def set_word2vec_fit_info(self, word2vec_file, word2vec_size, iter_num):
        self.word2vec_file = word2vec_file
        self.word2vec_size = word2vec_size
        self.word2vec_iter = iter_num

    def fit(self, data_dir, temp_dir, sample_balanced, tokenizer):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        if self.word2vec_file:
            w2v_file = self.word2vec_file
        # elif self.word2vec_size > 0:
        #     w2v_file = os.path.join(temp_dir, 'word2vec')
        #     if not os.path.exists(w2v_file):
        #         from train.word2vec import W2V
        #         wv = W2V(w2v_file, size=self.word2vec_size, iter=self.word2vec_iter)
        #         wv.fit_by_mongo(is_debug=True)
        # else:
        #     w2v_file = None

        trainset_file = os.path.join(temp_dir, 'trainset')
        vocab_file = os.path.join(temp_dir, 'vocab')
        cate_file = os.path.join(temp_dir, 'label.list')
        # if not os.path.exists(trainset_file):
        #     if sample_balanced:
        #         category = self._preprocess_sample_balanced(data_dir, trainset_file, tokenizer)
        #     else:
        #         category = self._preprocess(data_dir, trainset_file, tokenizer)
        #     tuple = sorted(category.items(), key=lambda item: item[1])
        #     label_list = [k for (k, _) in tuple]
        #     dump_json(label_list, cate_file)
        model_file = super(_SexClassifier, self).fit(trainset_file, vocab_file, w2v_file)
        return model_file, vocab_file, cate_file

class ResumeJudge(object):
    def __init__(self, model_dir, model_name):
        self.rcf = _SexClassifier(model_dir, model_name, embeding_size=20, filter_sizes=[2, 3, 4], num_filters=16)
        self.rcf.set_max_text_limit_len(5000)
        self.rcf.set_batch_size(64)
        self.rcf.set_test_batch_size(100)
        self.rcf.set_word2vec_fit_info(None, 0, 0)
        self.seg = Segment(filter_stopwords=False)

    def _tokenizer(self, text):
        return self.seg.cut(text.strip())

    def set_word2vec_fit_info(self, word2vec_file, word2vec_size, iter_num):
        self.rcf.set_word2vec_fit_info(word2vec_file, word2vec_size, iter_num)

    def set_epoches(self, epoches):
        self.rcf.set_epoches(epoches)

    def set_batch_size(self, size, test_size):
        self.rcf.set_batch_size(size)
        self.rcf.set_test_batch_size(test_size)

    def fit(self, data_dir, temp_dir, sample_balanced=False):
        return self.rcf.fit(data_dir, temp_dir, sample_balanced, self._tokenizer)

class ResumePredict(object):
    def __init__(self, model_dir, model_name):
        self.rcf = _SexClassifier(model_dir, model_name, embeding_size=20, filter_sizes=[2, 3, 4, 5], num_filters=32)
        self.rcf.set_max_text_limit_len(5000)
        self.rcf.set_batch_size(64)
        self.rcf.set_test_batch_size(100)
        self.seg = Segment()

    def _tokenizer(self, text):
        return self.seg.cut(clean_text(text))

    def set_word2vec_fit_info(self, word2vec_file, word2vec_size, iter_num):
        self.rcf.set_word2vec_fit_info(word2vec_file, word2vec_size, iter_num)

    def set_epoches(self, epoches):
        self.rcf.set_epoches(epoches)

    def set_batch_size(self, size, test_size):
        self.rcf.set_batch_size(size)
        self.rcf.set_test_batch_size(test_size)

    def fit(self, data_dir, temp_dir, sample_balanced=False):
        return self.rcf.fit(data_dir, temp_dir, sample_balanced, self._tokenizer)