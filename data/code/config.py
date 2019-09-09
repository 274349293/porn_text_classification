#coding:utf-8
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

STOPWORDS_DIR = os.path.join(DATA_DIR, 'stopwords')
VOCAB_FILE = os.path.join(DATA_DIR, 'vocab', 'vocab.dat')
SEGMENT_DIR = os.path.join(DATA_DIR, 'userdict')

VOCAB_FILE = os.path.join(DATA_DIR, 'vocab', 'vocab_0.0.1')
TFIDF_FILE = os.path.join(DATA_DIR, 'tfidf', 'tfidf_0.0.1')

# resume目录包含:
#   resume_0.0.1: 版本文件
#   judge.pb: 判断简历文件二分类模型
#   predict.pb: 预测简历行业多分类模型
#   judge.vocab: 判别模型词汇表
#   predict.vocab:预测模型的词汇表
#   predict.label:简历行业类别列表
RESUME_MODEL_FILE = os.path.join(DATA_DIR, 'resume', 'resume_0.0.1')

# word2vec目录包含:
#   word2vec: 完整的词向量模型
#   wx50: 微信50dim向量
#   wx100: 微信100dim向量
#   wx256: 微信256dim向量
#   wps50: wps50dim向量
#   wps100: wps100dim向量
#   wps256: wps256dim向量
_WORD2VEC_DIR = os.path.join(DATA_DIR, 'word2vec')
WORD2VEC_FILES = {
    'word2vec': os.path.join(_WORD2VEC_DIR, 'word2vec', 'word2vec_0.0.1'),
    'wx50': os.path.join(_WORD2VEC_DIR, 'wx50', 'wx50_0.0.1'),
    'wx100': os.path.join(_WORD2VEC_DIR, 'wx100', 'wx100_0.0.1'),
    'wx256': os.path.join(_WORD2VEC_DIR, 'wx256', 'wx256_0.0.1'),
    'wps50': os.path.join(_WORD2VEC_DIR, 'wps50', 'wps50_0.0.1'),
    'wps100': os.path.join(_WORD2VEC_DIR, 'wps100', 'wps100_0.0.1'),
    'wps256': os.path.join(_WORD2VEC_DIR, 'wps256', 'wps256_0.0.1'),
    'news20': os.path.join(_WORD2VEC_DIR, 'news20', 'news20_0.0.1')
}

# doc2vec目录包含:
#   newstitle20: 新闻标题20dim
#   fname20: wps文档名20dim
#   urltitle20: url标题20dim
_DOC2VEC_DIR = os.path.join(DATA_DIR, 'doc2vec')
DOC2VEC_FILES = {
    'newstitle20': os.path.join(_DOC2VEC_DIR, 'newstitle20', 'newstitle20_0.0.1'),
    'fname20': os.path.join(_DOC2VEC_DIR, 'fname20', 'fname20_0.0.1'),
    'urltitle20': os.path.join(_DOC2VEC_DIR, 'urltitle20', 'urltitle20_0.0.1'),
}

# cluster目录包含:
# words: 词语聚类特征模型
# shorttext: 短文本向量聚类特征模型
_CLUSTER_DIR = os.path.join(DATA_DIR, 'cluster')
CLUSTER_FILES = {
    'words': os.path.join(_CLUSTER_DIR, 'words', 'words_0.0.1'),
    'shorttext': os.path.join(_CLUSTER_DIR, 'shorttext', 'shorttext_0.0.1')
}

# newword目录:
#   newword_0.0.1: 版本文件
#   ne.scaler: 数据正则化模型
#   ne.model: 邻接熵模型
#   ne.newword: 新词识别模型
NEWWORD_MODEL_FILE = os.path.join(DATA_DIR, 'newword', 'newword_0.0.1')

# shortext目录:
#   fname2vec: wps文件名向量化
#       fname2vec_0.0.1: 版本
#       fname2vec.encoder: 编码器模型
#       fname2vec.vocab: 词汇表
#       fname2vec.meta: 元数据文件

#   news2vec: 新闻标题向量化
#       news2vec_0.0.1: 版本
#       news2vec.encoder: 编码器模型
#       news2vec.vocab: 词汇表
#       news2vec.meta: 元数据文件
_SHORTEXT_DIR = os.path.join(DATA_DIR, 'shortext')
SHORTEXT_FILES = {
    'fname2vec': os.path.join(_SHORTEXT_DIR, 'fname2vec', 'fname2vec_0.0.1'),
    'news2vec': os.path.join(_SHORTEXT_DIR, 'news2vec', 'news2vec_0.0.1'),
    'url2vec': os.path.join(_SHORTEXT_DIR, 'url2vec', 'url2vec_0.0.1')
}

def get_download_url(name, version=None):
    DATA_DEFAULT_URL = 'http://120.92.16.247:8001/model_url/?name='
    url = DATA_DEFAULT_URL + name
    if version:
        url += '&version=' + version
    return url

def get_model_datainfo(data_path):
    (dirname, fname) = os.path.split(data_path)
    (name, version) = fname.split('_')
    return dirname, name, version
