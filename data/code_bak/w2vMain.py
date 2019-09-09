#coding:utf-8

from train.word2vec import W2V as wv_train
from gensim.models.word2vec import Word2Vec

if __name__ == '__main__':
    model_file = '/data/model/wordvec.model'
    wv = wv_train(model_file)
    wv.fit_by_mongo()

    #词向量的简单使用
    loadWv = Word2Vec.load(model_file)
    print(loadWv.wv.most_similar('屄'))
    print(loadWv.wv['屄'])
