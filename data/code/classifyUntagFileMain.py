import multiprocessing
import pickle
from truthUtils.mongo_util import MongoUtil
from sklearn.externals import joblib
from truthUtils.segment import Segment


def classifyAndUpdate(doc):
    mongoUtil = MongoUtil()
    rawDoc = doc['raw_doc']
    keyWordList = seg.cut(rawDoc)
    keyWord = " ".join(keyWordList)
    keyWordDoc = [keyWord]
    x = vectorizer.transform(keyWordDoc)
    x = x.toarray()

    predictResult = sexClassifyModel.predict(x)
    mongoUtil.update_category(doc['_id'], predictResult[0])
    print(predictResult[0], "   ", doc['_id'])

if __name__ == '__main__':
    mongoUtil = MongoUtil()
    untagDoc = mongoUtil.get_untag_doc_curosr()
    tfidfModelFile = '/data/model/stable/vectorizer.pk'
    pklFile = open(tfidfModelFile, 'rb')
    vectorizer = pickle.load(pklFile)
    sexClassifyModel = joblib.load('/data/model/stable/sexClassify.model')
    seg = Segment()
    seg = Segment()
    pool = multiprocessing.Pool(processes=16)

    for index, doc in enumerate(untagDoc):
        print(index)
        pool.apply_async(classifyAndUpdate, (doc, ))

    pool.close()
    pool.join()



