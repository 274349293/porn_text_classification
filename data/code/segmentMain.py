#coding:utf-8
import multiprocessing
from truthUtils.segment import Segment
from truthUtils.mongo_util import MongoUtil

def cutAndUpdate(doc):
    mongoUtil = MongoUtil()
    raw_doc = doc['raw_doc']
    cut_doc = seg.cut(raw_doc)
    mongoUtil.update_one(doc['_id'], "cut_doc", " ".join(cut_doc))

if __name__ == '__main__':
    seg = Segment()

    pool = multiprocessing.Pool(processes=16)

    mongoUtil = MongoUtil()
    for index, doc in enumerate(mongoUtil.get_no_cut_doc_curosr()):
        if index % 100 == 0:
            print(index)
        pool.apply_async(cutAndUpdate, (doc, ))

    pool.close()
    pool.join()

