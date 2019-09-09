#coding:utf-8
import multiprocessing
from utils.segment import Segment
from utils.mongo_util import MongoUtil

def cutAndUpdate(doc):
    mongoUtil = MongoUtil()
    raw_doc = doc['raw_doc']
    cut_doc = seg.cut(raw_doc)
    mongoUtil.update_one(doc['_id'], "cut_doc", " ".join(cut_doc))

if __name__ == '__main__':
    seg = Segment()

    count = 0
    pool = multiprocessing.Pool(processes=9)
    mongoUtil = MongoUtil()

    for i in mongoUtil.get_no_cut_doc_curosr():
        print(count)
        count += 1
        pool.apply(cutAndUpdate, (i, ))
