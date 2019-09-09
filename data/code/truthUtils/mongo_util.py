# encoding=utf-8
from bson import ObjectId
from pymongo.mongo_client import MongoClient
import threading
import urllib.parse

class MongoUtil(object):
    @property
    def client(self):
        return self._client

    @property
    def db(self):
        return self._db

    @property
    def collection(self):
        return self._collection

    doc_array = []
    lock = threading.Lock()

    def __init__(self):
        username = urllib.parse.quote_plus('root')
        password = urllib.parse.quote_plus('Truthso#MDB@2016')

        # self._client = MongoClient('mongodb://%s:%s@dds-2zed3cf7c1b18a642242-pub.mongodb.rds.aliyuncs.com'%(username, password), 3717, )
        self._client = MongoClient('mongodb://%s:%s@dds-2zed3cf7c1b18a641.mongodb.rds.aliyuncs.com'%(username, password), 3717, )
        self._db = self._client.nlp
        self._collection = self._db.corpus_sex
        self._client.au

    def find_one(self):
        return self._collection.find_one()

    def find(self):
        return self._collection.find(no_cursor_timeout=True)

    def get_untag_doc_curosr(self):
        columns = {"_id": 1}
        columns["raw_doc"] = 1

        mongo_filter = {}
        mongo_filter["category"] = {"$eq": ''}

        return self._collection.find(mongo_filter, columns, no_cursor_timeout=True)

    def update_category(self, id, category):
        self._collection.update({'_id': ObjectId(id)}, {'$set': {'category': category}})

    def get_no_cut_doc_curosr(self):
        columns = {"_id": 1}
        columns["raw_doc"] = 1

        mongo_filter = {}
        mongo_filter["cut_doc"] = {"$exists": False}

        return self._collection.find(mongo_filter, columns, no_cursor_timeout=True)

    def get_all_cut_doc(self):
        columns = {"_id": 1}
        columns["cut_doc"] = 1
        columns["category"] = 1

        mongo_filters = []
        mongo_filter = {}
        mongo_filter["cut_doc"] = {"$exists": True}
        mongo_filters.append(mongo_filter)

        mongo_filter = {}
        mongo_filter["category"] = {"$ne": ''}
        mongo_filters.append(mongo_filter)

        mongo_filter = {}
        mongo_filter["$and"] = mongo_filters

        return self._collection.find(mongo_filter, columns, no_cursor_timeout=True)


    def get_all_raw_doc(self):
        columns = {"_id": 1}
        columns["raw_doc"] = 1
        columns["category"] = 1

        mongo_filter = {}
        mongo_filter["category"] = {"$ne": ''}

        return self._collection.find(mongo_filter, columns, no_cursor_timeout=True)

    def get_all_raw_doc_count(self):
        return self._collection.find(None, None).count()

    def get_all_cut_normal_doc(self, count):
        columns = {"_id": 1}
        columns["cut_doc"] = 1
        columns["category"] = 1

        mongo_filters = []
        mongo_filter = {}
        mongo_filter["cut_doc"] = {"$exists": True}
        mongo_filters.append(mongo_filter)

        mongo_filter = {}
        mongo_filter["category"] = {"$eq": '0'}
        mongo_filters.append(mongo_filter)

        mongo_filter = {}
        mongo_filter["$and"] = mongo_filters

        if count == 0:
            return self._collection.find(mongo_filter, columns, no_cursor_timeout=True)
        else:
            return self._collection.find(mongo_filter, columns, no_cursor_timeout=True).limit(count)


    def get_all_cut_sex_doc(self):
        columns = {"_id": 1}
        columns["cut_doc"] = 1
        columns["category"] = 1

        mongo_filters = []
        mongo_filter = {}
        mongo_filter["cut_doc"] = {"$exists": True}
        mongo_filters.append(mongo_filter)

        mongo_filter = {}
        mongo_filter["category"] = {"$eq": '1'}
        mongo_filters.append(mongo_filter)

        mongo_filter = {}
        mongo_filter["$and"] = mongo_filters

        return self._collection.find(mongo_filter, columns, no_cursor_timeout=True)

    def get_all_cut_user_tag_doc(self):
        columns = {"_id": 1}
        columns["cut_doc"] = 1
        columns["category"] = 1

        mongo_filters = []
        mongo_filter = {}
        mongo_filter["cut_doc"] = {"$exists": True}
        mongo_filters.append(mongo_filter)

        mongo_filter = {}
        mongo_filter["reviewer"] = {"$eq": 'user'}
        mongo_filters.append(mongo_filter)

        mongo_filter = {}
        mongo_filter["$and"] = mongo_filters

        return self._collection.find(mongo_filter, columns, no_cursor_timeout=True)

    def insert_one(self, doc):
        try:
            self._collection.insert(doc)
        except Exception as e:
            print(e)

    def update_one(self, id, key, value):
        self._collection.update({"_id" : {'$eq' : id}}, {'$set' : {key : value}})
