# encoding=utf-8
import hashlib
from pymongo.mongo_client import MongoClient
import urllib.parse
import pymongo

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

    def __init__(self):
        username = urllib.parse.quote_plus('nlp')
        password = urllib.parse.quote_plus('TruthsoMDB2016')
        self._client = MongoClient('mongodb://%s:%s@101.201.118.105'%(username, password), 27017, )
        self._db = self._client.nlp
        self._collection = self._db.corpus_sex

    def insert_one(self, rawDoc, userCategory, modelCategory):
        try:
            md5 = hashlib.md5()
            md5.update(rawDoc)

            doc = []
            doc['raw_doc'] = rawDoc
            doc['raw_md5'] = md5.hexdigest()
            doc['category'] = userCategory
            doc['modelCategory'] = modelCategory
            doc['reviewer'] = 'user'

            self._collection.insert(doc)
        except Exception as e:
            print(e)
