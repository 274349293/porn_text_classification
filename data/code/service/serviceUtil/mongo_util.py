# encoding=utf-8
import hashlib
from bson import ObjectId
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

    @property
    def sexKeyWordsCollection(self):
        return self._sexKeyWordsCollection

    def __init__(self):
        username = urllib.parse.quote_plus('nlp2')
        password = urllib.parse.quote_plus('TruthsoMDB2016')
        #self._client = MongoClient('mongodb://%s:%s@47.94.74.150'%(username, password), 27017, ) 
        self._client = pymongo.MongoClient('101.201.118.105')
        self._client.nlp.authenticate(username, password, mechanism='SCRAM-SHA-1')
        self._db = self._client.nlp
        self._collection = self._db.corpus_sex
        self._sexKeyWordsCollection = self._db.key_words

    def get_sex_key_words_curosr(self):
        columns = {"_id": 1}
        columns["keyWord"] = 1
        columns["level"] = 1

        return self._sexKeyWordsCollection.find({"category":'1'}, columns, no_cursor_timeout=True)

    def insert_one(self, rawDoc, userCategory, modelCategory, cutDoc):
        try:
            md5 = hashlib.md5()
            md5.update(rawDoc.encode('utf-8'))

            doc = {}
            doc['raw_doc'] = rawDoc
            doc['raw_md5'] = md5.hexdigest()
            doc['category'] = userCategory
            doc['modelCategory'] = modelCategory
            doc['reviewer'] = 'user'
            doc['cut_doc'] = cutDoc

            self._collection.insert(doc)
        except Exception as e:
            print(e)

    def updateKeyWord(self, id, keyWord, level):
        try:
            self._sexKeyWordsCollection.update({"_id" : {'$eq' : ObjectId(id)}}, {'$set' : {"keyWord": keyWord, "level": level}})
        except Exception as e:
            print(e)

    def insertKeyWord(self, keyWord, level):
        try:
            self._sexKeyWordsCollection.insert({"keyWord": keyWord, "level": level})
        except Exception as e:
            print(e)
