import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.mongo_util import MongoUtil
mongoUtil = MongoUtil()

rawList = mongoUtil.get_all_cut_doc()

xList = []
yList = []

for (i, doc) in enumerate(rawList):
    xList.append(doc['cut_doc'])

vectorizer = TfidfVectorizer()
vectorizer.fit(xList)
x = vectorizer.transform(xList)

with open('/data/model/vectorizer.pk', 'wb') as fin:
     pickle.dump(vectorizer, fin)
