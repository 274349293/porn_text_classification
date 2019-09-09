import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from truthUtils.mongo_util import MongoUtil
mongoUtil = MongoUtil()

rawList = mongoUtil.get_all_cut_doc()

xList = []

for (i, doc) in enumerate(rawList):
    xList.append(doc['cut_doc'])

# vectorizer = TfidfVectorizer(max_features=200000)
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(xList)
print(x.shape)

with open('/data/model/vectorizer.pk', 'wb') as fin:
     pickle.dump(vectorizer, fin)
