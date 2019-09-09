import pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from data import dataset
from data.dataset import dataSet
from utils.mongo_util import MongoUtil
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np
from sklearn.naive_bayes import BernoulliNB

mongoUtil = MongoUtil()

tfidfModelFile = '/data/model/vectorizer.pk'
pklFile = open(tfidfModelFile, 'rb')
vectorizer = pickle.load(pklFile)

# gnb = GaussianNB()
# gnb = BernoulliNB()  效果很差
gnb = MultinomialNB()

dataset = dataSet(mongoUtil)

for epoch in range(300):
    first=True
    batch = 0
    data = dataset.iter_minibatches(minibatch_size=2000)
    for (x, y) in data:
        x = vectorizer.transform(x)
        x = x.toarray()

        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=epoch)

        if first:
            model = gnb.partial_fit(xTrain, yTrain, np.unique(yTrain))
            first = False
        else:
            model = gnb.partial_fit(xTrain, yTrain)

        acc = gnb.score(xTest, yTest)
        print("epoch:%d  score:%f"%(epoch, acc))

        joblib.dump(model, '/data/model/sexClassify.model_%d_%d_%f'%(epoch, batch, acc*100))
        batch += 1
