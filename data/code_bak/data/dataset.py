from sklearn.model_selection import train_test_split
import time

class dataSet():
    def __init__(self, mongoUtil):
        sexRawList = mongoUtil.get_all_cut_sex_doc()
        normarRawList = mongoUtil.get_all_cut_normal_doc(0)

        self.sexXList = []
        self.sexYList = []

        for (i, doc) in enumerate(sexRawList):
            self.sexXList.append(doc['cut_doc'])
            self.sexYList.append(doc['category'])

        self.normalXList = []
        self.normalYList = []
        for (i, doc) in enumerate(normarRawList):
            self.normalXList.append(doc['cut_doc'])
            self.normalYList.append(doc['category'])

    def iter_minibatches(self, minibatch_size=1000):
        X = []
        y = []

        X.extend(self.sexXList)
        y.extend(self.sexYList)

        # X_train, X_test, y_train, y_test
        _, tempNormalX, _, tempNormalY = train_test_split(self.normalXList, self.normalYList, test_size=0.9, random_state=int(time.time()%100))

        X.extend(tempNormalX[:len(self.sexXList)])
        y.extend(tempNormalY[:len(self.sexXList)])

        _, tempNormalX, _, tempNormalY = train_test_split(X, y, test_size=0.9, random_state=int(time.time()%100))

        yield tempNormalX[:minibatch_size],tempNormalY[:minibatch_size]
