import random
import numpy as np

class dataSet():
    def __init__(self, mongoUtil):
        # 获取所有色情文章
        self.allSexDocList = []
        allSexDocCursor = mongoUtil.get_all_cut_sex_doc()
        self.sexDocCount = allSexDocCursor.count()
        for sexDoc in allSexDocCursor:
            self.allSexDocList.append((sexDoc['cut_doc'], sexDoc['category']))

        # 获取所有用户标注的文章
        self.allUserTagDocList = []
        allUserTagDocCursor = mongoUtil.get_all_cut_user_tag_doc()
        for userTagDoc in allUserTagDocCursor:
            self.allUserTagDocList.append((userTagDoc['cut_doc'], userTagDoc['category']))

        # 获取所有正常文档
        self.allNormalDocList = []
        allNormalDocCursor = mongoUtil.get_all_cut_normal_doc(0)
        for normalDoc in allNormalDocCursor:
            self.allNormalDocList.append((normalDoc['cut_doc'], normalDoc['category']))


    def reshuffle(self):
        # 组合在一起
        self.rawX = []
        self.rawY = []
        for doc in self.allSexDocList:
            self.rawX.append(doc[0])
            self.rawY.append(doc[1])
        for doc in self.allUserTagDocList:
            self.rawX.append(doc[0])
            self.rawY.append(doc[1])

        # 打乱正常文本顺序
        random.shuffle(self.allNormalDocList)

        # 按照色情文本的数量 添加正常文本
        for index, doc in enumerate(self.allNormalDocList):
            self.rawX.append(doc[0])
            self.rawY.append(doc[1])
            if index > self.sexDocCount:
                break

        # 组合起来
        self.rawXArray = np.array(self.rawX)
        self.rawYArray = np.array(self.rawY)

        self.rawXYArray = np.c_[self.rawXArray, self.rawYArray]
        self.total = self.rawXYArray.shape[0]
        np.random.shuffle(self.rawXYArray)

    def iter_mini_batches(self, mini_batch_size=1000):
        self.totalSteps = int(self.total/mini_batch_size)

        for self.currentStep in range(self.totalSteps):
            start = self.currentStep * mini_batch_size
            end = (self.currentStep + 1) * mini_batch_size
            tempArray = self.rawXYArray[start : end]
            yield tempArray.T[0], tempArray.T[1]
