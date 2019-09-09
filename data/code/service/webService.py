# #coding:utf-8
import re
import urllib
from flask_cors import *
import pickle
from flask import Flask, request
from gensim.models import Word2Vec
from sklearn.externals import joblib
from segment import Segment
from serviceUtil.mongo_util import MongoUtil
from multiprocessing import cpu_count
import logging

import re
import json
import sys

app = Flask(__name__)
CORS(app)

mongoUtil = MongoUtil()
modelFile = '/data/model/stable/wordvec.model'
wv = Word2Vec.load(modelFile)
tfidfModelFile = '/data/model/stable/vectorizer.pk'
pklFile = open(tfidfModelFile, 'rb')
vectorizer = pickle.load(pklFile)
sexClassifyModel = joblib.load('/data/model/stable/sexClassify.model')
seg = Segment()

@app.route("/getKeyWords", methods=['GET'])
def getKeyWordsAPI():
    keyWordSet = getSexKeyWords()

    result = {}
    result['code'] = 200
    result['message'] = 'success'
    result['data'] = list(keyWordSet)
    jsonResult = json.dumps(result, ensure_ascii=False)
    return jsonResult


@app.route("/", methods=['GET'])
def hello():
    keyWord = request.args.get('p')
    result = {}

    try:
        resultList = wv.wv.most_similar(keyWord, topn=20)
    except Exception as e:
        result['code'] = 201
        result['message'] = "该词不存在"
        jsonResult = json.dumps(result, ensure_ascii=False)
        return jsonResult

    result['code'] = 200
    result['message'] = 'success'
    result['data'] = resultList
    jsonResult = json.dumps(result, ensure_ascii=False)
    return jsonResult

@app.route("/check", methods=['GET'])
def check():
    keyWord = request.args.get('p')
    keyWordList = seg.cut(keyWord)
    keyWord = " ".join(keyWordList)
    keyWordDoc = [keyWord]
    x = vectorizer.transform(keyWordDoc)
    x = x.toarray()

    predictResult = sexClassifyModel.predict(x)
    return predictResult[0]


def getSexKeyWords():
    sexKeyWords = mongoUtil.get_sex_key_words_curosr()
    sexKeyWordsList = []
    sexKeyWordsLevelMap = {}
    for keyWord in sexKeyWords:
        sexKeyWordsList.append(keyWord['keyWord'])
        sexKeyWordsLevelMap[keyWord['keyWord']] = keyWord['level']

    for keyWord in sexKeyWordsList:
        level = sexKeyWordsLevelMap[keyWord]
        try:
            relativeKeyWordList = wv.wv.most_similar(keyWord, topn=40)
        except Exception as e:
            continue
        for result in relativeKeyWordList:
            if result[1] >= 0.9:
                sexKeyWordsLevelMap[result[0]] = level

    return sexKeyWordsLevelMap

sexKeyWords = getSexKeyWords()
app.logger.info("数据库+word2vec联想出来的关键词:", len(sexKeyWords))

def getDocSexProba(cutDocList):
    predictProbaResult = [1.0, 0.0]
    if len(cutDocList) > 0:
        cutDoc = " ".join(cutDocList)
        cutDocArray = [cutDoc]
        x = vectorizer.transform(cutDocArray)
        x = x.toarray()
        predictResult = int(sexClassifyModel.predict(x)[0])
        # 返回结果是一个array array，第一个array是文本数组，第二个array是该文本每个类型的概率
        predictProbaResult = sexClassifyModel.predict_proba(x)[0]
        app.logger.info("色情分类概率是：%f, 模型分类结果：%d, 分词结果：%s", predictProbaResult[1], predictResult, cutDoc)
    return int(predictProbaResult[1]*10)/10


def getParagraphList(doc):
    # 按照这几类符号进行文本切割，然后根据长度进行组合
    separator = '(。|！|？|\n)'
    paragraphList = re.split(separator, doc)
    pLen = len(paragraphList)

    start = 0
    returnList = []
    currentPara = ""
    paragraphMaxLen = 200
    while start < pLen:
        # 如果当前组合出来的段落还没有达到长度上限，加入下一部分
        if len(currentPara) < paragraphMaxLen and start < pLen:
            currentPara += paragraphList[start]
            start += 1
        if len(currentPara) >= paragraphMaxLen or start >= pLen:
            # 如果已经超过了长度上限，将符号加入到段落尾部
            # 加入之前先判断一下长度是否正确
            if start < pLen - 1:
                currentPara += paragraphList[start]
                start += 1
            # 然后将这一段落，加入到整体的list中
            returnList.append(currentPara)
            # 将当前段落置空
            currentPara = ""

    app.logger.info("分段长度为：%d", len(returnList))
    return returnList

posConvert = {}
posConvert['Ag']= u'形语素'
posConvert['a']= u'形容词'
posConvert['ad']= u'副形词'
posConvert['an']= u'名形词'
posConvert['b']= u'区别词'
posConvert['c']= u'连词'
posConvert['dg']= u'副语素'
posConvert['d']= u'副词'
posConvert['e']= u'叹词'
posConvert['f']= u'方位词'
posConvert['g']= u'语素'
posConvert['h']= u'前接成分'
posConvert['i']= u'成语'
posConvert['j']= u'简称略语'
posConvert['k']= u'后接成分'
posConvert['l']= u'习用语'
posConvert['m']= u'数词'
posConvert['Ng']= u'名语素'
posConvert['n']= u'名词'
posConvert['nr']= u'人名'
posConvert['ns']= u'地名'
posConvert['nt']= u'机构团体'
posConvert['nz']= u'其他专名'
posConvert['o']= u'拟声词'
posConvert['p']= u'介词'
posConvert['q']= u'量词'
posConvert['r']= u'代词'
posConvert['s']= u'处所词'
posConvert['tg']= u'时语素'
posConvert['t']= u'时间词'
posConvert['u']= u'助词'
posConvert['vg']= u'动语素'
posConvert['v']= u'动词'
posConvert['vd']= u'副动词'
posConvert['vn']= u'名动词'
posConvert['w']= u'标点符号'
posConvert['x']= u'非语素字'
posConvert['y']= u'语气词'
posConvert['z']= u'状态词'
posConvert['eng']= u'英文'
posConvert['un']= u'未知词'
posConvert['uj']= u'未知词'
posConvert['ul']= u'未知词'

@app.route("/istext", methods=['GET', 'POST'])
def istext():
    result = {}
    fullDoc = ""
    if request.method =='GET':
        fullDoc = request.args.get('istext')
    elif request.method =='POST':
        fullDoc = request.form.get('istext')

    app.logger.info("用户上传文本: %s", fullDoc)
    cutDocList = seg.cut(fullDoc)
    app.logger.info("分词之后长度: %d", len(cutDocList))
    if len(cutDocList) == 0:
        predictSexProba = 0
    else:
        predictSexProba = getDocSexProba(cutDocList)

    result['code'] = 200
    result['message'] = 'Success'

    # 文本分段
    paragraphs = []
    paragraphList = getParagraphList(fullDoc)
    maxParagraphProba = 0
    for partParagraph in paragraphList:
        words = []
        cutParagraph = seg.cut(partParagraph)
        sexParagraphProba = getDocSexProba(cutParagraph)
        maxParagraphProba = max(maxParagraphProba, sexParagraphProba)
        if sexParagraphProba < 0.5:
            continue

        isCutParagraphHasSexKeyWord = False
        for keyWord in cutParagraph:
            if sexKeyWords.keys().__contains__(keyWord):
                isCutParagraphHasSexKeyWord = True

        if isCutParagraphHasSexKeyWord == False:
            continue

        # 1. 使用|对色情关键字进行分割
        # 2. 使用()将整个表达式包括起来，这样可以保留所有的关键字
        separator = '('+'|'.join(sexKeyWords.keys())+')'
        returnWords = re.split(separator, partParagraph)

        paragraph = {}
        if len(returnWords) > 0:
            isSexParagraph = False
            for keyWord in returnWords:
                word = {}
                if sexKeyWords.keys().__contains__(keyWord):
                    word['weight'] = sexKeyWords[keyWord]
                    isSexParagraph = True
                else:
                    word['weight'] = 0
                word['word'] = keyWord
                words.append(word)

            # 该分段中，需要有色情关键字才可以被加入
            if isSexParagraph:
                paragraph['negative'] = sexParagraphProba
                paragraph['words'] = words

        # 该分段中，需要有色情关键字才可以被加入
        if len(paragraph) > 0:
            paragraphs.append(paragraph)

    paragraphs.sort(key=lambda k: k['negative'], reverse=True)
    datas = {}
    datas['paragraphs'] = paragraphs

    article = {}
    # 如果没有段落是被检查出来有嫌疑的，将分值折半
    if len(paragraphs) == 0:
        article['negative'] = predictSexProba/2
    else:
        # 根据雅琪的要求，不管文章篇幅有多大，只要有一处有问题，就得取最高分 来显示整体评分
        if maxParagraphProba > predictSexProba:
            article['negative'] = maxParagraphProba
        else:
            article['negative'] = predictSexProba

    results = []
    cutDocWithPartOfSpeech = seg.cut(fullDoc, pos=True)
    posCount = {}
    for pos in cutDocWithPartOfSpeech:
        if not posConvert.__contains__(pos[1]):
            posType = posConvert['un']
        else:
            posType = posConvert[pos[1]]

        if posCount.__contains__(posType):
            tempCount = posCount[posType]
            posCount[posType] = tempCount+1
        else:
            posCount[posType] = 1

        tempResult = {}
        tempResult['type'] = posType
        tempResult['value'] = pos[0]
        results.append(tempResult)
    article['results'] = results

    partOfSpeechList = []
    for key in posCount.keys():
        partOfSpeech = {}
        partOfSpeech['num'] = posCount[key]
        partOfSpeech['type'] = key
        partOfSpeechList.append(partOfSpeech)
    article['partOfSpeech'] = partOfSpeechList

    datas['article'] = article

    result['datas'] = datas
    jsonResult = json.dumps(result, ensure_ascii=False)
    return jsonResult

def excepthook(type, value, trace):
    app.logger.error("全局捕获异常: %s: %s %s"%(str(type), str(value), str(trace)))
    sys.__excepthook__(type, value, trace)

if __name__ == '__main__':
    sys.excepthook = excepthook
    handler = logging.handlers.TimedRotatingFileHandler("/data/log/webService", 'midnight', 1, 0, encoding='UTF-8')
    # 设置后缀名称，跟strftime的格式一样
    handler.suffix = "%Y%m%d.log"
    handler.setLevel(logging.DEBUG)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)

    print("cpu count:", cpu_count())
    app.run(debug=True, port=8080, host='0.0.0.0', threaded=False, processes=cpu_count())
