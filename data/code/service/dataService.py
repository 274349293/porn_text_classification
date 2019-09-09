# #coding:utf-8
from flask_cors import *
from flask import Flask, request, Response
from serviceUtil.mongo_util import MongoUtil
from multiprocessing import cpu_count
import json
from bson import ObjectId
from segment import Segment

app = Flask(__name__)
CORS(app)

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

mongoUtil = MongoUtil()
segment = Segment()

def insertIntoDb(doc, userCategory, modelCategory):
    cutDocList = segment.cut(doc)
    cutDocStr = " ".join(cutDocList)

    mongoUtil.insert_one(doc, userCategory=userCategory, modelCategory=modelCategory, cutDoc=cutDocStr)

def updateSexKeyWord(jsonData):
    id = jsonData['id']
    keyWord = jsonData['keyWord']
    level = jsonData['level']

    mongoUtil.updateKeyWord(id, keyWord, level)

def addSexKeyWordIntoDB(jsonData):
    keyWord = jsonData['keyWord']
    level = jsonData['level']

    mongoUtil.insertKeyWord(keyWord, level)

@app.route("/uploadFeedback", methods=['POST'])
def uploadFeedback():
    data = request.form.getlist('data')
    jsonData = data[0]
    jsonData = json.loads(jsonData)
    insertIntoDb(jsonData['doc'], jsonData['userClassify'], jsonData['modelClassify'])

    rt = {'code':200, 'message' : 'success'}
    return Response(json.dumps(rt),  mimetype='application/json')

@app.route("/getSexKeyWords", methods=['get'])
def getSexKeyWords():
    sexKeyWords = mongoUtil.get_sex_key_words_curosr()
    sexKeyWordsList = []
    for keyWord in sexKeyWords:
        sexKeyWordsList.append(keyWord)

    rt = {'code':200, 'message' : 'success', 'data':sexKeyWordsList}
    rt = JSONEncoder().encode(rt)

    return Response(rt,  mimetype='application/json')

@app.route("/editSexKeyWord", methods=['post'])
def editSexKeyWord():
    data = request.form.getlist('data')
    jsonData = data[0]
    jsonData = json.loads(jsonData)
    updateSexKeyWord(jsonData)

    rt = {'code':200, 'message' : 'success'}
    return Response(json.dumps(rt),  mimetype='application/json')

@app.route("/addSexKeyWord", methods=['post'])
def addSexKeyWord():
    data = request.form.getlist('data')
    jsonData = data[0]
    jsonData = json.loads(jsonData)
    addSexKeyWordIntoDB(jsonData)

    rt = {'code':200, 'message' : 'success'}
    return Response(json.dumps(rt),  mimetype='application/json')

if __name__ == '__main__':

    print("cpu count:", cpu_count())
    app.run(debug=True, port=8081, host='0.0.0.0', threaded=False, processes=cpu_count())
