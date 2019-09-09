from flask import Flask, request
import requests, json
import sys

from flask_cors import *

app = Flask(__name__)
CORS(app)

def excepthook(type, value, trace):
    # print('Unhandled Error: %s: %s %s'%(str(type), str(value), str(trace)))
    sys.__excepthook__(type, value, trace)

@app.route("/", methods=['GET'])
def getKeyWordsAPI():
    keyWord = request.args.get('p')
    if keyWord == '0':
        raise RuntimeError
    result = {}
    result['code'] = 200
    result['message'] = 'success'
    jsonResult = json.dumps(result, ensure_ascii=False)
    return jsonResult

if __name__ == '__main__':
    sys.excepthook = excepthook

    app.run(debug=True, port=8081, host='0.0.0.0', threaded=False, processes=0)

#  uploadInfo = {'doc':'中文的色情内容测试','userCategory':0,'modelCategory':0.6}
# jsondata=json.dumps(uploadInfo)
# print(jsondata)
#
# headers = {'content-type': 'application/json'}
#
# r = requests.post("http://47.94.191.54:8081/uploadFeedback", data=jsondata, headers=headers)
# print(r.headers)
# print(r.json())

#
# sexKeyWord = {'id':'5a40ce8b76d77a8dee0f031a', 'keyWord':'龟头', 'level':3}
# jsondata=json.dumps(sexKeyWord)
# print(jsondata)
#
# headers = {'content-type': 'application/json'}
#
# r = requests.post("http://47.94.191.54:8081/editSexKeyWord", data=jsondata, headers=headers)
# print(r.headers)
# print(r.json())


# sexKeyWord = {'keyWord':'菊花', 'level':3}
# jsondata=json.dumps(sexKeyWord)
# print(jsondata)
#
# headers = {'content-type': 'application/json'}
#
# r = requests.post("http://47.94.191.54:8081/addSexKeyWord", data=jsondata, headers=headers)
# print(r.headers)
# print(r.json())
