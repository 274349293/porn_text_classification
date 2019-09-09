import sys
from utils.mongo_util import MongoUtil
from urllib import parse,request

def sendRequest(textmod):
    textmod = parse.urlencode(textmod)
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    url='http://localhost:8080/check'
    req = request.Request(url='%s%s%s' % (url,'?',textmod),headers=header_dict)
    res = request.urlopen(req)
    res = res.read()
    return res


mongoUtil = MongoUtil()
total = 0
fail = 0
success = 0
totalCount = mongoUtil.get_all_raw_doc_count()

for doc in mongoUtil.get_all_raw_doc():
    total += 1
    rawDoc = doc['raw_doc']
    category = int(doc['category'])
    textmod={'p': rawDoc}
    result = int(sendRequest(textmod))
    if total % 50 == 0:
        print(total)
    if result != category:
        fail += 1
        print("expect:%d, actual:%d, doc:%s"%(category, result, rawDoc))
    else:
        success += 1

    sys.stdout.write('\r')
    sys.stdout.write("Suc:%s Err:%s Run:%s/%s %s%% |%s" %(success, fail, total, totalCount, float(total/totalCount), int(float(total/totalCount)*100)*'#'))
    sys.stdout.flush()

print("total:%d, success:%d, fail:%d, acc:%f"%(total, success, fail, success/total))