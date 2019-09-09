# #coding:utf-8
import re
from flask_cors import *
import pickle
from flask import Flask, request
from gensim.models import Word2Vec
from sklearn.externals import joblib
# from segment import Segment
import logging

from ..truthUtils.segment import Segment

app = Flask(__name__)
CORS(app)

modelFile = '/data/model/stable/wordvec.model'
wv = Word2Vec.load(modelFile)

tfidfModelFile = '/data/model/stable/vectorizer.pk'
pklFile = open(tfidfModelFile, 'rb')
vectorizer = pickle.load(pklFile)

sexClassifyModel = joblib.load('/data/model/stable/sexClassify.model')

seg = Segment()
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


keyWords="玉腿,玉杵,椒乳,酥腻,鸡头肉,玉臀,肉菇,小乳,yáng具,浪叫，xiāo穴，胴体，肉吊,马眼,龙茎,肉冠,乳球,下身,花蕊,incest,美少妇,色情影片,狼友,幼交,奸杀,一本道,前戏,口活,爆草,性交,淫荡,迷情粉,陰唇,精子,乳爆,嫖娼,下体,与狗性,破处,欠干,妹疼,赤裸,凌辱,fuck,摸奶,禁脔,性虐待,性福,爱液,吸精,精液,淫荡,风骚,母奸,阉割,淫騷妹,口交,强奸,要射了,急需嫖,擦你妈,阴阜,轮奸,嫖鸡,充气娃,母子乱伦,洞小口紧,黄色小电影,看JJ,性交视频,肉具,sexinsex,捏弄,我操,巨奶,干你妈,露b,干自己,婊子养的,肛交,一夜欢,暴干,淫水,干你娘,淫魔舞,口射,屁眼,胸器,大sb,secom,淫妇,大鸡巴,呻吟,峰壁上咬,淫贱,性爱,嫖雞,色电影,逼奸,h动画,骆冰淫传,淫教师,色小说,色视频,浪逼,yin荡,嫖客,a级情片,爽片,阳具,阴部,肉欲,嫖妓,肉穴,爆乳,色欲,插阴,做爱,暴乳,性饥渴,性技巧,玉女心经,成人网站,淫母,艹你,屄,挨了一炮,嫩逼,浪穴,欲仙欲死,淫靡,坐交,操黑,强暴,换妻俱乐部,干你,迷奸,乳峰,催情药,缩阴,阴户,色情小说,屌,粉嫩的小沟,集体淫,沙比,强jian,裸露,淫液,成人电影,粉穴,买淫,谜奸药,色区,暴淫,鸡奸,人妻,丝袜网,淫糜,桜田,丝诱,菊花洞,包办色情娱乐服务,肉缝,贱比,食精,国产av,色情图片,江湖淫娘,乖乖粉,鸡巴,奶子,淫色漫画,gay片,狗草,女優,性感妖娆,乱交,校鸡,奶头吃起来,穴图,少妇,失身,淫照,日你妈,操嫂子,小逼,肏死,犬交,amateur,裸聊,豪乳,死逼,失身水,奶头,人兽,迷昏药,大b,拳交,潮吹,颜射,自慰,下身一挺,肉茎,嫩阴,内裤,揉乳,操逼,潮喷,美穴,xing伴侣,酥痒,要泄了,裸照,姐疼,小穴,身体里抽送,迷奸粉,肉唇,玉体,包你射,操你全家,浪女,插你,性奴,淫亵,阴茎增大,暴菊,迷情水,骚浪,熟妇,挺起的命根子,狂操,抓胸,淫声浪语,插我,操我,和狗做,淫虫,骚货,狗操,性福情,美艳少妇,淫兽学,淫浪,煞笔,日烂,迷魂药,漏乳,卖比,豪乳,乳交,液体炸,后穴,g片,狂插,处男,兽性,骚比,迷奸药,媛交,煞逼,乱奸,淫虐,玉乳,色界,按摩棒,我日你,叫床,狗娘养,cao你,卖淫女,干穴,喷精,淫术炼金士,淫兽,浪叫,小骚货,阴道,淫穴,舔阴,蜜液,蜜穴,肥逼,奸情,肛门,操你妈,幼女,清純壆,狗日的,男奴,色情片,口爆,骚穴,大雞巴,激情小电影,淫情,阴蒂,大波,插屁屁,美逼,tokyohot,迷昏藥,轮操,阴茎,骚女,激情炮,卖春,性交图片,大乳,弟痛,抽插,妹按摩,成人卡通,根浴,淫样,情色,射爽,morphine,中年美妇,丝袜妹,荡妇,兽欲,激情短,兩性淫亂,欲火,星野桃,熟母,妈逼,爽死我了,a片,春药,色情服务,欲女,摸胸,援交,应召,惹火身材,迷药,被操,乳罩解开,擠乳汁,脚交,淫叫,群交,狗杂种,奇淫散,骚妇,sm女王,g点,亚情人人,发浪,作爱,猛男,放尿,熟女,插暴,妹痛,吃精,迷情药,体奸,光着身子,炮友,裸聊,虎骑,插b,鸳鸯洗,色盟,阴b,贱b,援助交际,和狗性,hardcore,風花,干死你,激情电,浓精,奸成瘾,巨乳,几吧,脱内裤,诱奸,幼男,解开乳罩,sm,情趣用品,阴茎助勃,肏你,淫情女,买春,操了嫂,小xue,手淫,阴核,女伟哥,h动漫,乳头,下体,浑圆,淫肉,淫威,嫩穴,骚水,殖器护,叫春,招鸡,全裸,淫女,胸推,肉棒,砲友,强奸处女,兽交,暴奸,体位,操死,牝户,乱伦,嫩女,精子,porn,淫书,淫娃,浪妇,性奴集中营,偷欢,双臀,伦理电影,爆你菊,护士的胸,裙中性运动,艳丽片,成人视,失身药,淫乱,迷魂藥,轮暴,滚圆大乳,透视药,美乳,释欲,流淫,喷尿,原味内衣,肉洞,平井,淫媚,成人聊,开苞,禁网,我就色,骚逼,迷魂香,性虐,傻b,无码,淫电影,黑逼,日逼,操烂,阴唇,穴口,挤乳汁,后庭,色诱,做鸭,插比,赛后骚,内射,艳情小说,好嫩,性伴侣,夜激情,肉棍,深喉冰,陰戶,密穴,傻比,金澤文子,性伙伴,荡女,相奸,操他妈,菊门,色情电影,宮下杏奈,爆菊,多人轮,淫河,三秒射,色逼,淫荡视频,招妓,大力抽送,妓女,裸陪,高潮,性感少,柔胸粉,秘唇,色b,巨屌,傻逼,做鸡,黄片,成人游戏,抽一插,爱液横流,和狗交,淫魔,进入我的身体,妹上门,菊穴,乱伦,被干,摇头丸,射颜,菊暴,扌由插,玉穴,兽奸,陰道,骚嘴,学生妹,刹笔,催情藥,射精,口淫,迷幻藥,淫妻,性欲,推开胸罩,迷藥,美幼,装b,櫻井,阴毛,鸡吧,被插,搞媛交,色色,丝袜恋,操你娘,叫鸡,毛片,成人论坛,色情表演,阴间来电,插逼,阴精,淫色,肉逼,卖逼,淫兽学园,色猫,姐包夜,乳沟,性器,掰穴,爆菊,爆菊花,屄,逼奸,逼痒,操b,操屄,操逼,插逼,插你,潮吹,潮喷,吃精,耻毛,大屌,多人性爱,肥逼,肥穴,粉穴,干炮,干屁眼,肛奸,肛肉,含屌,花逼,鸡巴,菊花洞,抠穴,口爆,口射,浪穴,老汉推车,撸管,露点,美穴,密穴,蜜洞,蜜穴,嫩逼,嫩穴,女人穴,群奸,肉棒,肉唇,肉洞,肉缝,肉沟,肉棍,肉壶,肉茎,肉枪,肉穴,骚逼,骚洞,骚穴,湿穴,兽奸,兽交,水穴,吞精,吸精,销魂洞,小逼,阳具,阴屄,阴精,淫逼,淫荡,淫棍,淫叫,淫乱,淫蜜,淫奴,淫枪,淫水,淫娃,淫穴,幼交,玉穴,欲龙,B嫩,扒穴,拔屌,暴菊花,暴奶,爆你菊,逼毛,逼样,操操妹妹,操烂,操了,操妈妈,操你,操妻,操爽,操死,操他,操我,操穴,操肿,肏,插b,插暴,插爆,插插,插进,插他,插她,插我,插穴,抽插呻吟,大逼,大花逼,大鸡巴,大力抽送,粉逼,粉嫩小穴,疯狂抽送,干妓女,干穴,干姨妈,幹炮,黑逼,花穴,换妻,鸡巴暴胀,鷄巴,奸尸,奸屍,奸幼,姦淫,交换夫妻,脚交,精水,菊花蕾,口淫,狂操,狂干,露逼,露毛,露奶,露穴,露阴毛,裸陪,裸乳,卖逼,卖比,摸逼,摸阴蒂,男人的鸡巴,嫩屄,女淫,炮图,炮友之家,喷精,噴精,嫖妓指南,枪淫少妇,强奸处女,强奸图,群射,人兽交,人妖互干,肉具,肉淫器,骚水,少妇白洁,少妇中出,生姦,绳虐,獸交,舔逼,舔到你射,舔屁眼,舔阴,调教性奴,调妻派对,万淫堂,洗Ｂ,小穴,穴图,穴淫,颜射,咬着龟头,要射了,阴穴,陰穴,淫虫,淫蕩,淫洞,淫妇,淫河,淫间道,淫贱,淫姐,淫浪,淫流,淫龙,淫亂,淫民,淫魔,淫母,淫妞,淫女,淫妻,淫情,淫色,淫僧,淫声浪语,淫术,淫图,淫液,淫欲,淫汁,狗交,指奸,肏逼,吃屌,吃鸡巴,后穴,扩肛运动,马奸,奶交,屁股洞,强操,强草,强插,强干,犬奸,群P,群草,群女干,乳奸,乳炮,幼奸,轮姦,淫男,秘穴,尿口,鸡八,淫根,巨屌,巨吊,虐奸,肉菊,操翻,被操,被插,被干,日逼,掰逼,淫肉,淫战,穴痒,浓精,菊门,穴儿,骚屄,牝户,扒开阴户,拔屄,掰B,屄肥,屄肉,屄爽,操干,操弄,操她,插屄,插进小穴,插弄,插死,插死我,插死我了,插小穴,超淫,大鸡巴臣服,大鸡巴大起大落,大鸡巴洞,大鸡巴亲,大乱交,大阴唇,粉B,粉嫩穴口,干骚屄,干骚屄荡妇,狠狠的抽插,鸡巴好大,鸡巴火烫,娇嫩花穴,娇嫩蜜穴,娇俏淫荡,娇穴,叫快插,叫快速抽插,紧紧的小穴,菊花穴,口交服务,媚穴,猛烈的抽插,蜜穴口处,奈战金钢杵,男茎,嫩B,嫩嫩的小穴,嫩穴口,嫰穴,虐肛,炮穴,喷嘲,喷潮,人兽杂交,骚B,骚屄爽,骚屄痒,射进小穴,射在嫂子内裤上,湿润的小穴,收紧穴儿,私穴,桃源洞套,舔小穴,舔穴,握鸡巴,无码内射,小屄,小浪穴,小浪穴弄大点,小穴儿,小穴弄湿,小淫娃儿,穴好涨,穴里,穴流,穴心,穴穴,穴嘴,阳精喷,阳具喷,阴肉,阴穴儿,淫棒,淫唇,淫货,淫弄,淫妻交换,淫辱,淫声,淫态,淫猥,淫物,淫兴,淫性,淫言浪语,淫艳,撞进蜜穴,花穴儿,热穴,爱液横流,暴奸,暴菊,爆草,操了嫂,操你妈,操你娘,操你全家,操嫂子,操他妈,肏你,肏死,插比,插屁屁,插阴,大b,大雞巴,洞小口紧,和狗交,和狗性,和狗做,奸成瘾,江湖淫娘,菊暴,菊穴,狂插,兩性淫亂,乱奸,美逼,母奸,母子乱伦,乱伦,男奴,嫩阴,犬交,日烂,肉逼,乳爆,色b,身体里抽送,食精,小xue,阴b,淫媚,淫虐,淫色,淫兽,淫亵,龙具";
@app.route("/getKeyWords", methods=['GET'])
def getKeyWordsAPI():
    keyWordList = keyWords.split(',')

    tempKeyWordList = []
    for keyWord in keyWordList:
        try:
            resultList = wv.wv.most_similar(keyWord, topn=40)
        except Exception as e:
            continue
        for result in resultList:
            if result[1] > 0.9:
                tempKeyWordList.append(result[0])

    keyWordList.extend(tempKeyWordList)
    keyWordSet = set(keyWordList)

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

    print("result len:", len(resultList), " result:", resultList)
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
    # print("拼接起来之后：", keyWord)

    keyWordDoc = [keyWord]
    x = vectorizer.transform(keyWordDoc)
    x = x.toarray()

    predictResult = sexClassifyModel.predict(x)
    return predictResult[0]

import json


def getSexKeyWords():
    keyWordList = keyWords.split(',')

    tempKeyWordList = []
    for keyWord in keyWordList:
        try:
            resultList = wv.wv.most_similar(keyWord, topn=40)
        except Exception as e:
            continue
        for result in resultList:
            if result[1] > 0.9:
                tempKeyWordList.append(result[0])

    keyWordList.extend(tempKeyWordList)
    keyWordSet = set(keyWordList)
    return keyWordSet

sexKeyWords = getSexKeyWords()

@app.route("/istext", methods=['GET'])
def istext():
    result = {}
    fullDoc = request.args.get('istext')

    print("get full doc", fullDoc)
    keyWordList = seg.cut(fullDoc)
    print("kew word len:", len(keyWordList))

    if len(keyWordList) == 0:
        predictResult = 0
    else:
        cutDoc = " ".join(keyWordList)
        print("分词之后", cutDoc)
        cutDocArray = [cutDoc]
        x = vectorizer.transform(cutDocArray)
        x = x.toarray()
        predictResult = int(sexClassifyModel.predict(x)[0])

    print('predictResult', predictResult)
    result['code'] = 200
    result['message'] = 'Success'

    returnSexKeyWords = set()
    if predictResult > 0.5:
        for keyWord in keyWordList:
            if sexKeyWords.__contains__(keyWord):
                returnSexKeyWords.add(keyWord)

        if len(returnSexKeyWords) == 0:
            predictResult = 0

    paragraphs = []
    if len(returnSexKeyWords) > 0:
        # 文本分段
        maxLen = 100
        for i in range(int(len(fullDoc) / maxLen) + 1):
            words = []
            cutParagraph = fullDoc[i*maxLen : (i+1)*maxLen]

            # 1. 使用|对色情关键字进行分割
            # 2. 使用()将整个表达式包括起来，这样可以保留所有的关键字
            separator = '('+'|'.join(returnSexKeyWords)+')'
            returnWords = re.split(separator, cutParagraph)

            paragraph = {}
            if len(returnWords) > 0:
                isPass = False
                for keyWord in returnWords:
                    word = {}
                    if returnSexKeyWords.__contains__(keyWord):
                        word['weight'] = 3
                        isPass = True
                    else:
                        word['weight'] = 0
                    word['word'] = keyWord
                    words.append(word)

                # 该分段中，需要有色情关键字才可以被加入
                if isPass:
                    paragraph['negative'] = predictResult
                    paragraph['words'] = words

            # 该分段中，需要有色情关键字才可以被加入
            if len(paragraph) > 0:
                paragraphs.append(paragraph)

    datas = {}
    datas['paragraphs'] = paragraphs

    article = {}
    article['negative'] = predictResult
    datas['article'] = article

    result['datas'] = datas

    jsonResult = json.dumps(result, ensure_ascii=False)

    print(jsonResult)
    return jsonResult

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0', threaded=False, processes=8)
