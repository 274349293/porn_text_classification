#coding:utf-8
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import re
import pickle
import json
if sys.version_info.major < 3:
    print('is 2.7')
else:
    text_type = str

import time
from functools import wraps
from glob import glob
def is_chinese_char(uchar):
    return u'\u4e00' <= uchar <= u'\u9fa5'

def is_alpha(text):
    if isinstance(text, text_type):
        text = text.encode('utf-8')
    return text.isalpha()

def is_alnum(text):
    if isinstance(text, text_type):
        text = text.encode('utf-8')
    return text.isalnum()

def has_number(text):
    return bool(re.search(r'\d', text))

def search_file(pattern, search_path, pathsep = os.pathsep):
    search_path_list = search_path.split(pathsep)
    all_match=list()
    for path in search_path_list:
        for match in glob(os.path.join(path, pattern)):
            all_match.append(match)
    return all_match

def dump_object(obj, file):
    with open(file, 'wb') as wf:
        pickle.dump(obj, wf, protocol=2)

def load_object(file):
    try:
        with open(file, 'rb') as rf:
            return pickle.load(rf)
    except UnicodeDecodeError:
        # python2 dump的文件用python3加载需要指定encoding
        with open(file, 'rb') as rf:
            return pickle.load(rf, encoding='latin1')

def dump_json(obj, file):
    with open(file, 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False)

def load_json(file):
    with open(file, 'r', encoding='utf-8') as rf:
        return json.load(rf)

http_pattern = '(((http|https)://)|www.)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?'
def clean_text(text):
    text = text.strip()
    text = re.sub(http_pattern, ' ', text)

    want = re.compile(u'[^\u4E00-\u9FA5\u0020a-zA-Z0-9]')
    text = want.sub(u' ', text)

    punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")
    text = punctuation.sub(u' ', text)
    return text

def clean_digit(text):
    text = text.strip()
    want = re.compile(u'[0-9]')
    text = want.sub(u' ', text)
    return text

def clean_text_only_chinese(text):
    text = text.strip()
    want = re.compile(u'[^\u4E00-\u9FA5]')
    text = want.sub(u' ', text)

    punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")
    text = punctuation.sub(u' ', text)
    return text

def escaped_time(func):
    @wraps(func)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        print("escaped_time %s: %d seconds" % (func.__name__, time.time() - t0))
        return result

    return function_timer

if __name__ == '__main__':
    print(clean_text(u'中国 人民。!@#$abc edef 0000 1111 222'))
    print(u'中国aa'.encode('utf-8').isalpha())
    print(is_alpha(u'中国aa'))
