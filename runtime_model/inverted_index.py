"""
Build inverted index saving into mongodb.
"""
from collections import defaultdict
import os.path
import sqlite3
from pymongo import MongoClient

import time

# Mongodb
client = MongoClient("mongodb://localhost:27017/")
db = client.invertedindex

# Corpus path
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CORPUS_PATH = os.path.join(BASE_PATH, "../data/corpus.sqlite")

# mongodb collections
char_index = db.char_index
doc_index = db.doc_index

# measure used time.
timeUsed = time.perf_counter()


def genDataFromSqlite(sqlitePath, limit=0):
    conn = sqlite3.connect(sqlitePath)
    cur = conn.cursor()
    articles = cur.execute("SELECT article from corpus")
    if limit == 0:
        for art in articles:
            text = art[0].replace("\n", "")
            yield text
    else:
        n = 0
        for art in articles:
            n += 1
            text = art[0].replace("\n", "")
            yield text
            if n >= limit:
                return


def buildCharIndex(charDict, collection, seperate=1):
    m = 0  # count number of chars.
    for char, docDict in charDict.items():
        m += 1
        t = time.perf_counter()
        dictLen = len(docDict)
        sepSize = int(dictLen / seperate)
        if dictLen % seperate is not 0:
            sepSize += 1
        pushList = []
        bufDict = {}
        n = 0
        for key in docDict:
            n += 1
            bufDict[key] = docDict[key]
            if n == sepSize:
                pushList.append(bufDict)
                bufDict = {}
                n = 0
        if len(bufDict) is not 0:
            pushList.append(bufDict)

        for docDict in pushList:
            collection.insert_one({
                "char": char,
                "index": docDict,
                "seperate": seperate
            })
        print("{} Char {} Time used:{} ".format(
                m, char, time.perf_counter() - t))


data = genDataFromSqlite(CORPUS_PATH)
n = 0
charIndexDict = {}
for doc in data:
    n += 1

    print("doc: {}".format(n))

    index = str(doc_index.insert_one(
        {
            "doc": doc
        }
    ).inserted_id)

    for i, char in enumerate(doc):
        if char in charIndexDict:
            charIndexDict[char][index].append(i)
        else:
            charIndexDict[char] = defaultdict(list)
            charIndexDict[char][index] = [i]

print("Uploading index...")

buildCharIndex(charIndexDict, char_index, seperate=6)

print("Total time usage: {}".format(time.perf_counter() - timeUsed))
