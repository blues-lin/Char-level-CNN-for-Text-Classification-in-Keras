"""
Build look-up char table from iterable corpus.
Build labels from training terms.
For traditional chinese text, uses corpus.sqlite. Corpus comes from ptt.cc.
"""
import os.path
import sqlite3
import csv
from lib import table_builder

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CHAR_PATH = os.path.join(BASE_PATH, 'lib/data/charTable.txt')
LABEL_PATH = os.path.join(BASE_PATH, 'lib/data/label.txt')
TRAINING_TERM_PATH = os.path.join(BASE_PATH, 'training_terms.tsv')
# Analyze character used for discard char.
CHAR_USAGE_PATH = os.path.join(BASE_PATH, 'charUsageCount.csv')
CORPUS_PATH = os.path.join(BASE_PATH, 'data/corpus.sqlite')


def buildCharTable(corpus):
    discardCharFile = open("discardChar.txt", "r", encoding='utf-8').read()
    discardCharSet = set()
    for c in discardCharFile:
        discardCharSet.add(c)
    charDict = dict()
    table = table_builder.LookupTableBuilder(CHAR_PATH)
    i = 0
    for doc in corpus:
        text = doc.strip('\n')
        i += 1
        for c in text:
            if c in discardCharSet:
                continue
            if type(c) is str:
                table.addChar(c)
                # Analyze char usage count.
                if c in charDict:
                    charDict[c] += 1
                else:
                    charDict[c] = 1
    table.saveChar()

    # For counting char.
    csvfile = open(CHAR_USAGE_PATH, 'w', encoding='utf-8')
    fieldnames = ['char', 'number']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for key, value in charDict.items():
        writer.writerow({'char': key, 'number': value})
    csvfile.close()


def buildLabel(filePath):
    f = open(filePath, "r", encoding="utf-8")
    labels = set()
    for row in f:
        r = row.strip("\n").split("\t")
        l = r[1].split(" ")
        for lab in l:
            labels.add(lab)
    print("Build labels file: {}".format(labels))
    labelFile = open("label.txt", "w", encoding="utf-8")
    for label in labels:
        labelFile.write(label + "\n")
        labelFile.close()


conn = sqlite3.connect(CORPUS_PATH)
cur = conn.cursor()
article = cur.execute("SELECT article from corpus")
docs = (x[0] for x in article)

buildCharTable(docs)
buildLabel(TRAINING_TERM_PATH)
