""" Search text from corpus. """
import sqlite3
import array
from collections import defaultdict


class TextSearcher:
    """ Directly search text related to input term. """

    def __init__(self, sqlitePath):
        conn = sqlite3.connect(sqlitePath)
        self._cur = conn.cursor()

    def genDocs(self, term):
        "Generate text if term included."
        articles = self._cur.execute("SELECT article from corpus")
        for art in articles:
            if term in art[0]:
                yield art[0]


def genDataFromSqlite(sqlitePath, limit=0):
    conn = sqlite3.connect(sqlitePath)
    cur = conn.cursor()
    articles = cur.execute("SELECT article from corpus")
    if limit == 0:
        for art in articles:
            yield art[0]
    else:
        n = 0
        for art in articles:
            n += 1
            yield art[0]
            if n >= limit:
                return


class InvertedIndex:
    """
        Build Inverted Index from text corpus.
        Increase searching time.
        Input: iterable text corpus.
    """

    def __init__(self, data):
        self._charIndexDict = dict()
        self._docIndex = []

        # Build char index dict.
        for index, doc in enumerate(data):

            self._docIndex.append(doc)

            for i, char in enumerate(doc):
                if char in self._charIndexDict:
                    self._charIndexDict[char][index].extend([i])
                else:
                    deDict = defaultdict(lambda: array.array('L'))
                    charPosition = [i]
                    self._charIndexDict[char] = deDict
                    self._charIndexDict[char][index].extend(charPosition)

    def getCharDict(self):
        return self._charIndexDict

    def getDocFromIndex(self):
        return self._docIndex

    def _checkSequenceNum(self, numList, matchLength):
        # Check a list has sequence numbers or not.
        sortedList = sorted(numList)
        match = 1
        for i in range(len(sortedList) - 1):
            if sortedList[i] + 1 == sortedList[i + 1]:
                match += 1
            else:
                match = 1
            if match == matchLength:
                return True
        return False

    def _checkSequenceLists(self, listA, listB):
        # Check sequence numbers between two lists.
        # Ex: A[1, 4] B[2] return True.
        s = set()
        for num in listA:
            s.add(num + 1)
        return bool(s & set(listB))

    def _search(self, text):
        textLen = len(text)
        bufIndex = []
        for docIndex in self._charIndexDict[text[0]]:
            # Get start index list from first char.
            bufIndex.append(docIndex)
        for i in range(textLen - 1):
            char = text[i]
            nextChar = text[i + 1]
            bufIndex2 = bufIndex
            bufIndex = []
            for docIndex in bufIndex2:
                if docIndex in self._charIndexDict[nextChar]:
                    listA = self._charIndexDict[char][docIndex]
                    listB = self._charIndexDict[nextChar][docIndex]
                    if self._checkSequenceLists(listA, listB):
                        bufIndex.append(docIndex)
        return bufIndex

    def searchGenerator(self, text, withIndex=False):
        indices = self._search(text)
        for index in indices:
            if withIndex is False:
                yield self._docIndex[index]
            else:
                yield (index, self._docIndex[index])
