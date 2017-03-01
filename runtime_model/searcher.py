""" Search documents from inverted index in mongodb. """
from pymongo import MongoClient
from bson.objectid import ObjectId


class Searcher:

    def __init__(self):
        client = MongoClient("mongodb://localhost:27017/")
        db = client.invertedindex
        self._charIndex = db.char_index
        self._docIndex = db.doc_index

    def _checkSequenceLists(self, listA, listB):
        # Check sequence numbers between two lists.
        # Ex: A[1,3] B[2] return True.
        s = set()
        for num in listA:
            s.add(num + 1)
        return bool(s & set(listB))

    def _queryChar(self, char):
        # return all char index.
        indices = {}
        charDicts = self._charIndex.find({"char": char})
        for charDict in charDicts:
            index = charDict["index"]
            indices = {**indices, **index}
        return indices

    def _search(self, text):
        textLen = len(text)
        bufIndex = []
        charDict = self._queryChar(text[0])
        nextCharDict = None
        for docIndex in charDict:
            # Get start index list from first char.
            bufIndex.append(docIndex)

        if textLen <= 1:
            # For only one char search.
            return bufIndex

        for i in range(textLen - 1):
            nextChar = text[i + 1]
            bufIndex2 = bufIndex
            bufIndex = []
            nextCharDict = self._queryChar(nextChar)
            for docIndex in bufIndex2:
                if docIndex in nextCharDict:
                    listA = charDict[docIndex]
                    listB = nextCharDict[docIndex]
                    if self._checkSequenceLists(listA, listB):
                        bufIndex.append(docIndex)
            charDict = nextCharDict

        return bufIndex

    def searchDocGenerator(self, text):
        indices = self._search(text)
        for index in indices:
            yield self._docIndex.find_one({"_id": ObjectId(index)})["doc"]

    def searchIdGenerator(self, text):
        indices = self._search(text)
        for index in indices:
            yield index
