"""Get prediction from mongodb."""
from pymongo import MongoClient
from searcher import Searcher
from bson.objectid import ObjectId


class RuntimePredict:
    """Input term, return predict scores."""

    def __init__(self):
        # Mongodb
        client = MongoClient("mongodb://localhost:27017/")
        db = client.invertedindex
        # mongodb collections
        self._char_index = db.char_index
        self._doc_index = db.doc_index
        self._searcher = Searcher()

    def predict(self, term):
        docIdGen = self._searcher.searchIdGenerator(term)
        scoresList = []
        for docId in docIdGen:
            doc = self._doc_index.find_one({"_id": ObjectId(docId)})

            scoresList.append(doc["score"])
        doc_nb = len(scoresList)
        scoreDict = {}

        # Sum label value in from all doc scores.
        for scores in scoresList:
            for score in scores:
                if score[0] not in scoreDict:
                    scoreDict[score[0]] = 0
                scoreDict[score[0]] += score[1]

        # Calculate average.
        for label in scoreDict:
            scoreDict[label] = scoreDict[label] / doc_nb

        return scoreDict
