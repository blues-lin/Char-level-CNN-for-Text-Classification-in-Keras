from operator import itemgetter
import os.path
import argparse

from keras.models import model_from_json
import numpy as np

from lib import vector_generator
from lib import text_searcher


class Predictor:
    """Use trained model to predict catalog of terms."""

    def __init__(self):
        # Setting model.
        BASE_PATH = os.path.dirname(os.path.realpath(__file__))
        json_file = open(os.path.join(BASE_PATH, 'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self._model = model_from_json(loaded_model_json)

        # Load weights into new model
        self._model.load_weights(os.path.join(BASE_PATH, 'model.h5'))
        print("Loaded model and weights from disk")

        # Corpus path.
        self._corpusPath = os.path.join(BASE_PATH, "data/corpus.sqlite")
        # Text length should be as same as training model.
        self.TEXT_LENGTH = 800

        # Inverted index.
        self._inverted_data = None

    def predict(self, term, useInvertedIndex=True):

        if self._inverted_data is None and useInvertedIndex is True:
            print("Building inverted index from data...")
            self._inverted_data = text_searcher.InvertedIndex(
                text_searcher.genDataFromSqlite(self._corpusPath))

        searchList = []
        if useInvertedIndex is True:
            for doc in self._inverted_data.searchGenerator(term):
                # ["食物"] is a fillter for input argument. It won't be used.
                searchList.append((doc, ["食物"]))
        else:
            textSearcher = text_searcher.TextSearcher(self._corpusPath)
            for doc in textSearcher.genDocs(term):
                # ["食物"] is a fillter for input argument. It won't be used.
                searchList.append((doc, ["食物"]))
        nb_doc = len(searchList)
        print("doc found numbers: {}".format(nb_doc))

        vec_gen = vector_generator.VectGenerator(
            searchList, self.TEXT_LENGTH, forPredict=True)
        y = self._model.predict_generator(
            vec_gen.predictGenerator(1),
            vec_gen.nb_predict_samples(),
            max_q_size=10, nb_worker=1, pickle_safe=False)

        predictY = np.zeros(y[0].shape)
        for r in y:
            predictY = predictY + r
        predictY = predictY / nb_doc
        scores = []
        for i, v in enumerate(predictY):
            scores.append((vec_gen.getVectorizer().getLabels()[i], v))
        scores = sorted(scores, key=itemgetter(1), reverse=True)

        print("{} predict summary:".format(term))
        for score in scores:
            print("cateory: {}		score: {} %".format(score[0], score[1] * 100))
        return scores


if __name__ == '__main__':
    p = Predictor()
    aparser = argparse.ArgumentParser()
    aparser.add_argument('text')
    aparser.add_argument('--invindex', '-i',
                     action="store_true",
                     help="build inverted index for better search speed.")

    args = aparser.parse_args()
    p.predict(args.text, useInvertedIndex=args.invindex)
