""" Build prediction for every documents. """
import time
import os.path
from keras.models import model_from_json
from pymongo import MongoClient
from lib import VectGenerator

# Mongodb
client = MongoClient("mongodb://localhost:27017/")
db = client.invertedindex

# mongodb collections
char_index = db.char_index
doc_index = db.doc_index

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
json_file = open(os.path.join(BASE_PATH, 'model/model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights into new model
model.load_weights(os.path.join(BASE_PATH, 'model/model.h5'))
print("Loaded model and weights from disk")

# Text length should be as same as training model.
TEXT_LENGTH = 800


def docGenerator():
    while 1:
        curs = doc_index.find({"score": {"$exists": False}}).limit(100)
        if curs.count() is 0:
            return
        for doc in curs:
            yield doc


timeUsed = time.perf_counter()
n = 0
for docDict in docGenerator():
    n += 1
    t = time.perf_counter()
    docId = docDict["_id"]
    dataList = [(docDict["doc"], ["食物"])]
    vec_gen = VectGenerator(
        dataList,
        TEXT_LENGTH,
        forPredict=True)

    y = model.predict_generator(
        vec_gen.predictGenerator(1),
        vec_gen.nb_predict_samples(),
        max_q_size=10, nb_worker=1, pickle_safe=False)
    scores = []
    for i, v in enumerate(y[0]):
        scores.append([vec_gen.getVectorizer().getLabels()[i], float(v)])
    print(scores)
    result = doc_index.update_one({"_id": docId}, {"$set": {"score": scores}})
    if result.matched_count is 1:
        print("updated successes.")
    print("doc: {} time: {}".format(n, time.perf_counter() - t))

print("Total time used: {}".format(time.perf_counter() - timeUsed))
