"""
Character level cnn for chinese text classification.
Model reference: https://arxiv.org/abs/1509.01626

Accuracy gets to 71-72% in validation data
and 70-71% in test data after 10 epochs with following setting:
TEXT_LENGTH = 500
NB_FILTER = [64, 128]
NB_GRAM = [4, 3, 3]
FULLY_CONNECTED_UNIT = 256
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adamax
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import EarlyStopping

from lib import vector_generator
from lib import text_searcher

# Data source
SEARCH_TRAINING_TERMS = "training_terms.tsv"
TEXT_CORPUS = "data/corpus.sqlite"

# Model Hyperparameters
TEXT_LENGTH = 500
NB_FILTER = [64, 128]
NB_GRAM = [4, 3, 3]
FULLY_CONNECTED_UNIT = 256
DROPOUT = [0.5, 0.5]  # dropout rate in 2 fully connected layers.

# Training parameters
TEST_PERCENT = 0.1  # test data %
VALID_PERCENT = 0.1  # valid data %
epochs = 10
batch_size = 16
EARLY_STOP = False

# Prepare training data.
print("Preparing Data...")

trainingFile = open(
    "training_terms.tsv", "r", encoding="utf-8").read().strip().split("\n")

training_terms = []
for row in trainingFile:
    r = row.split("\t")
    x = r[0]
    y = r[1].split(" ")
    training_terms.append((x, y))

print("Building inverted index from data...")
inverted_data = text_searcher.InvertedIndex(
    text_searcher.genDataFromSqlite(TEXT_CORPUS))

print("Searching docs...")
searchResultList = []
for terms in training_terms:
    query = terms[0]
    for doc in inverted_data.searchGenerator(query):
        searchResultList.append((doc, terms[1]))
print("Docs: {}".format(len(searchResultList)))

print("Delete inverted index.")
inverted_data = None

vec_gen = vector_generator.VectGenerator(
    searchResultList,
    TEXT_LENGTH,
    training=(1 - TEST_PERCENT - VALID_PERCENT),
    testing=TEST_PERCENT,
    validation=VALID_PERCENT)


print("Loaded training data: {}".format(len(training_terms)))
print("Convert to docs: {}".format(len(searchResultList)))
print("Training samples: {}".format(vec_gen.nb_train_samples()))
print("Test samples: {}".format(vec_gen.nb_test_samples()))
print("Validation samples: {}".format(vec_gen.nb_val_samples()))

nb_char = vec_gen.nb_char_space()
nb_classes = vec_gen.nb_classes()

# Create the model
model = Sequential()
model.add(Convolution2D(
    NB_FILTER[0], nb_char, NB_GRAM[0],
    input_shape=vec_gen.input_shape(), border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Convolution2D(
    NB_FILTER[0], 1, NB_GRAM[1],
    border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Convolution2D(
    NB_FILTER[0], 1, NB_GRAM[2],
    border_mode='valid', activation='relu'))
model.add(Convolution2D(
    NB_FILTER[1], 1, NB_GRAM[2],
    border_mode='valid', activation='relu'))
model.add(Convolution2D(
    NB_FILTER[1], 1, NB_GRAM[2],
    border_mode='valid', activation='relu'))
model.add(Convolution2D(
    NB_FILTER[1], 1, NB_GRAM[2],
    border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Flatten())
model.add(Dropout(DROPOUT[0]))
model.add(Dense(
    FULLY_CONNECTED_UNIT, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(DROPOUT[1]))
model.add(Dense(
    FULLY_CONNECTED_UNIT, activation='relu', W_constraint=maxnorm(3)))
model.add(Dense(nb_classes, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
print(model.summary())

# Fit the model
samples_per_epoch = int(vec_gen.nb_train_samples() / batch_size) * batch_size

print("Start fitting model.")
print("Batch size: {}".format(batch_size))
print("Samples per epoch: {}".format(samples_per_epoch))

if EARLY_STOP is True:
    callback = [EarlyStopping(monitor='val_loss', patience=2)]
else:
    callback = []

model.fit_generator(
    vec_gen.trainGenerator(batch_size),
    samples_per_epoch,
    epochs,
    verbose=1,
    callbacks=callback,
    validation_data=vec_gen.validGenerator(batch_size),
    nb_val_samples=vec_gen.nb_val_samples(),
    class_weight=None,
    max_q_size=10,
    nb_worker=1,
    pickle_safe=False)

# Final evaluation of the model
test_nb = int(vec_gen.nb_test_samples() / batch_size) * batch_size
print("Test model with : {} test sets.".format(test_nb))
scores = model.evaluate_generator(
            vec_gen.testGenerator(batch_size),
            test_nb,
            max_q_size=10,
            nb_worker=1,
            pickle_safe=False)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk.")
