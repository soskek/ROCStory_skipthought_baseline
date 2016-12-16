import os
import sys
import skipthoughts
from collections import OrderedDict
import json
import numpy as np

from scripts import preprocess
os.environ["THEANO_FLAGS"] = "device=gpu0"  # ?


print("load model")
model = skipthoughts.load_model()


print("extract stories")
file_paths = sys.argv[1:4]

train_data = preprocess.extract_story(file_paths[0], train=True)
valid_data = preprocess.extract_story(file_paths[1], train=False)
test_data = preprocess.extract_story(file_paths[2], train=False)


print("save data dict")
json.dump(train_data, open("train_data.json", "w"))
json.dump(valid_data, open("valid_data.json", "w"))
json.dump(test_data, open("test_data.json", "w"))


print("get chain sentences")
train_sentences = preprocess.get_chain_sentences(
    train_data)  # Each 5 sentences is a set
# Each 6 sentences is a set, first 4 of which are story-chain and 2 after
# that are choices for question
valid_sentences = preprocess.get_chain_sentences(valid_data)
# Each 6 sentences is a set, first 4 of which are story-chain and 2 after
# that are choices for question
test_sentences = preprocess.get_chain_sentences(test_data)


def get_chunk(iterable, size):
    return [iterable[x:x + size] for x in xrange(0, len(iterable), size)]


def save_vectors(dataset, chain_vectors, save_file_name, train=True):
    vectors_dict = {}
    size = 5 if train else 6
    for key, vectors in zip(dataset.keys(), get_chunk(chain_vectors, size)):
        vectors_dict[key] = vectors
    print("save", save_file_name)
    shape = vectors.shape
    shape = (1, shape[0], shape[1])
    vectors_matrix = np.concatenate([v.reshape(shape) for k, v in sorted(
        vectors_dict.items(), key=lambda x:x[0])], axis=0)
    np.save(save_file_name, vectors_matrix)

print("encode valid vectors")
valid_vectors = skipthoughts.encode(model, valid_sentences)
save_vectors(valid_data, valid_vectors, "valid_vectors", train=False)

print("encode test vectors")
test_vectors = skipthoughts.encode(model, test_sentences)
save_vectors(test_data, test_vectors, "test_vectors", train=False)

print("encode train vectors")
train_vectors = skipthoughts.encode(model, train_sentences)
save_vectors(train_data, train_vectors, "train_vectors", train=True)
