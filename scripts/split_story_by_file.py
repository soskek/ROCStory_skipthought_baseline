from collections import OrderedDict
from itertools import chain
import json
import sys

def extract_story(file_name, train=True):
    dataset = OrderedDict()
    for line in open(file_name):
        sp = line.strip().split('\t')
        #InputStoryid    InputSentence1  InputSentence2  InputSentence3  InputSentence4  RandomFifthSentenceQuiz1        RandomFifthSentenceQuiz2        AnswerRightEnding
        if train:
            idx, sentences, title = sp[0], sp[2:], sp[1]
            dataset[idx] = {"sentences": sentences, "title": title}
        else:
            idx, sentences, answer = sp[0], sp[1:7], sp[7]
            dataset[idx] = {"sentences": sentences, "answer": answer}
    dataset.popitem(last=False)
    return dataset

def flatten(list_list):
    return list(chain.from_iterable(list_list))

def get_chain_sentences(dataset):
    return flatten([d["sentences"] for d in dataset.values()])

######

def writedown(dataset, outdir, train=True):
    for idx, data in dataset.items():
        if train:
            print >>open(outdir + idx, "w"), "\n".join(data["sentences"]).encode('utf-8')
        else:
            print >>open(outdir + idx + ".1", "w"), "\n".join(data["sentences"][:5]).encode('utf-8')
            print >>open(outdir + idx + ".2", "w"), "\n".join(data["sentences"][:4] + data["sentences"][5:]).encode('utf-8')

if __name__ == "__main__":
    train_data = extract_story("/home/sosuke.k/data/ROCStories/ROCStories__spring2016-ROC-Stories-naacl-camera-ready.tsv")
    valid_data = extract_story("/home/sosuke.k/data/ROCStories/cloze_test_val__spring2016-cloze_test_ALL_val.tsv", train=False)
    test_data = extract_story("/home/sosuke.k/data/ROCStories/cloze_test_test__spring2016-cloze_test_ALL_test.tsv", train=False)

    jbasedir = sys.argv[1].rstrip("/")+"/"
    json.dump(train_data, open(jbasedir + "train_data.json", "w"))
    json.dump(valid_data, open(jbasedir + "valid_data.json", "w"))
    json.dump(test_data, open(jbasedir + "test_data.json", "w"))

    """
    jbasedir = sys.argv[1].rstrip("/")+"/"
    train_data = json.load(open(jbasedir + "train_data.json"))
    valid_data = json.load(open(jbasedir + "valid_data.json"))
    test_data = json.load(open(jbasedir + "test_data.json"))
    """

    print len(train_data)
    print len(valid_data)
    print len(test_data)

    """
    basedir = sys.argv[2].rstrip("/")+"/"
    writedown(train_data, basedir + "train/", train=True)
    writedown(valid_data, basedir + "valid/", train=False)
    writedown(test_data, basedir + "test/", train=False)
    """
