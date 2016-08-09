from collections import OrderedDict
from itertools import chain

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

if __name__ == "__main__":
    extract_story(sys.argv[1])
