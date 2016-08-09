from corenlp_xml import document
from itertools import chain
from collections import OrderedDict
from collections import defaultdict


class Coreference():
    def __init__(self, idx, coref):
        self.idx = idx
        self.surface = " ".join(token.word for token in coref.representative.tokens)
        self.bow = list(chain.from_iterable([token.word for token in mention.tokens] for mention in coref.mentions))
        self.spans = dict((mention.sentence.id-1, [mention.tokens[0].id-1, mention.tokens[-1].id-1+1]) for mention in coref.mentions)
        self.points = dict(self.spans)


    def __str__(self):
        return " ".join(["<Coreference", str(self.idx), self.surface, str(self.points)])


class Document():
    def __init__(self, file_name):
        self.file_name = file_name.strip()
        self.doc_name = file_name.strip().split("/")[-1]
        self.split_type = file_name.strip().split("/")[-2]# train/valid/test
        self.coherence = not self.split_type.endswith(".2.out")# True/False
        self.load_story(open(file_name).read())


    def load_story(self, doc_string):
        doc = document.Document(doc_string)
        
        self.sentences = [[token.word for token in sentence.tokens] for sentence in doc.sentences]

        if doc.coreferences is None:
            self.coreferences = []
            self.sentence_idx2coref_idsD = {}
            return 0
            
        self.coreferences = [Coreference(i, coref) for i, coref in enumerate(doc.coreferences)]

        new_sentences = []
        self.sentence_idx2coref_idsD = defaultdict(list)
        for i, sentence in enumerate(list(self.sentences)):

            span_idx_list = [(coref.spans[i], coref.idx) for coref in self.coreferences if i in coref.spans]

            if span_idx_list:
                skips = set(chain.from_iterable(range(span_idx[0][0]+1, span_idx[0][1]) for span_idx in span_idx_list))
                points = dict((span_idx[0][0], span_idx[1]) for span_idx in span_idx_list)

                new_sentence = [token if not j in points else points[j]
                                for j, token in enumerate(sentence) if not j in skips]
                new_sentences.append(new_sentence)
                for j, idx in [(j, token) for j, token in enumerate(new_sentence) if isinstance(token, int)]:
                    self.coreferences[idx].points[i] = j
                    #self.coreferences[idx].spans[i] = None
                    self.sentence_idx2coref_idsD[i].append(idx)
            else:
                new_sentences.append(sentence)

        for c in self.coreferences:
            c.spans = {}
        self.sentences = new_sentences
        self.sentence_idx2coref_idsD = dict(self.sentence_idx2coref_idsD)


    def __str__(self):
        return " ".join(["<Document", "[", " ".join([self.sentence_surface(i) for i in range(5)]), "] {" + ", ".join(str(c.idx) + " " + c.surface for c in self.coreferences)]) + "}>"


    def sentence_surface(self, idx=0):
        return " ".join(str(t) for t in self.sentences[idx])

"""
We would go camping every summer.
We would pack lunches and trek out.
We would stop at a rest stop for lunch.
Typically it would be a McDonald's and I'd order fries.
I love french fries.
"""

if __name__ == "__main__":
    #doc = Document(open("/home/sosuke.k/data/ROCStories/ROCS_coref/test/aca1cf58-a0cd-4110-ae36-f2523a72cdee.1.out").read())
    doc = Document("/home/sosuke.k/data/ROCStories/ROCS_coref/test/aca1cf58-a0cd-4110-ae36-f2523a72cdee.1.out")

    cores = doc.coreferences
    print cores[0]
    print cores[1]
    print doc
