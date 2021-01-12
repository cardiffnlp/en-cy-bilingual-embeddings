import pandas as pd
from collections import defaultdict
import re

def clean_token(t):
    stripped = re.sub(r"\s+", "", t)
    stripped = re.sub(r"\.", "", stripped)
    stripped = stripped.split("´")[0]
    stripped = stripped.split("'")[0]
    return stripped


class BaseCorpus:
    def __init__(self, path, sep = ' '):
        self.path = path
        self.sep = sep

    def __iter__(self):
        for line in open(self.path):
            out = [clean_token(t) for t in line.split(self.sep)]
            yield out

class ExampleCorpus(BaseCorpus):

    def __init__(self,path,sep=' '):
        BaseCorpus.__init__(self,path,sep)
        

    def count_words(self):

        self.freqDist = defaultdict(int)
        for line in open(self.path):
            toks = line.strip().split()
            for t in toks:
                self.freqDist[t] += 1

    def get_topk_words(self, topk=10):

        self.count_words()
        return sorted(self.freqDist.items(), key=lambda x:x[1], reverse=True)[:topk]

