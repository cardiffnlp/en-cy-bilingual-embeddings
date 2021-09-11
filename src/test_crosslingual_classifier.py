import pandas as pd
import gensim
import os
import sys
import numpy as np
import pickle
import data_manager
from argparse import ArgumentParser
from utils import embeddings,neuralnets
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from collections import defaultdict,Counter
### keras 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input, Dense, Bidirectional, LSTM, Activation, MaxPooling1D, Conv1D, Dropout, Embedding, ActivityRegularization, concatenate
from keras.models import Model, Sequential, load_model


def load_tokenizer(tokenizer_path):
	print('Loading tokenizer from: ',tokenizer_path)
	with open(tokenizer_path, 'rb') as handle:
		tokenizer = pickle.load(handle)
		return tokenizer


def load_labelmap(labelfile):
	out={}
	out2={}
	for line in open(labelfile):
		cols = line.strip().split('\t')
		lab,lab_id = cols
		out[lab]=int(lab_id)
		out2[int(lab_id)]=lab
	return out,out2

def load(data_folder):
    # d = {}
    d = defaultdict(list)
    # your code here
    # the returned dictionary should be of the form:
    # {'pos': [
    #			['word1', 'word2', word3, ... ], # each of these nested lists of strings are imdb reviews
    #			['word1', 'word2', word3, ... ]
    # 			... }
    # {'neg': [
    #			['word1', 'word2', word3, ... ],
    #			['word1', 'word2', word3, ... ]
    # 			... }
    files = os.listdir(data_folder)
    for infile in files:
        f = open(os.path.join(data_folder, infile), 'r', encoding='utf-8')  #  (Windows)
        documents = []
        for line in f:
            words = [word for word in line.split()]
            documents.append(words)
        d[infile] = documents
    # process the review file 'f' and populate the dictionary 'd'
    return d


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-d', '--dataset', help='Tab-separated dictionary (DLE only!) path', required=True)
	parser.add_argument('-m', '--maxlen', help='maxlen (required to fit into keras model)', required=False)	
	parser.add_argument('-im', '--input-model', help='keras classifier', required=True)
	parser.add_argument('-it', '--input-tokenizer', help='keras trained tokenizer', required=True)
	parser.add_argument('-il', '--input-labelmap', help='labelmap', required=True)

	args = parser.parse_args()

	t = load_tokenizer(args.input_tokenizer)
	model = load_model(args.input_model)
	label2id,id2label = load_labelmap(args.input_labelmap)

	#print(label2id,id2label)

	data = load(args.dataset)

	docs = []
	labels = []
	for label in data:
		for d in data[label]:
			docs.append(d)
			labels.append(label2id[label])


	seqs = t.texts_to_sequences(docs)

	# set default value for maxlen
	if not args.maxlen == None:
		maxlen = int(args.maxlen)
	else:
		maxlen = 100		

	X_test = pad_sequences(seqs, maxlen=maxlen, padding='post')
	
	y_test = [np.argmax(j) for j in to_categorical(np.asarray(labels))]

	preds = model.predict(X_test)
	pred_labels = [np.argmax(k) for k in preds]

	print(classification_report(y_test,pred_labels,digits=4))