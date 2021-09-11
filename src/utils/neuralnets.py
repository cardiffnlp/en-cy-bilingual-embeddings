import numpy as np
import pickle
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input, Dense, Bidirectional, LSTM, Activation, MaxPooling1D, Conv1D, Dropout, Embedding, ActivityRegularization, concatenate
from keras.models import Model, Sequential

### HELPERS TO SAVE AND LOAD KERAS TOKENIZER ###
def save_tokenizer(tokenizer, outpath):
	with open(outpath, 'wb') as handle:
	    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print('Tokenizer: ',tokenizer,' | Saved to: ',outpath)

def load_tokenizer(tokenizer_path):
	print('Loading tokenizer from: ',tokenizer_path)
	with open(tokenizer_path, 'rb') as handle:
		tokenizer = pickle.load(handle)
		return tokenizer

def preprocess(train_docs, train_labels, test_docs, test_labels, tokenizer_path, maxlen):
	if not os.path.exists(tokenizer_path):
		t = Tokenizer()
		print('No tokenizer found, fitting to train set')
		t.fit_on_texts(train_docs)
		save_tokenizer(t, tokenizer_path)
	else:
		#print('Using tokenizer found at ',tokenizer_path)
		t = load_tokenizer(tokenizer_path)
	# convert raw docs to padded sequences of ints
	train_sequences = t.texts_to_sequences(train_docs)
	test_sequences = t.texts_to_sequences(test_docs)
	X_train = pad_sequences(train_sequences, maxlen=maxlen)
	X_test = pad_sequences(test_sequences, maxlen=maxlen)
	# convert labels to one-hot encoded vectors
	y_train = to_categorical(np.asarray(train_labels))
	y_test = to_categorical(np.asarray(test_labels))
	print('Shape of train data tensor:', X_train.shape)
	print('Shape of train label tensor:', y_train.shape)
	print('Shape of test data tensor:', X_test.shape)
	print('Shape of test label tensor:', y_test.shape)
	return t,X_train,y_train,X_test,y_test

class CBLSTM:
	def __init__(self,labels_idx, embedding_layer):
		# cnn config
		self.FILTERS=100
		self.KERNEL_SIZE=3
		self.POOL_SIZE=4
		self.STRIDES=1
		self.PADDING='valid'
		self.CNN_ACTIVATION='relu'
		# lstm config
		self.LSTM_SIZE=100
		# global
		self.OPTIMIZER='adam'
		self.labels_idx = labels_idx
		self.embedding_layer = embedding_layer
	def build_model(self):
		# last (prediction) layer hyperparams
		softmax_size=len(self.labels_idx)
		nnmodel = Sequential()
		nnmodel.add(self.embedding_layer)
		nnmodel.add(Dropout(0.25))
		nnmodel.add(Conv1D(self.FILTERS,
		                 self.KERNEL_SIZE,
		                 padding=self.PADDING,
		                 activation=self.CNN_ACTIVATION,
		                 strides=self.STRIDES))
		nnmodel.add(MaxPooling1D(pool_size=self.POOL_SIZE))
		nnmodel.add(Bidirectional(LSTM(self.LSTM_SIZE)))
		nnmodel.add(Dropout(0.25))
		nnmodel.add(Dense(softmax_size))
		nnmodel.add(Activation('softmax'))
		nnmodel.compile(loss='categorical_crossentropy',
		              optimizer=self.OPTIMIZER,
		              metrics=['accuracy'])
		print('=== MODEL SUMMARY ===')
		print(nnmodel.summary())
		return nnmodel

#class TermPredictorFromDef:

def term_predictor_from_def(maxlen, embedding_layer, vector_size, lstm_size):
	model = Sequential()
	model.add(embedding_layer)
	model.add(Bidirectional(LSTM(lstm_size)))
	model.add(Dense(vector_size))
	model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
	return model
