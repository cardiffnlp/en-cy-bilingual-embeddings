import gensim
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input, Dense, Bidirectional, LSTM, Activation, MaxPooling1D, Conv1D, Dropout, Embedding, ActivityRegularization, concatenate
from keras.models import Model, Sequential

def make_embedding_layer(tokenizer,vector_size,embedding_vocab,embedding_model,maxlen):
    # build word embedding layer based on dataset vocabulary
    word_index=tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, vector_size))
    for word, index in word_index.items():
        if word in embedding_vocab:
            embedding_matrix[word_index[word]]=embedding_model[word]
    embedding_layer = Embedding(len(word_index) + 1,
                                vector_size,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)
    return embedding_layer

def sent_cleaner(tokens,stopwords):
    out=[]
    for token in tokens:
        if not token in stopwords:
            out.append(token)
    return out

def load_embeddings(path):
    print('Loading embeddings:',path)
    try:
        model=gensim.models.Word2Vec.load(path)
    except:
        try:
            model=gensim.models.KeyedVectors.load_word2vec_format(path)
        except:
            try:
                model=gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
            except:
                sys.exit('Couldnt load embeddings')
    vocab=set(model.vocab)
    dims=model.vector_size
    return model,vocab,dims

def add_header(path, outpath):
	print('Counting vectors')
	nvecs = 0
	for line in open(path):
		nvecs += 1
	vsize = len(line.strip().split())-1

	print('Saving new file at: ',outpath)
	with open(outpath,'w') as outf:
		outf.write(str(nvecs)+' '+str(vsize)+'\n')
		for line in open(path):
			outf.write(line)

def array2string(a):
    """
    Helper function for converting numpy array to string
    """
    if not len(a.shape) == 1:
        raise Exception('You are passing an array of 1+ dimensions')
    return ' '.join([str(k) for k in a])

def weigh_by_position(embedding_sequence, strength=0.9):
    """
    Function to apply a weight to each embedding in the sequence based on position. Gives less importance to words at the beginning.
    :@param: embedding_sequence - An array
    :@param: strength - Float between 0 and 1. The higher the more aggressive the penalty to words at first positions.

    @input: 
        >>> a = np.array([[1,2,3],
                          [1,2,3],
                          [1,2,3]])
    @output: 
        >>> weigh_by_position(a)
        
        [array([0.1, 0.2, 0.3]),
        array([0.4, 0.8, 1.2]),
        array([0.56666667, 1.13333333, 1.7])]
    """
    out_seq=[]
    for idx,vec in enumerate(embedding_sequence):
        prop = abs(strength-(1/(idx+1)))
        newvec = vec*prop
        out_seq.append(newvec)
    return out_seq
