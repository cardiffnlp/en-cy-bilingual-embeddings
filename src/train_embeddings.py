import os
import sys
sys.path.append('../src')
import data_manager
from argparse import ArgumentParser
from gensim.models import FastText,Word2Vec
import logging

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-c','--corpus', help='Corpus path', required=True)
	parser.add_argument('-m','--model', help='Embeddings model', required=True, 
		choices = 'fasttext word2vec'.split())
	parser.add_argument('-o','--output-directory', help='Folder to store the embeddings', required=True)

	args = parser.parse_args()
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	modelmap = {'fasttext':FastText,'word2vec':Word2Vec}

	print(f'Loading corpus: {args.corpus}')
	# for welsh tokenization (tokens separated by ", ")
	corpus = data_manager.ExampleCorpus(args.corpus, sep=', ')
	# for english tokenization (tokens separated by " ")
	#corpus = data_manager.ExampleCorpus(args.corpus)
	model = modelmap[args.model](size=300, window=5, min_count=3, sentences=corpus, iter=10)

	if not os.path.exists(args.output_directory):
		print(f'Output folder {args.output_directory} does not exist, creating it...')
		os.makedirs(args.output_directory)
	outpath = os.path.join(args.output_directory,args.corpus.split('/')[-1]+'_model='+args.model+'_vectors.vec')
	model.wv.save_word2vec_format(outpath)
	print(f'Embeddings saved to {outpath}')
