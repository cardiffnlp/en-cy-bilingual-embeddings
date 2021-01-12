import data_manager
from argparse import ArgumentParser
from gensim.models import FastText,Word2Vec
import logging

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-c','--corpus', help='Corpus file', required=True)
	args = parser.parse_args()

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	corpus = data_manager.ExampleCorpus(args.corpus)

	
	for line in corpus:
		print(line)

	# Gets most frequent words
	topk = corpus.get_topk_words(topk=100)
	print('Most frequent words:')
	for k in topk:
		print(k)

	ft = FastText(size=100, window=5, min_count=3, sentences=corpus, iter=10)
	
	for a,b in ft.most_similar('felltithio'): 
		print(a,b)
