import sys
import os
from argparse import ArgumentParser
import pandas as pd

def parseline(l):
	# Coverage:100.00%  Accuracy:  14.37%
	cov = l.split(' Accuracy:')[0].split('Coverage:')[1].strip()[:-1]
	acc = l.split(' Accuracy:')[1].strip()[:-1]
	print(cov)
	print(acc)
	return {'coverage':float(cov), 'accuracy':float(acc)}

def get_config(path):
	# model=fasttext_s__retrieval=nn__s=100_mc=6_w=8.txt
	model = path.split('model=')[1].split('_s__')[0]
	size = int(path.split('s=')[1][:3])
	mincount = int(path.split('_mc=')[1][:1])
	window = int(path.split('_w=')[1][:1])
	retrieval = path.split('_retrieval=')[1].split('__')[0]
	return model,size,mincount,window,retrieval


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-r','--results-folder', help='Folder that stores results', required=True)
	
	args = parser.parse_args()

	listing = os.listdir(args.results_folder)

	rows = []
	for inf in listing:
		line = open(os.path.join(args.results_folder,inf)).readlines()[0]
		res = parseline(line)
		model,s,mc,w,retrieval = get_config(inf)
		res['model'] = model
		res['size'] = s
		res['mc'] = mc
		res['window'] = w
		res['retrieval'] = retrieval
		rows.append(res)

	df = pd.DataFrame(rows)