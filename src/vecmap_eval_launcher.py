import sys
import os
from argparse import ArgumentParser

def get_all_filepaths(root,mapped_suffix='_mapped.vec'):
	for path, subdirs, files in os.walk(root):
		for name in files:
			if name.endswith(mapped_suffix):
				yield os.path.join(path, name)

def get_config(path):

	if 'word2vec_s' in path:
		prefix = 'word2vec_s'
	elif 'fasttext_s' in path:
		prefix = 'fasttext_s'

	size = int(path.split(prefix)[1][:3])
	mincount = int(path.split('_mc')[1][:1])
	window = int(path.split('_w')[1][:1])
	return prefix,size,mincount,window

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-td','--testdict', help='Training dictionary path', required=True)
	parser.add_argument('-sv','--source-vectors-folder', help='Source (English) embeddings folder', required=True)
	parser.add_argument('-tv','--target-vectors-folder', help='Target (Welsh) embeddings folder', required=True)
	parser.add_argument('-r','--results-folder', help='Folder to store results', required=True)

	args = parser.parse_args()


	envecs = sorted(list(get_all_filepaths(args.source_vectors_folder,mapped_suffix='_mapped.vec')))
	welvecs = sorted(list(get_all_filepaths(args.target_vectors_folder,mapped_suffix='_mapped.vec')))

	retrieval_opts = ['nn', 'invsoftmax', 'csls']

	for env in envecs:
		enprefix, ens, enmc, enw = get_config(env)
		for welv in welvecs:
			welprefix, wels, welmc, welw = get_config(welv)
			if ens==wels and enmc == welmc and enw == welw and enprefix == welprefix:
				for opt in retrieval_opts:
					
					print('EN: ',env)
					print('WEL: ',welv)
					print('RETR: ',opt)
					
					runstr = 'python3 vecmap/eval_translation.py "'+\
					env+'" "'+welv+'" '+\
					'-d '+args.testdict+' --retrieval '+opt

					print('=== CMD ===')
					print(runstr)
					print('--------')
					#os.system(runstr)
					res = os.popen(runstr).read()

					with open(os.path.join(args.results_folder,f'model={enprefix}__retrieval={opt}__s={ens}_mc={enmc}_w={enw}.txt'),'w') as outf:
						outf.write(res)