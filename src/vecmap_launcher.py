import sys
import os
from argparse import ArgumentParser

def get_all_filepaths(root,forbidden_suffix='_mapped.vec'):
	for path, subdirs, files in os.walk(root):
		for name in files:
			if not name.endswith(forbidden_suffix):
				yield os.path.join(path, name)

def get_config(path):

	print('path: ',path)

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
	parser.add_argument('-td','--traindict', help='Training dictionary path', required=True)
	parser.add_argument('-sv','--source-vectors-folder', help='Source (English) embeddings folder', required=True)
	parser.add_argument('-tv','--target-vectors-folder', help='Target (Welsh) embeddings folder', required=True)

	args = parser.parse_args()


	envecs = sorted(list(get_all_filepaths(args.source_vectors_folder,forbidden_suffix='_mapped.vec')))
	welvecs = sorted(list(get_all_filepaths(args.target_vectors_folder,forbidden_suffix='_mapped.vec')))

	out = []
	for env in envecs:
		enpr, ens, enmc, enw = get_config(env)
		for welv in welvecs:
			welpr, wels, welmc, welw = get_config(welv)
			if ens==wels and enmc == welmc and enw == welw and enpr == welpr:
				print('EN: ',env)
				print('WEL: ',welv)
				
				runstr = "python3 vecmap/map_embeddings.py --supervised "+\
				args.traindict+' "'+env+'" "'+welv+'" "'+\
				env+'_mapped.vec" "'+welv+'_mapped.vec"'

				print('=== CMD ===')
				print(runstr)
				os.system(runstr)
				out.append(runstr)

	outfile = 'cmds_log.txt'
	with open(outfile,'w') as outf:
		for x in out:
			outf.write(x+'\n')