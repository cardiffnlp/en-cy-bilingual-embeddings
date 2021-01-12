import sys
import os
from argparse import ArgumentParser
from collections import defaultdict

def get_all_filepaths(root,forbidden_suffix='_mapped.vec'):
    for path, subdirs, files in os.walk(root):
        for name in files:
            if not name.endswith(forbidden_suffix):
                yield os.path.join(path, name)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-td','--dict', help='Training dictionary path', required=True)
    parser.add_argument('-sv','--source-vectors-folder', help='Source (English) embeddings folder', required=True)
    parser.add_argument('-tv','--target-vectors-folder', help='Target (Welsh) embeddings folder', required=True)
    parser.add_argument('-o','--output-folder', help='Folder where vocab (EN and CY) is stored', required=True)

    args = parser.parse_args()

    source_vecs = sorted(list(get_all_filepaths(args.source_vectors_folder,forbidden_suffix='_mapped.vec')))
    target_vecs = sorted(list(get_all_filepaths(args.target_vectors_folder,forbidden_suffix='_mapped.vec')))

    source_vocab = set()
    target_vocab = set()

    for inf in source_vecs:
        print(f'Processing {inf}')
        with open(inf) as f:
            lc = 0
            for line in f:
                if lc != 0:
                    w = line.split()[0]
                    source_vocab.add(w)
                lc += 1
    for inf in target_vecs:
        print(f'Processing {inf}')
        with open(inf) as f:
            lc = 0
            for line in f:
                if lc != 0:
                    w = line.split()[0]
                    target_vocab.add(w)
                lc += 1
    print(f'Source vocab is {len(source_vocab)} tokens')
    print(f'Target vocab is {len(target_vocab)} tokens')

    with open(os.path.join(args.output_folder, 'source_vocab.txt'),'w') as outf:
        for a in source_vocab:
            outf.write(a+'\n')
    with open(os.path.join(args.output_folder, 'target_vocab.txt'),'w') as outf:
        for a in target_vocab:
            outf.write(a+'\n')