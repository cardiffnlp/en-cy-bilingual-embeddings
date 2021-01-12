import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import gensim
from argparse import ArgumentParser

def parse_dict(dpath):
    pairs = []
    for line in open(dpath):
        cols = line.strip().split('\t')
        if len(cols) == 2:
            weng,wwel = cols
            pairs.append((weng,wwel))
    return pairs

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dict', help='Dictionary', required=True)
    parser.add_argument('-sv','--source-vocab', help='Source vocab', required=True)
    parser.add_argument('-tv','--target-vocab', help='Target vocab', required=True)
    parser.add_argument('-o','--output-folder', help='Directory to store train and test dictionaries', required=True)

    args = parser.parse_args()

    pairs = parse_dict(args.dict)

    print('Loading source vocab')
    source_vocab = set([line.strip() for line in open(args.source_vocab)])
    print(f'Source vocab has {len(source_vocab)} words')
    print('---')
    print('Loading target vocab')
    target_vocab = set([line.strip() for line in open(args.target_vocab)])
    print(f'Target vocab has {len(target_vocab)} words')

    # Split dictionary based on predefined vocab

    fpairs = []
    i = 0
    for engw,welw in pairs:
        if engw in source_vocab and welw in target_vocab:
            fpairs.append((engw,welw))
        i += 1
        if i % 1000 == 0:
            print(f'Loading dictionary - Done {i} of {len(pairs)}')

    print(f'Loaded {len(fpairs)} entries from the original dictionary, which had: {len(pairs)}')

    fpairs = list(set(fpairs))

    fdf = pd.DataFrame(fpairs)
    fdf.columns = ['english', 'welsh']
    train_eng, test_eng, train_wel, test_wel = train_test_split(fdf.english, fdf.welsh, test_size = 0.2, shuffle=True)

    # save train dictionary
    out_train = pd.DataFrame(columns=['english', 'welsh'])
    out_train.english = train_eng
    out_train.welsh = train_wel
    out_train.to_csv(os.path.join(args.output_folder,'train_dict.csv'),
                     index=False, sep=' '
                    )
    # save test dictionary
    out_test = pd.DataFrame(columns=['english', 'welsh'])
    out_test.english = test_eng
    out_test.welsh = test_wel
    out_test.to_csv(os.path.join(args.output_folder,'test_dict.csv'),
                     index=False,sep=' '
                    )

