"""
Usage: split_and_convert.py [options]

Options:
-i, --input=FILE     input file [default: /fs/clip-psych/shing/task_B_user_embedding_with_binary_pretrain/task_B.train.all.prediction]
-o, --output=DIR     output dir [default: ./]
-s, --seed=SEED      random seed  [default: 19937]
-p, --prob=PROB      test probability [default: 0.2]
"""

import json
from docopt import docopt
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os

score_mapping = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3
}


def split_and_convert_to_libsvm_format(path, stratified_shuffle_split, outdir):
    out_path_train = os.path.join(outdir, 'train.svm')
    out_path_dev = os.path.join(outdir, 'dev.svm')
    X = []
    Y = []
    with open(path, 'r') as inputs:
        for line in inputs:
            jsonl = json.loads(line)
            user_embedding = jsonl['user_embedding']
            true_label = score_mapping[jsonl['true_label']]
            X.append(user_embedding)
            Y.append(true_label)

    X = np.array(X)
    Y = np.array(Y)

    with open(out_path_train, 'w') as outputs_train, open(out_path_dev, 'w') as outputs_dev:
        for train_index, test_index in stratified_shuffle_split.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            for _x, _y in zip(X_train, Y_train):
                outputs_train.write('{} {} {}\n'.format(int(_y),
                                                        'qid:1',
                                                        ' '.join(['{}:{}'.format(idx + 1, float(feature))
                                                                  for idx, feature in enumerate(_x)])))

            for _x, _y in zip(X_test, Y_test):
                outputs_dev.write('{} {} {}\n'.format(int(_y),
                                                      'qid:1',
                                                      ' '.join(['{}:{}'.format(idx + 1, float(feature))
                                                                for idx, feature in enumerate(_x)])))

    return out_path_train, out_path_dev


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

    seed = int(args['--seed'])
    p_test = float(args['--prob'])
    np.random.seed(seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=p_test, random_state=seed)

    train_path_jsonl = args['--input']
    out_dir = args['--output']
    train_split_path, dev_split_path = split_and_convert_to_libsvm_format(train_path_jsonl, sss, out_dir)
