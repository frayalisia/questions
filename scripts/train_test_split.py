#!/usr/bin/env python

import argparse
import os
import csv

import pandas as pd
from sklearn.model_selection import StratifiedKFold


TARGET = 'is_duplicate'


def train_test_split(data_file, n_splits, output_dir, random_state=None):
    data = pd.read_csv(data_file, index_col=0)
    # Create necessary folders.
    for fold in range(1, n_splits + 1):
        name = os.path.join(output_dir, 'f{}'.format(fold))
        os.makedirs(name, exist_ok=True)
    # Split.
    cv_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train, test) in enumerate(cv_iter.split(data, data[TARGET]), start=1):
        folder = os.path.join(output_dir, 'f{}'.format(fold))
        data.iloc[train].to_csv(os.path.join(folder, 'train.csv'), quoting=csv.QUOTE_ALL)
        data.iloc[test].to_csv(os.path.join(folder, 'test.csv'), quoting=csv.QUOTE_ALL)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True)
    parser.add_argument('-s', '--splits', type=int, default=5)
    parser.add_argument('-o', '--output', type=str, default='.')
    parser.add_argument('-r', '--seed', type=int, default=0)
    args = parser.parse_args()

    train_test_split(args.data, args.splits, args.output, random_state=args.seed)
