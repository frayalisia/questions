import argparse
from gensim.models.keyedvectors import KeyedVectors
import pickle
import pandas as pd

def fit(data):
    pass

def w2v_sim(row):
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    sent1 = [w for w in row['question1'] if w in model.vocab]
    sent2 = [w for w in row['question2'] if w in model.vocab]
    if sent1 and sent2:
        return model.n_similarity(sent1, sent2)
    return np.nan

def transform(data, output):
    cosines = data.apply(w2v_sim, axis=1)
    result = cosines.to_frame(name='cos_sim_google')
    result.to_pickle(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fit')
    parser.add_argument('-t', '--transform')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    if args.fit is not None:
        fit(args.fit)
    if args.transform is not None:
        if args.output is None:
            print('No output file specified.')
            exit()
        transform(args.transform, args.output)
