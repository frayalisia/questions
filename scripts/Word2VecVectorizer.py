import argparse
from gensim.models import word2vec
import pickle
import pandas as pd

def fit(data):
    all_questions = build_corpus(data) 
    vectorizer = word2vec.Word2Vec(all_questions, size=300, window=20, min_count=2, workers=4)
    model.save("word2vec")

def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['question1', 'question2']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
    return corpus

def w2v_sim(row):
    model = word2vec.Word2Vec.load('word2vec')
    sent1 = [w for w in row['question1'].split() if w in model.wv.vocab]
    sent2 = [w for w in row['question2'].split() if w in model.wv.vocab]
    if sent1 and sent2:
        return model.n_similarity(sent1, sent2)
    return np.nan

def transform(data, output):
    cosines = data.apply(w2v_sim, axis=1)
    result = cosines.to_frame(name='cos_sim')
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
