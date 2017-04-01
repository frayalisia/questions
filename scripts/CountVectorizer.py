import argparse
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.metrics.pairwise import paired_distances
import pandas as pd

def fit(data):
    vectorizer = CountVectorizer(analyzer="word", max_features=5000)
    all_questions = pd.concat([data['question1'], data['question2']])
    model = vectorizer.fit(all_questions)
    with open('./vectorizer.pkl', 'wb') as outfile:
        pickle.dump(model, outfile)

def get_cosine_similarities(model, data):
    question1 = model.transform(data['question1']).toarray()
    question2 = model.transform(data['question2']).toarray()
    return paired_distances(question1, question2, metric='cosine')

def transform(data, output):
    with open('./vectorizer.pkl', 'rb') as infile:
        model = pickle.load(infile)
    cosines = get_cosine_similarities(model, data)
    result = pd.DataFrame(data=cosines, index=data.index, columns=['cos_sim'])
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
