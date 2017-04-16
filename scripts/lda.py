import argparse
from gensim import corpora, models, similarities
import pickle
import pandas as pd

def fit(data, n_topics=None, n_chunk=None, n_passes=None):
    all_questions = build_corpus(data)
    dictionary = corpora.Dictionary(all_questions)
    corpus = [dictionary.doc2bow(text) for text in all_questions]
    lda = models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, update_every=1,
                          chunksize=n_chunk, passes=n_passes)
    lda.save("lda")

def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['question1', 'question2']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
    return corpus

def get_topic_lda(row):
    lda = models.LdaModel.load('lda')
    sent1 = row['question1'].split()
    sent2 = row['question2'].split()
    
    sent1_lda = max(lda[dictionary.doc2bow(sent1)], key=lambda x: x[1])[0]
    sent2_lda = max(lda[dictionary.doc2bow(sent2)], key=lambda x: x[1])[0]
    return sent1_lda == sent2_lda

def get_topic_prob(row):
    lda = models.LdaModel.load('lda')
    sent1 = row['question1'].split()
    sent2 = row['question2'].split()
    
    sent1_lda = np.array(lda.get_document_topics(dictionary.doc2bow(sent1),
                                                 minimum_probability=0))[:, 1]
    sent2_lda = np.array(lda.get_document_topics(dictionary.doc2bow(sent2),
                                                 minimum_probability=0))[:, 1]
    return np.dot(sent1_lda, sent2_lda)

def transform(data, output):
    data['lda_topic'] = data.apply(get_topic_lda, axis=1)
    data['lda_one_topic'] = data.apply(get_topic_prob, axis=1)
    result = data[['lda_topic', 'lda_topic_prob']]
    result.to_pickle(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fit')
    parser.add_argument('-t', '--transform')
    parser.add_argument('-o', '--output')
    parser.add_argument('--n_topics', type=int, default=10)
    parser.add_argument('--n_chunk', type=int, default=1000)
    parser.add_argument('--n_passes', type=int, default=10)
    args = parser.parse_args()

    if args.fit is not None:
        fit(args.fit)
    if args.transform is not None:
        if args.output is None:
            print('No output file specified.')
            exit()
        transform(args.transform, args.output,
                  n_topics=arg.n_topics, n_chunk=arg.n_chunk,
                  n_passes=arg.n_passes)
