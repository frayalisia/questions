from collections import defaultdict
import os
import pickle
import re

import luigi
import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances
import spacy


class CSVFile(luigi.ExternalTask):
    name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget('{}.csv'.format(self.name))
    

class Preprocess(luigi.Task):
    sample = luigi.Parameter()
    lemmas = luigi.BoolParameter()
    drop_stop_words = luigi.BoolParameter()

    pos_transform = {
        'JJ':'a', 'JJR':'a', 'JJS':'a', 'NN':'n', 'NNS':'n', 'NNP':'n',
        'NNPS':'n', 'RB':'r', 'RBR':'r', 'RBS':'r', 'VB':'v', 'VBD':'v',
        'VBG':'v', 'VBN':'v', 'VBP':'v', 'VBZ':'v'}
    lemmatizer = WordNetLemmatizer()

    def lemmatize(self, tagged_token):
        token, tag = tagged_token
        return self.lemmatizer.lemmatize(token, pos=self.pos_transform.get(tag, 'n'))

    def preprocess(self, text):
        clean_text = re.sub("[^\w]", ' ', text).lower().split()
        if self.lemmas:
            clean_text = list(map(self.lemmatize, pos_tag(clean_text)))
        if self.drop_stop_words:
            clean_text = [word for word in clean_text
                          if word not in stopwords.words('english')]
        return ' '.join(clean_text)

    def requires(self):
        return CSVFile(self.sample)

    def output(self):
        postfix = '_preprocessed'
        if self.lemmas:
            postfix += '_lemmas'
        if self.drop_stop_words:
            postfix += '_nostop'
        return luigi.LocalTarget('{0}{1}.pkl'.format(self.sample, postfix))

    def run(self):
        sample_path = self.input().path
        data = pd.read_csv(sample_path, index_col=0)
        data['question1'] = data['question1'].apply(self.preprocess)
        data['question2'] = data['question2'].apply(self.preprocess)
        data.to_pickle(self.output().path)


class RawFeatures(luigi.Task):
    sample = luigi.Parameter()

    def agree(self, data, regex):
        s1 = data['question1'].str.contains(regex)
        s2 = data['question2'].str.contains(regex)
        return (s1 == s2)

    def contain(self, data, regex):
        s1 = data['question1'].str.contains(regex)
        s2 = data['question2'].str.contains(regex)
        return (s1 | s2)

    def len_diff(self, data, unit='symbol'):
        if unit == 'symbol':
            s1 = data['question1'].str.len()
            s2 = data['question2'].str.len()
        else:
            s1 = data['question1'].str.split().str.len()
            s2 = data['question2'].str.split().str.len()
        abs_diff = (s1 - s2).abs()
        rel_diff = abs_diff / pd.concat([s1, s2], axis=1).max(axis=1)
        return abs_diff, rel_diff

    def get_unicode_features(self, row):
        s1 = row['question1']
        s2 = row['question2']
        ord1 = list(map(ord, s1))
        ord2 = list(map(ord, s2))
        uni1 = set(filter(lambda s: s >= 256, ord1))
        uni2 = set(filter(lambda s: s >= 256, ord2))

        features = {}
        features['unicode_jaccard'] = len(uni1 & uni2) / len(uni1 | uni2) if (uni1 | uni2) else np.nan
        features['diff_sum_ord_abs'] = abs(sum(ord1) - sum(ord2))
        features['diff_sum_ord_rel'] = features['diff_sum_ord_abs'] / max(sum(ord1), sum(ord2))

        return pd.Series(features)


    def requires(self):
        return CSVFile(self.sample)

    def output(self):
        return luigi.LocalTarget('{}_basic.pkl'.format(self.sample))

    def run(self):
        sample_path = self.input().path
        data = pd.read_csv(sample_path, index_col=0)
        data['agree_qmark'] = self.agree(data, '\?')
        data['agree_digit'] = self.agree(data, '\d+')
        data['agree_math'] = self.agree(data, '\[math\]')
        data['contain_math'] = self.contain(data, '\[math\]')
        data['diff_len_symb_abs'], data['diff_len_symb_rel'] = self.len_diff(data, 'symbol')
        data['diff_len_word_abs'], data['diff_len_word_rel'] = self.len_diff(data, 'word')

        unicode_features = data.apply(self.get_unicode_features, axis=1)

        result = data[['agree_qmark', 'agree_digit', 'agree_math', 'contain_math',
                       'diff_len_symb_abs', 'diff_len_symb_rel',
                       'diff_len_word_abs', 'diff_len_word_rel']]
        result = pd.concat([result, unicode_features], axis=1)
        result.to_pickle(self.output().path)


class Jaccard(luigi.Task):
    sample = luigi.Parameter()
    lemmas = luigi.BoolParameter()
    drop_stop_words = luigi.BoolParameter()

    def get_jaccard_features(self, row):
        s1 = set(row['question1'].split())
        s2 = set(row['question2'].split())

        features = {}
        if (s1 | s2):
            features['sim'] = len(s1 & s2) / len(s1 | s2)
            features['diff'] = min(len(s1 - s2), len(s2 - s1)) / len(s1 | s2)
        else:
            features['sim'] = np.nan
            features['diff'] = np.nan
        return pd.Series(features)

    def requires(self):
        return Preprocess(sample=self.sample, lemmas=self.lemmas,
                          drop_stop_words=self.drop_stop_words)

    def output(self):
        postfix = '_jaccard'
        if self.lemmas:
            postfix += '_lemmas'
        if self.drop_stop_words:
            postfix += '_nostop'
        return luigi.LocalTarget('{0}{1}.pkl'.format(self.sample, postfix))

    def run(self):
        data = pd.read_pickle(self.input().path)
        jaccard_features = data.apply(self.get_jaccard_features, axis=1)
        jaccard_features.to_pickle(self.output().path)


class BagOfWordsModel(luigi.Task):
    corpus = luigi.Parameter()
    tf_idf = luigi.BoolParameter()
    max_features = luigi.Parameter(default=None)

    def requires(self):
        return Preprocess(sample=self.corpus, lemmas=True, drop_stop_words=True)

    def output(self):
        postfix = '_bow_model'
        if self.tf_idf:
            postfix += '_tfidf'
        if self.max_features is not None:
            postfix += '_{}'.format(self.max_features)
        return luigi.LocalTarget('{0}{1}.pkl'.format(self.corpus, postfix))

    def run(self):
        data = pd.read_pickle(self.input().path)
        corpus = pd.concat([data['question1'], data['question2']])

        if self.max_features is not None:
            self.max_features = int(self.max_features)
        params = {'analyzer' : str.split, 'max_features' : self.max_features}

        if self.tf_idf:
            vectorizer = TfidfVectorizer(**params)
        else:
            vectorizer = CountVectorizer(**params)

        model = vectorizer.fit(corpus)
        with open(self.output().path, 'wb') as outfile:
            pickle.dump(model, outfile)


class BagOfWordsFeatures(luigi.Task):
    sample = luigi.Parameter()
    tf_idf = luigi.BoolParameter()
    max_features = luigi.Parameter(default=None)

    def get_cosine_similarities(self, model, data):
        question1 = model.transform(data['question1'])
        question2 = model.transform(data['question2'])
        return 1.0 - paired_distances(question1, question2, metric='cosine')

    def requires(self):
        return [BagOfWordsModel(tf_idf=self.tf_idf, max_features=self.max_features),
                Preprocess(sample=self.sample, lemmas=True, drop_stop_words=True)]

    def output(self):
        postfix = '_bow'
        if self.tf_idf:
            postfix += '_tfidf'
        if self.max_features is not None:
            postfix += '_{}'.format(self.max_features)
        return luigi.LocalTarget('{0}{1}.pkl'.format(self.sample, postfix))

    def run(self):
        model_file, data_file = self.input()
        model = pd.read_pickle(model_file.path)
        data = pd.read_pickle(data_file.path)
        cosines = self.get_cosine_similarities(model, data)
        features = pd.DataFrame(data=cosines, index=data.index, columns=['sim'])
        features.to_pickle(self.output().path)


class Spacy(luigi.Task):
    sample = luigi.Parameter()
    
    nlp = spacy.load('en')
    ent2group = {
        'GPE' : 'LOC',
        'LOC' : 'LOC',
        'FACILITY' : 'LOC',

        'PERSON' : 'PERSON',

        'ORG' : 'ORG',

        'PRODUCT' : 'OTHER',
        'EVENT' : 'OTHER',
        'WORK_OF_ART' : 'OTHER',
        'LANGUAGE' : 'OTHER'
    }

    def set_features(self, s1, s2):
        agree = (bool(s1) == bool(s2))
        if (s1 | s2):
            jac_sim = len(s1 & s2) / len(s1 | s2)
        else:
            jac_sim = np.nan
        return agree, jac_sim
    
    def spacy_features(self, row):
        features = {}
        doc1 = self.nlp(row['question1'])
        doc2 = self.nlp(row['question2'])

        features['sim'] = doc1.similarity(doc2)

        ent1 = defaultdict(set)
        for el in doc1:
            ent1[self.ent2group.get(el.ent_type_, 'NONE')].add(el.text)
        ent2 = defaultdict(set)
        for el in doc2:
            ent2[self.ent2group.get(el.ent_type_, 'NONE')].add(el.text)

        for group in ['LOC', 'PERSON', 'ORG', 'OTHER']:
            agree, jac_sim = self.set_features(ent1[group], ent2[group])
            features['agree_{}'.format(group)] = agree
            features['jac_sim_{}'.format(group)] = jac_sim
        return pd.Series(features)

    def requires(self):
        return CSVFile(self.sample)

    def output(self):
        return luigi.LocalTarget('{}_spacy.pkl'.format(self.sample))

    def run(self):
        sample_path = self.input().path
        data = pd.read_csv(sample_path, index_col=0)
        
        spacy_features = data.apply(self.spacy_features, axis=1)
        spacy_features.to_pickle(self.output().path)


class Word2VecModel(luigi.Task):
    corpus = luigi.Parameter()

    def requires(self):
        return Preprocess(sample=self.corpus, lemmas=True, drop_stop_words=True)

    def output(self):
        postfix = '_w2v_model'
        return luigi.LocalTarget('{0}{1}.bin'.format(self.corpus, postfix))

    def run(self):
        data = pd.read_pickle(self.input().path)
        corpus = (data['question1'].str.split().tolist() +
                  data['question2'].str.split().tolist())
        model = word2vec.Word2Vec(corpus, size=300, window=20, min_count=2, workers=16)
        model.wv.save_word2vec_format(self.output().path, binary=True)


class Word2VecFeatures(luigi.Task):
    corpus = luigi.Parameter()
    sample = luigi.Parameter()

    def get_similarity(self, row, model=None):
        s1 = [w for w in row['question1'].split() if w in model.vocab]
        s2 = [w for w in row['question2'].split() if w in model.vocab]
        if (s1 and s2):
            return model.n_similarity(s1, s2)
        return np.nan

    def requires(self):
        return [Word2VecModel(corpus=self.corpus),
                Preprocess(sample=self.sample, lemmas=True, drop_stop_words=True)]

    def output(self):
        postfix = '_w2v'
        postfix += '_' + os.path.basename(self.corpus)
        return luigi.LocalTarget('{0}{1}.pkl'.format(self.sample, postfix))

    def run(self):
        model_file, data_file = self.input()
        model = KeyedVectors.load_word2vec_format(model_file.path, binary=True)
        data = pd.read_pickle(data_file.path)
        cosines = data.apply(self.get_similarity, model=model, axis=1)
        features = pd.DataFrame(data=cosines, index=data.index, columns=['sim'])
        features.to_pickle(self.output().path)


class CollectFeatures(luigi.Task):
    sample = luigi.Parameter()

    def requires(self):
        return {
            'RAW' : RawFeatures(sample=self.sample),
            'JAC' : Jaccard(sample=self.sample, lemmas=True, drop_stop_words=True),
            'SPACY' : Spacy(sample=self.sample),
            'BOW_COUNT' : BagOfWordsFeatures(sample=self.sample),
            'BOW_TFIDF' : BagOfWordsFeatures(sample=self.sample, tf_idf=True),
            'W2V_CORPUS' : Word2VecFeatures(sample=self.sample, corpus='./nlp_models/corpus'),
            'W2V_GOOGLE' : Word2VecFeatures(sample=self.sample, corpus='./nlp_models/google_news')
        }, CSVFile(name=self.sample)

    def output(self):
        return luigi.LocalTarget('{}_features.pkl'.format(self.sample))
    
    def run(self):
        features_dict, sample = self.input()
        features_list = []
        for prefix, features_file in features_dict.items():
            features = pd.read_pickle(features_file.path).add_prefix(prefix + '_')
            features_list.append(features)
        target = pd.read_csv(sample.path, index_col=0, usecols=[0, 5],
                             names=['id', 'TARGET'], header=0)
        df = pd.concat(features_list + [target], axis=1)
        df.to_pickle(self.output().path)


if __name__ == '__main__':
    luigi.run(main_task_cls=CollectFeatures)
