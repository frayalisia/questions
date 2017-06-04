from collections import defaultdict
import os
import pickle
import re
import sys
import tempfile

import luigi
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.matutils import unitvec
from gensim.models import word2vec
from gensim.models import LdaMulticore as GensimLdaModel
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import paired_distances
from sklearn.model_selection import StratifiedKFold
import spacy
from fuzzywuzzy import fuzz as fz


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

    stops = set(stopwords.words('english'))

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

    def get_hamming_features(self, row):
        s1 = row['question1'].split()
        s2 = row['question2'].split()
        
        features = {}
        features['ham'] = sum(1 for i in zip(s1, s2) if i[0]==i[1]) / max(len(s1), len(s2))
        features['ham_ints'] = sum(1 for i in zip(s1, s2) if i[0]==i[1]) / len(set(s1) | set(s2))
        return pd.Series(features)

    def get_avg_features(self, data):
        s1_symb = data['question1'].str.len()
        s2_symb = data['question2'].str.len()
        s1_word = data['question1'].str.split().str.len()
        s2_word = data['question2'].str.split().str.len()

        s1_avg_word = s1_symb / s1_word
        s2_avg_word = s2_symb / s2_word
        avg_word_min = np.minimum(s1_avg_word, s2_avg_word)
        avg_word_max = np.maximum(s1_avg_word, s2_avg_word)
        avg_word_diff = avg_word_max - avg_word_min
        avg_word_mean = (avg_word_min + avg_word_max) / 2
        return avg_word_min, avg_word_max, avg_word_diff, avg_word_mean

    def get_stop_ratio(self, row):
        s1 = set(str(row['question1']).lower().split())
        s2 = set(str(row['question2']).lower().split())
    
        s1words = s1.difference(self.stops)
        s2words = s2.difference(self.stops)

        s1stops = s1.intersection(self.stops)
        s2stops = s2.intersection(self.stops)
    
        features = {}
        stops_s1 = len(s1stops) / len(s1words) if s1words else np.nan
        stops_s2 = len(s2stops) / len(s2words) if s2words else np.nan
        features['stops_min'], features['stops_max'] = np.sort([stops_s1, stops_s2])
        return pd.Series(features)

    def get_fuzzy_features(self, row):
        s1 = row['question1']
        s2 = row['question2']
        
        features = {}
        features['simple_ratio'] = fz.ratio(s1, s2)
        features['partial_ratio'] = fz.partial_ratio(s1, s2)
        features['part_tok_set_ratio'] = fz.partial_token_set_ratio(s1, s2)
        features['part_tok_sort_ratio'] = fz.partial_token_sort_ratio(s1, s2)
        features['tok_sort_ratio'] = fz.token_sort_ratio(s1, s2)
        features['tok_set_ratio'] = fz.token_set_ratio(s1, s2)
        features['Wratio'] = fz.WRatio(s1, s2)
        return pd.Series(features)

    def get_loc_features(self, row, loc_regex=None):
        s1 = set(loc_regex.findall(row['question1'].lower()))
        s2 = set(loc_regex.findall(row['question2'].lower()))

        features = {}
        features['loc_jaccard'] = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else np.nan
        features['loc_agree'] = (s1 == s2)
        features['loc_min'] = min(len(s1), len(s2))
        features['loc_max'] = max(len(s1), len(s2))
        return pd.Series(features)
    
    def requires(self):
        return CSVFile(self.sample)

    def output(self):
        return luigi.LocalTarget('{}_raw.pkl'.format(self.sample))

    def run(self):
        sample_path = self.input().path
        data = pd.read_csv(sample_path, index_col=0)
        features = pd.DataFrame(index=data.index)
        features['agree_qmark'] = self.agree(data, '\?')
        features['agree_digit'] = self.agree(data, '\d+')
        features['agree_math'] = self.agree(data, '\[math\]')
        features['contain_math'] = self.contain(data, '\[math\]')
        for el in ['how', 'what', 'which', 'who', 'whom', 'where', 'when', 'why']:
            features['agree_{}'.format(el)] = self.agree(data, el)
            features['contain_{}'.format(el)] = self.contain(data, el)
        features['diff_len_symb_abs'], features['diff_len_symb_rel'] = self.len_diff(data, 'symbol')
        features['diff_len_word_abs'], features['diff_len_word_rel'] = self.len_diff(data, 'word')
        features['avg_word_min'], features['avg_word_max'], features['avg_word_diff'], features['avg_word_mean'] = self.get_avg_features(data)

        unicode_features = data.apply(self.get_unicode_features, axis=1)
        hamming_features = data.apply(self.get_hamming_features, axis=1)
        stops_features = data.apply(self.get_stop_ratio, axis=1)
        fuzzy_features = data.apply(self.get_fuzzy_features, axis=1)

        with open(os.path.join(sys.path[0], 'regex_geo.pickle'), 'rb') as infile:
            loc_regex = pickle.load(infile)
        loc_features = data.apply(self.get_loc_features, axis=1, loc_regex=loc_regex)

        features = pd.concat([features, unicode_features, hamming_features, stops_features, fuzzy_features, loc_features], axis=1)
        features.to_pickle(self.output().path)


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
    corpus = luigi.Parameter()
    sample = luigi.Parameter()
    tf_idf = luigi.BoolParameter()
    max_features = luigi.Parameter(default=None)

    def get_cosine_similarities(self, model, data):
        question1 = model.transform(data['question1'])
        question2 = model.transform(data['question2'])
        return 1.0 - paired_distances(question1, question2, metric='cosine')

    def requires(self):
        return [BagOfWordsModel(corpus=self.corpus, tf_idf=self.tf_idf, max_features=self.max_features),
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

    def get_features(self, row, model=None, vectorizer=None):
        s1 = [w for w in row['question1'].split() if w in model.vocab]
        s2 = [w for w in row['question2'].split() if w in model.vocab]

        features = {}
        if not (s1 and s2):
            features['sim'] = np.nan
            features['wm_dist'] = np.nan
            features['tfidf_sim'] = np.nan
            features['min_dist'] = np.nan
            features['max_dist'] = np.nan
            features['mean_dist'] = np.nan
            features['std_dist'] = np.nan
        else:
            # similarity
            features['sim'] = model.n_similarity(s1, s2)
            # wm_distance
            features['wm_dist'] = model.wmdistance(s1, s2)
            # tfidf similarity
            ids1 = [vectorizer.vocabulary_[w] for w in s1]
            ids2 = [vectorizer.vocabulary_[w] for w in s2]
            tf_idfs1 = vectorizer.transform([row['question1']])[:, ids1].toarray().flatten()
            tf_idfs2 = vectorizer.transform([row['question2']])[:, ids2].toarray().flatten()
            vecs1 = model[s1]
            vecs2 = model[s2]
            features['tfidf_sim'] = unitvec(tf_idfs1 @ vecs1) @ unitvec(tf_idfs2 @ vecs2)
            # distances
            dist_matrix = cdist(vecs1, vecs2, metric='cosine')
            features['min_dist'] = dist_matrix.min()
            features['max_dist'] = dist_matrix.max()
            features['mean_dist'] = dist_matrix.mean()
            features['std_dist'] = dist_matrix.std()
        return pd.Series(features)

    def requires(self):
        return [Word2VecModel(corpus=self.corpus),
                BagOfWordsModel(corpus=self.corpus, tf_idf=True),
                Preprocess(sample=self.sample, lemmas=True, drop_stop_words=True)]

    def output(self):
        postfix = '_w2v'
        postfix += '_' + os.path.basename(self.corpus)
        return luigi.LocalTarget('{0}{1}.pkl'.format(self.sample, postfix))

    def run(self):
        model_file, vectorizer_file, data_file = self.input()
        model = KeyedVectors.load_word2vec_format(model_file.path, binary=True)
        vectorizer = pd.read_pickle(vectorizer_file.path)
        data = pd.read_pickle(data_file.path)
        features = data.apply(self.get_features, model=model, vectorizer=vectorizer, axis=1)
        features.to_pickle(self.output().path)


class LdaModel(luigi.Task):
    corpus = luigi.Parameter()
    num_topics = luigi.IntParameter(default=10)
    passes = luigi.IntParameter(default=10)

    def requires(self):
        return Preprocess(sample=self.corpus, lemmas=True, drop_stop_words=True)

    def output(self):
        postfix = '_lda_model_{0}_{1}'.format(self.num_topics, self.passes)
        return [luigi.LocalTarget('{0}{1}.bin'.format(self.corpus, postfix)),
                luigi.LocalTarget('{0}{1}_dict.bin'.format(self.corpus, postfix))]

    def run(self):
        data = pd.read_pickle(self.input().path)
        sentences = (data['question1'].str.split().tolist() +
                     data['question2'].str.split().tolist())
        dictionary = corpora.Dictionary(sentences)
        corpus = list(map(dictionary.doc2bow, sentences))
        lda = GensimLdaModel(corpus, num_topics=self.num_topics, id2word=dictionary,
                             chunksize=1000, passes=self.passes,
                             minimum_probability=-1.0)
        lda_file, dictionary_file = self.output()
        lda.save(lda_file.path)
        dictionary.save(dictionary_file.path)


class LdaFeatures(luigi.Task):
    corpus = luigi.Parameter()
    num_topics = luigi.IntParameter(default=10)
    passes = luigi.IntParameter(default=1)
    sample = luigi.Parameter()

    def get_features(self, row, model=None, dictionary=None):
        s1 = row['question1'].split()
        s2 = row['question2'].split()

        prob1 = np.array(model.get_document_topics(dictionary.doc2bow(s1)))[:, 1]
        prob2 = np.array(model.get_document_topics(dictionary.doc2bow(s2)))[:, 1]

        features = {}
        features['same_top_topic'] = (prob1.argmax() == prob2.argmax())
        features['same_topic_prob'] = prob1 @ prob2
        features['kullback_leibler'] = (prob1 - prob2) @ (np.log(prob1) - np.log(prob2))

        return pd.Series(features)


    def requires(self):
        return [LdaModel(corpus=self.corpus, num_topics=self.num_topics, passes=self.passes),
                Preprocess(sample=self.sample, lemmas=True, drop_stop_words=True)]

    def output(self):
        postfix= '_lda_{0}_{1}'.format(self.num_topics, self.passes)
        return luigi.LocalTarget('{0}{1}.pkl'.format(self.sample, postfix))

    def run(self):
        (model_file, dictionary_file), data_file = self.input()
        model = GensimLdaModel.load(model_file.path)
        dictionary = corpora.Dictionary.load(dictionary_file.path)
        data = pd.read_pickle(data_file.path)
        features = data.apply(self.get_features, model=model, dictionary=dictionary, axis=1)
        features.to_pickle(self.output().path)


class VowpalWabbitSplits(luigi.Task):
    sample = luigi.Parameter()
    n_splits = 5

    def to_vw_format(self, df):
        return (df.is_duplicate.astype(str) + ' ' + df.index.astype(str)
                + '|first ' + df['question1'] + ' |second ' + df['question2'])

    def requires(self):
        if self.sample.endswith('train'):
            return CSVFile(self.sample)
        else:
            train_path = self.sample[:-4] + 'train'
            return [CSVFile(train_path), CSVFile(self.sample)]

    def output(self):
        splits = []
        folds = range(1, self.n_splits + 1) if self.sample.endswith('train') else [0]
        for fold in folds:
            train = luigi.LocalTarget('{0}_vw_split_{1}_train.txt'.format(self.sample, fold))
            test = luigi.LocalTarget('{0}_vw_split_{1}_test.txt'.format(self.sample, fold))
            splits.append({'train' : train, 'test' : test})
        return splits

    def run(self):
        if self.sample.endswith('train'):
            sample_path = self.input().path
            data = pd.read_csv(sample_path, index_col=0)
            data_vw = self.to_vw_format(data)
            cv_iter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=0)
            for fold, (train, test) in enumerate(cv_iter.split(data_vw, data.is_duplicate)):
                self.output()[fold]['train'].write('\n'.join(data_vw.iloc[train]))
                self.output()[fold]['test'].write('\n'.join(data_vw.iloc[test]))
        else:
            train_path, test_path = self.input()
            train_vw = self.to_vw_format(pd.read_csv(train_path, index_col=0))
            test_vw = self.to_vw_format(pd.read_csv(test_path, index_col=0))
            self.output()[0]['train'].write('\n'.join(train_vw))
            self.output()[0]['test'].write('\n'.join(test_vw))


class VowpalWabbitFeature(luigi.Task):
    sample = luigi.Parameter()
    variant = luigi.Parameter()
    vw_args_dict = {
        'linear': ['--l1', '0.0000001', '-c', '--passes', '500', '--loss_function', 'logistic'],
        'quad': ['-b', '23', '-l', '0.4', '--l1', '0.00000001', '--l2', '0.00000001', '-q', 'fs',
                 '-c', '--passes', '500', '--loss_function', 'logistic'],
        'quad_bigram': ['-b', '26', '-l', '0.4', '--l1', '0.00000001', '--l2', '0.00000001',
                        '--ngram', '2', '-q', 'fs', '-c', '--passes', '500',
                        '--loss_function', 'logistic']
    }

    def vw_fit_predict(self, train, test, vw_args=None):
        if vw_args is None:
            vw_args = []
        temp_dir = tempfile.mkdtemp()
        subprocess.check_call(['cp', train, os.path.join(temp_dir, 'train.txt')])
        subprocess.check_call(['cp', test, os.path.join(temp_dir, 'test.txt')])
        start_dir = os.cwd()
        os.chdir(temp_dir)
        subprocess.check_call(['vw', '-d', 'train.txt', '-f', 'model.vw'] + vw_args)
        subprocess.check_call(['vw', '-d', 'test.txt', '-i', 'model.vw', '-p', 'test_preds.txt'])
        vw_pred = pd.read_csv('test_preds.txt', header=None, squeeze=True)
        os.chdir(start_dir)
        os.rmdir(temp_dir)
        return expit(vw_pred)

    def requires(self):
        return VowpalWabbitSplits(sample=self.sample)

    def output(self):
        postfix= '_vw_{0}'.format(self.variant)
        return luigi.LocalTarget('{0}{1}.pkl'.format(self.sample, postfix))

    def run(self):
        vw_pred_list = []
        for train, test in self.input():
            vw_pred_list.append(self.vw_fit_predict(
                train.path, test.path, vw_args=self.vw_args_dict[self.variant]))
        vw_pred = pd.concat(vw_pred_list)
        # TODO: check correct join order
        vw_pred.to_pickle(self.output().path)


class CollectFeatures(luigi.Task):
    sample = luigi.Parameter()

    def requires(self):
        return {
            'RAW' : RawFeatures(sample=self.sample),
            'JAC' : Jaccard(sample=self.sample, lemmas=True, drop_stop_words=True),
            'SPACY' : Spacy(sample=self.sample),
            'BOW_COUNT' : BagOfWordsFeatures(sample=self.sample, corpus='./nlp_models/corpus'),
            'BOW_TFIDF' : BagOfWordsFeatures(sample=self.sample, tf_idf=True, corpus='./nlp_models/corpus'),
            'W2V_CORPUS' : Word2VecFeatures(sample=self.sample, corpus='./nlp_models/corpus'),
            'LDA10' : LdaFeatures(sample=self.sample, corpus='./nlp_models/corpus', num_topics=10),
            'LDA100' : LdaFeatures(sample=self.sample, corpus='./nlp_models/corpus', num_topics=100),
            'LDA500' : LdaFeatures(sample=self.sample, corpus='./nlp_models/corpus', num_topics=500, passes=10)
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
