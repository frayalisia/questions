import argparse
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

POS_TRANSFORM = {'JJ':'a', 'JJR':'a', 'JJS':'a', 'NN':'n', 'NNS':'n', 'NNP':'n', 'NNPS':'n', 'RB':'r', 
                 'RBR':'r', 'RBS':'r', 'VB':'v', 'VBD':'v', 'VBG':'v', 'VBN':'v', 'VBP':'v', 'VBZ':'v'}

LEMMATIZER = WordNetLemmatizer()

def fit(data):
    pass

def lemmatize(tagged_token):
    token, tag = tagged_token
    return lemmatizer.lemmatize(token, pos=POS_TRANSFORM.get(tag, 'n'))

def transform_data(data, lemmas=False, drop_stop_words=False):
    clean_text = re.sub("[^\w]", ' ', data)
    clean_text = clean_text.lower().split()
#   lemmatization    
    if lemmas:
        clean_text = list(map(lemmatize, nltk.pos_tag(clean_text)))
#     stopwords
    if drop_stop_words:
        clean_text = [word for word in clean_text if word not in stopwords.words('english')]
    return (' '.join(clean_text))

def transform(data, output, lemmas=False, drop_stop_words=False):
    data['question1'] = data['question1'].apply(
        transform_data, lemmas=False, drop_stop_words=False)
    data['question2'] = data['question2'].apply(
        transform_data, lemmas=False, drop_stop_words=False)
    data.to_pickle(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fit')
    parser.add_argument('-t', '--transform')
    parser.add_argument('-o', '--output')
    parser.add_argument('--lemmas', action='store_true', default=False)
    parser.add_argument('--drop_stop_words', action='store_true', default=False)
    args = parser.parse_args()

    if args.fit is not None:
        fit(args.fit)
    if args.transform is not None:
        if args.output is None:
            print('No output file specified.')
            exit()
        transform(args.transform, args.output,
                  lemmas=args.lemmas, drop_stop_words=args.drop_stop_words)
