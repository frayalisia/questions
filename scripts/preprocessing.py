import argparse
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

POS_TRANSFORM = {'JJ':'a', 'JJR':'a', 'JJS':'a', 'NN':'n', 'NNS':'n', 'NNP':'n', 'NNPS':'n', 'RB':'r', 'RBR':'r', 'RBS':'r', 'VB':'v', 'VBD':'v', 'VBG':'v', 'VBN':'v', 'VBP':'v', 'VBZ':'v'}

def fit(data):
    pass

def lemmatize(tagged_token):
    token, tag = tagged_token
    return lemmatizer.lemmatize(token, pos=POS_TRANSFORM.get(tag, 'n'))

#передавать ключи = нормализация, стоп-слова(тру-фолс)
def transform(data, output):
    lemmatizer = WordNetLemmatizer()
    clean_text = re.sub("[^\w]", ' ', text)
#     lower-case and split into words
    clean_text = clean_text.lower().split()
#   лемматизация
    clean_text = list(map(lemmatize, nltk.pos_tag(clean_text)))
#     stopwords
    words = [word for word in clean_text if word not in stopwords.words('english')]
    return (' '.join(words))


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
