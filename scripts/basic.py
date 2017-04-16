import argparse
import pickle
import pandas as pd

def fit(data):
    pass

def contain_str_mark(table):
    sent1 = table['question1'].str.contains("\?")
    sent2 = table['question2'].str.contains("\?")
    table['contain_questionmark'] = (sent1 == sent2)

def contain_str_digit(table):
    sent1 = table['question1'].str.contains("\d+")
    sent2 = table['question2'].str.contains("\d+")
    table['contain_digit'] = (sent1 == sent2)

def contain_str_math(table):
    sent1 = table['question1'].str.contains("\[math\]")
    sent2 = table['question2'].str.contains("\[math\]")
    table['contain_math'] = (sent1 == sent2)

def diff_len_symb(table):
    sent1 = table['question1'].str.len()
    sent2 = table['question2'].str.len()
    
    table['diff_len_symb_abs'] = (sent1 - sent2).abs()
    table['diff_len_symb_rel'] = table['diff_len_symb_abs'] / pd.concat([sent1, sent2], axis=1).max(axis=1)

def diff_len_word(table):
    sent1 = table['question1'].str.split().str.len()
    sent2 = table['question1'].str.split().str.len()
    
    table['diff_len_word_abs'] = (sent1 - sent2).abs()
    table['diff_len_word_rel'] = table['diff_len_word_abs'] / pd.concat([sent1, sent2], axis=1).max(axis=1)

def transform(data, output):
    contain_str_digit(data)
    contain_str_mark(data)
    contain_str_math(data)
    diff_len_symb(data)
    diff_len_word(data)
    result = data[['contain_questionmark', 'contain_math', 'contain_digit', 'diff_len_symb_abs', 'diff_len_symb_rel', 'diff_len_word_abs', 'diff_len_word_rel']]
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
