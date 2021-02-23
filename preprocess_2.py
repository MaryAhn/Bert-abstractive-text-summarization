import io, sys

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from tqdm import tqdm
import pandas as pd
import transformer.Constants as Constants
from tokenizer import FullTokenizer
from eunjeon import Mecab
import argparse

"""
import pandas.io.sql as psql
import sqlite3
import MeCab

    m = MeCab.Tagger('-Owakati')
"""


def build_vocab(path_to_file):
    with open(path_to_file, 'r', encoding='utf-8') as f:
        vocab = f.readlines()
    token2text = {k: v.rstrip() for k, v in enumerate(vocab)}
    text2token = {v: k for k, v in token2text.items()}

    return text2token, token2text


def get_content_summary_from_df(df):
    content = df['content'].values.tolist()
    summary = df['summary'].values.tolist()

    return content, summary


def convert_text_to_token(content, summary, text2token, max_len, tokenizer):
    tokens_content = []
    tokens_summary = []
    for d_content, d_summary in tqdm(zip(content, summary), ascii=True, total=len(content)):
        tokens_content.append(convert_text_to_token_seq(d_content, text2token, max_len, tokenizer))
        tokens_summary.append(convert_text_to_token_seq(d_summary, text2token, max_len, tokenizer))

    return tokens_content, tokens_summary


def convert_text_to_token_seq(text, text2token, max_len, tokenizer):
    if len(text) > 2000:
        text = text[:2000]
    mecab = Mecab()
    splited_text = mecab.morphs(text)
    # splited_text = tokenizer.tokenize(convert_num_half_to_full(text.replace('.\n', '\n').replace('\n', '.\n')))
    if len(splited_text) > (max_len - 2):
        splited_text = splited_text[:max_len - 2]
    splited_text = [Constants.BOS] + \
                   [text2token.get(i, Constants.UNK) for i in splited_text] + \
                   [Constants.EOS]
    return splited_text


def convert_num_half_to_full(text):
    table = str.maketrans({
        '0': '０',
        '1': '１',
        '2': '２',
        '3': '３',
        '4': '４',
        '5': '５',
        '6': '６',
        '7': '７',
        '8': '８',
        '9': '９',
    })
    return text.translate(table)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-vocab', required=True)
    parser.add_argument('-data', required=True) 
    parser.add_argument('-save_data', required=True)
    parser.add_argument('--max_word_seq_len',  default=100, type=int)
    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len

    tokenizer = FullTokenizer(opt.vocab)
    df = pd.read_csv(opt.data)

    print('Finished reading db file.')
    text2token, token2text = build_vocab(opt.vocab)
    print('Finished building vocab.')
    content, summary = get_content_summary_from_df(df)
    print('Finished get content and summary from df')
    content, summary = convert_text_to_token(content, summary, text2token, opt.max_token_seq_len, tokenizer)
    print('Finished convert text2token')

    data = {
        'dict': {
            'src': text2token,
            'tgt': text2token},
        'train': {
            'src': content[:80],
            'tgt': summary[:80]},
        'valid': {
            'src': content[80:],
            'tgt': summary[80:]}}

    torch.save(data, opt.save_data)
    # # test2 = pd.DataFrame(data_f)
    # # test2.to_csv("data/save.csv")
    #
    # """
    # mecab = Mecab()
    # test = "스마트폰 보급 및 네트워크 환경의 진화로 전세계으로."
    # a = mecab.morphs(test)
    # """
    #
    # print('------')
    # # print('text2token: ', text2token)
    # # print('token2text: ', token2text)
    # #print('content: ', content)
    # #print('summary: ', summary)
    # print('------')
    # # print(a)


if __name__ == '__main__':
    main()
