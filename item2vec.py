"""
@Description: item2vec
@Author: 吕明伟
@Date: 2021-5-13
"""
from gensim.models import Word2Vec
import pandas as pd
import os
import pickle

def w2v(df, f, L=128):
    print("w2v",f)
    sentences=[]
    for line in df[f].values:
        sentences.append(line.split())
    print("Sentence Num {}".format(len(sentences)))
    w2v = Word2Vec(sentences, vector_size=L, window=20, min_count=1, sg=1, workers=8, epochs=10)
    print("save w2v to {}".format(os.path.join('data', f +".{}d".format(L))))
    pickle.dump(w2v,open(os.path.join('data', f + ".{}d".format(L)), 'wb'))  

def get_list(x):
    return " ".join(list(x['MovieID']))

if __name__ == '__main__':

    path = 'data/ml-1m/'
    ratings_col = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings = pd.read_csv(f'{path}ratings.dat', sep="::", names=ratings_col)
    ratings['MovieID'] = ratings['MovieID'].astype(str)
    tmp = ratings[['UserID', 'MovieID']].groupby('UserID').apply(get_list)
    tmp = tmp.reset_index()
    tmp.columns = ['UserID', 'MovieID']
    w2v(tmp, 'MovieID', L=64)
    