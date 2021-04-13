"""
@Description: 
@Author: 吕明伟
@Date: 2021-4-6
"""

import random
import math
import numpy as np
import pandas as pd
import json
import re
import tensorflow as tf
import tensorboard
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers, losses

from TextCNN import TextCNN
from config import Config
class Net(keras.Model):
    def __init__(self):
		super(Net, self).__init__()

        # user Net
        self.user_id_embedding = keras.Sequential([
            layers.Embedding(user_id_num, Config.user_id_embedding_output_dim, input_length=1, name="embedding"),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='use_id_fc')
        ], name='user_id_embedding')
        
        self.gender_embedding = keras.Sequential([
            layers.Embedding(gender_num, Config.gender_embedding_output_dim, input_length=1, name="embedding"),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='gender_fc')
        ], name='gender_embedding')

        self.age_embedding = keras.Sequential([
            layers.Embedding(age_num, Config.age_embedding_output_dim, input_length=1, name="embedding"),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='age_fc')
        ], name='age_embedding')

        self.occupation_embedding = keras.Sequential([
            layers.Embedding(occupation_num, Config.occupation_embedding_output_dim, input_length=1, name="embedding"),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='occupation_fc')
        ], name='occupation_embedding')

        
        self.fc1 = layers.Dense(200, kernel_initializer='he_normal')

        # movie net
        self.movie_id_embedding = keras.Sequential([
            layers.Embedding(Config.move_id_num, Config.move_id_embedding_output_dim, input_length=1, name='embedding'),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='movie_id_fc')
        ], name='movie_id_embedding')

        self.movie_categories_embedding = layers.Embedding(Config.movie_categories_num + 1, Config.movie_categories_output_num, mask_zero=True, name='movie_categories_embedding'),
        self.movie_categories_fc = layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='movie_categories_fc')
        self.fc2 = layers.Dense(200, kernel_initializer='he_normal')

        self.textCNN = TextCNN()



    def call(self, inputs, training=None):
        user_id = self.user_id_embedding(inputs[0])
        gender = self.gender_embedding(inputs[1])
        age = self.age_embedding(inputs[2])
        occupation = self.occupation_embedding(inputs[3])

        movie_id = self.movie_id_embedding(inputs[4])

        movie_categories = self.movie_categories_embedding(inputs[5])
        mask = tf.cast(movie_categories._keras_mask, dtype=tf.float32)
        movie_categories = movie_categories * tf.expand_dims(mask, axis=2)
        movie_categories = tf.reduce_sum(movie_categories, axis=1, keepdims=True)
        movie_categories = self.movie_categoris_fc(movie_categories)

        movie = layers.concatenate([movie_id, movie_categories])

        x = layers.concatenate([user_id, gender, age, occupation])[0]
        x = self.fc1(x)

        return x

class DataProcesser(object):
    def __init__(self):
        super(DataProcesser, self).__init__()
        self.movies = None
        self.users = None
        self.ratings = None
        self.inputs = []

        self.load_data()
        self.process_data()

    def load_data(self):
        path = 'data/ml-1m/'
        movies_col = ['MovieID', 'Title', 'Genres']
        users_col = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        ratings_col = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        self.movies = pd.read_csv(f'{path}movies.dat', sep="::", names=movies_col)
        self.users = pd.read_csv(f'{path}users.dat', sep="::", names=users_col)
        self.ratings = pd.read_csv(f'{path}ratings.dat', sep="::", names=ratings_col)
    
    def process_data(self):
        """
            数据格式处理
        """
        self._process_movies()
        self._process_users()
    
    def gen_input(self):
        

    def _process_movies(self):
        # 构造字典 {电影类型：标号}
        genres = self.movies['Genres'].apply(lambda x: x.split('|')).to_list()
        genres = list(set(list(np.hstack(genres))))
        genres_map = dict(zip(genres, [i + 1 for i in range(len(genres))]))
        
        self.movies['Genres'] = self.movies['Genres']\
                                .apply(lambda x: x.split('|'))\
                                .apply(lambda x: [genres_map[i] for i in x])
        
        title = self.movies['Title'].apply(lambda x: re.sub(r"[-()\"/@;:<>{}`+=~|.,]", "", x).split()).to_list()
        title = list(set(list(np.hstack(title))))
        title_map = dict(zip(title, [i + 1 for i in range(len(title))]))
        
        self.movies['Title'] = self.movies['Title'].apply(lambda x: re.sub(r"[-()\"/@;:<>{}`+=~|.,]", "", x).split())\
                        .apply(lambda x: [title_map[i] for i in x])
        
        return generes_map, title_map

    def _process_users(self):
        gender_map = {'F': 0, 'M': 1}
        self.users['Gender'] = self.users['Gender'].map(gender_map)

        age = list(set(users['Age'].to_list()))
        age_map = dict(zip(age, [i for i in range(len(age))]))
        self.users['age'] = self.users['age'].map(age_map)

        return gender_map, age_map