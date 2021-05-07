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
import time
import os
import tensorflow as tf
import tensorboard
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers, losses
from sklearn.model_selection import StratifiedKFold, train_test_split
import tqdm

from TextCNN import TextCNN
from config import Config
class Net(keras.Model):
    def __init__(self):
        super(Net, self).__init__()

        # user Net
        # self.user_id_embedding = keras.Sequential([
        #     layers.Embedding(Config.user_id_num, Config.user_id_embedding_output_dim, input_length=1, name="embedding"),
        #     layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='use_id_fc')
        # ], name='user_id_embedding')
        
        self.gender_embedding = keras.Sequential([
            layers.Embedding(Config.gender_num, Config.gender_embedding_output_dim, input_length=1, name="embedding"),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='gender_fc')
        ], name='gender_embedding')

        self.age_embedding = keras.Sequential([
            layers.Embedding(Config.age_num, Config.age_embedding_output_dim, input_length=1, name="embedding"),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='age_fc')
        ], name='age_embedding')

        self.occupation_embedding = keras.Sequential([
            layers.Embedding(Config.occupation_num, Config.occupation_embedding_output_dim, input_length=1, name="embedding"),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='occupation_fc')
        ], name='occupation_embedding')

        
        self.fc1 = layers.Dense(200, kernel_initializer='he_normal')

        # movie net
        # self.movie_id_embedding = keras.Sequential([
        #     layers.Embedding(Config.move_id_num, Config.move_id_embedding_output_dim, input_length=1, name='embedding'),
        #     layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='movie_id_fc')
        # ], name='movie_id_embedding')

        self.movie_categories_embedding = layers.Embedding(Config.movie_categories_num + 1, Config.movie_categories_output_dim, input_length=Config.generes_max_len, mask_zero=True, name='movie_categories_embedding')
        self.movie_categories_fc = layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='movie_categories_fc')
        self.fc2 = layers.Dense(200, kernel_initializer='he_normal')

        self.textCNN = TextCNN(Config.title_max_len, Config.title_word_num, Config.title_enbedding_output_dim, 32, 'relu').get_model()

    def call(self, inputs, training=None):
        # user_id = self.user_id_embedding(inputs[0])
        gender = self.gender_embedding(inputs[1])
        age = self.age_embedding(inputs[2])
        occupation = self.occupation_embedding(inputs[3])

        # movie_id = self.movie_id_embedding(inputs[4])

        movie_categories = self.movie_categories_embedding(inputs[5])
        mask = tf.cast(movie_categories._keras_mask, dtype=tf.float32)
        movie_categories = movie_categories * tf.expand_dims(mask, axis=2)
        movie_categories = tf.reduce_sum(movie_categories, axis=1)
        movie_categories = self.movie_categories_fc(movie_categories)

        title = self.textCNN(inputs[6])

        movie = layers.concatenate([movie_categories, title])

        user = layers.concatenate([gender, age, occupation])[0]
        user = self.fc1(user)
        
        movie = self.fc2(movie)

        result = tf.reduce_sum(user * movie, axis=1)

        return result
    

class DataProcesser(object):
    def __init__(self):
        super(DataProcesser, self).__init__()
        self.movies = None
        self.users = None
        self.ratings = None
        self.data = None

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
        self.data = pd.merge(self.ratings, self.movies, how='left', on=['MovieID'])
        self.data = pd.merge(self.data, self.users, how='left', on=['UserID'])

        self.ratings = None
        self.movies = None
        self.users = None
    
    def gen_input(self, data):
        inputs = []
        for col in ['UserID', 'Gender', 'Age', 'Occupation', 'MovieID']:
            inputs.append(tf.expand_dims(data[col], axis=0))
        
        for col in ['Genres', 'Title']:
            inputs.append(tf.keras.preprocessing.sequence.pad_sequences(
                                   data[col], padding="post"
                                )
            )

        return inputs

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
        Config.title_word_num = len(title) + 1
        title_map = dict(zip(title, [i + 1 for i in range(len(title))]))
        
        self.movies['Title'] = self.movies['Title'].apply(lambda x: re.sub(r"[-()\"/@;:<>{}`+=~|.,]", "", x).split())\
                        .apply(lambda x: [title_map[i] for i in x])
        
        # return generes_map, title_map

    def _process_users(self):
        gender_map = {'F': 0, 'M': 1}
        self.users['Gender'] = self.users['Gender'].map(gender_map)

        age = list(set(self.users['Age'].to_list()))
        age_map = dict(zip(age, [i for i in range(len(age))]))
        self.users['Age'] = self.users['Age'].map(age_map)

        # return gender_map, age_map

class RecomendSystem(object):
    def __init__(self):
        self.dataPreProcesser = DataProcesser()
        print("################## start preprocess data ###################")
        self.dataPreProcesser.load_data()
        self.dataPreProcesser.process_data()
        print("################## finish preprocess data ###################")
        self.losses = {'train': [], 'test': []}
        self.net = Net()
        self.net.build(input_shape=[(None, 1), (None, 1), (None, 1), (None, 1), (None, 1), (None, Config.generes_max_len), (None, Config.title_max_len)])
        self.optimizer = keras.optimizers.Adam(Config.LEARNING_RATE)
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()

        self.MODEL_DIR = "./models"
        if not tf.io.gfile.exists(self.MODEL_DIR):
            tf.io.gfile.makedirs(self.MODEL_DIR)

        train_dir = os.path.join(self.MODEL_DIR, 'summaries', 'train')
        test_dir = os.path.join(self.MODEL_DIR, 'summaries', 'eval')

        checkpoint_dir = os.path.join(self.MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.net, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def gen_data(self):
        features = ['UserID', 'Gender', 'Age', 'Occupation', 'MovieID', 'Genres', 'Title']
        return self.dataPreProcesser.data[features], self.dataPreProcesser.data['Rating']
    
    def get_batches(self, Xs, ys, batch_size):
        for start in range(0, len(Xs), batch_size):
            end = min(start + batch_size, len(Xs))
            yield Xs[start:end], ys[start:end]

    def train(self):
        # mse = tf.keras.losses.MeanSquaredError()
        x, y = self.gen_data()
        print("start train")
        for epoch in range(Config.EPOCH):
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=2021)
            batch_num = len(train_x) // Config.BATCH_SIZE
            train_batches = self.get_batches(train_x, train_y, Config.BATCH_SIZE)
            train_start = time.time()
            loss = None
            if True:
                start = time.time()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                for batch in tqdm.tqdm(range(batch_num)):
                    batch_x, batch_y = next(train_batches)
                    inputs = self.dataPreProcesser.gen_input(batch_x)
                    
                    with tf.GradientTape() as tape:
                        predict_y = self.net(inputs)
                        loss = self.ComputeLoss(batch_y, predict_y)
                        self.ComputeMetrics(batch_y, predict_y)

                    grads = tape.gradient(loss, self.net.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
                    # print(loss.numpy())
                    self.losses['train'].append(loss.numpy())

                    if self.optimizer.iterations % Config.log_freq == 0:
                        rate = Config.log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                                self.optimizer.iterations.numpy(),
                                epoch,
                                batch,
                                batch_num,
                                loss.numpy(), (self.ComputeMetrics.result()), rate))
                        self.ComputeMetrics.reset_states()
                        start = time.time()
                



if __name__ == '__main__':
    recomendSystem = RecomendSystem()
    recomendSystem.train()
    # dataPreProcesser = DataProcesser()
    # dataPreProcesser.load_data()
    # dataPreProcesser.process_data()
    # dataPreProcesser.gen_input()
    # net = Net()
    # net.build(input_shape=[(None, 1), (None, 1), (None, 1), (None, 1), (None, 1), (None, Config.generes_max_len), (None, Config.title_max_len)])
    # net(dataPreProcesser.inputs, training=True)