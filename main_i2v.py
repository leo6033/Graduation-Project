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
import pickle

from TextCNN import TextCNN
from config import Config
class Net(keras.Model):
    def __init__(self):
        super(Net, self).__init__()

        # user Net
        self.user_id_embedding = keras.Sequential([
            layers.Embedding(Config.user_id_num, Config.user_id_embedding_output_dim, input_length=1, name="embedding"),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='use_id_fc')
        ], name='user_id_embedding')
        
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

        # self.zip_code_embedding = keras.Sequential([
        #     layers.Embedding(Config.zip_code_num, Config.zip_code_embedding_output_dim, input_length=1, name="embedding"),
        #     layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='zip_code_fc')
        # ], name='zip_code_embedding')

        
        self.fc1 = layers.Dense(200, kernel_initializer='he_normal')

        # movie net
        self.movie_id_embedding = keras.Sequential([
            # layers.Embedding(Config.move_id_num, Config.move_id_embedding_output_dim, input_length=1, name='embedding'),
            layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='movie_id_fc')
        ], name='movie_id_embedding')

        self.movie_categories_embedding = layers.Embedding(Config.movie_categories_num + 1, Config.movie_categories_output_dim, input_length=Config.generes_max_len, mask_zero=True, name='movie_categories_embedding')
        self.movie_categories_fc = layers.Dense(Config.embedding_dim, kernel_initializer='he_normal', name='movie_categories_fc')
        self.fc2 = layers.Dense(200, kernel_initializer='he_normal')

        self.textCNN = TextCNN(Config.title_max_len, Config.title_word_num, Config.title_enbedding_output_dim, 32, 'relu').get_model()

        # self.fc3 = layers.Dense(128, kernel_initializer='he_normal')
        # self.fc4 = layers.Dense(64, kernel_initializer='he_normal')
        # self.fc5 = layers.Dense(6, kernel_initializer='he_normal')

    def call(self, inputs, training=None):

        user_id = self.user_id_embedding(inputs[0])
        gender = self.gender_embedding(inputs[1])
        age = self.age_embedding(inputs[2])
        occupation = self.occupation_embedding(inputs[3])
        # zip_code = self.zip_code_embedding(inputs[4])

        # movie_id = self.i2v.wv[inputs[4]]
        movie_id = self.movie_id_embedding(inputs[4])
        movie_categories = self.movie_categories_embedding(inputs[5])
        mask = tf.cast(movie_categories._keras_mask, dtype=tf.float32)
        movie_categories = movie_categories * tf.expand_dims(mask, axis=2)
        movie_categories = tf.reduce_sum(movie_categories, axis=1)
        movie_categories = self.movie_categories_fc(movie_categories)

        title = self.textCNN(inputs[6])

        movie = layers.concatenate([movie_id, movie_categories, title])

        user = layers.concatenate([user_id, gender, age, occupation])[0]
        user = tf.nn.leaky_relu(self.fc1(user))
        
        movie = tf.nn.leaky_relu(self.fc2(movie))

        result = tf.reduce_sum(user * movie, axis=1)
        # result = tf.nn.leaky_relu(self.fc3(user * movie))
        # result = tf.nn.leaky_relu(self.fc4(result))
        # result = tf.nn.relu(self.fc5(result))

        return result
    

class DataProcesser(object):
    def __init__(self, dim):
        super(DataProcesser, self).__init__()
        self.i2v = pickle.load(open(f'data/MovieID.{dim}d', 'rb'))
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
        tmp = data['MovieID'].apply(lambda x: list(self.i2v.wv[str(x)]))
        for col in ['UserID', 'Gender', 'Age', 'Occupation']:
            inputs.append(tf.expand_dims(data[col], axis=0))


        inputs.append(tf.convert_to_tensor(np.array(list(tmp.array))))

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

        self.users['Zip-code'] = self.users['Zip-code'].apply(lambda x: str(x)[:3])
        zip_code = list(set(self.users['Zip-code'].to_list()))
        zip_code_map = dict(zip(zip_code, [i + 1 for i in range(len(zip_code))]))
        Config.zip_code_num = len(zip_code) + 1
        print('len of zip_code', len(zip_code))
        self.users['Zip-code'] = self.users['Zip-code'].map(zip_code_map)

        # return gender_map, age_map

class RecomendSystem(object):
    def __init__(self, dim=128):
        self.dataPreProcesser = DataProcesser(dim)
        print("################## start preprocess data ###################")
        self.dataPreProcesser.load_data()
        self.dataPreProcesser.process_data()
        print("################## finish preprocess data ###################")
        self.losses = {'train': [], 'test': []}
        self.net = Net()
        self.net.build(input_shape=[(None, 1), (None, 1), (None, 1), (None, 1), (None, dim), (None, Config.generes_max_len), (None, Config.title_max_len)])
        self.optimizer = keras.optimizers.Adam(Config.LEARNING_RATE)
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()

        self.log_dir = f'./logs/{dim}'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

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
        point = int(len(x) * 0.95)
        print(f"start train. train set size {point}, test set size {len(x) - point}")
        _train_x, test_x, _train_y, test_y = x[:point],  x[point:], y[:point], y[point:]
        for epoch in range(Config.EPOCH):
            train_x, valid_x, train_y, valid_y = train_test_split(_train_x, _train_y, test_size=0.2, random_state=2021)
            batch_num = len(train_x) // Config.BATCH_SIZE
            train_batches = self.get_batches(train_x, train_y, Config.BATCH_SIZE)
            train_start = time.time()
            loss = None
            if True:
                start = time.time()
                for batch in tqdm.tqdm(range(batch_num)):
                    batch_x, batch_y = next(train_batches)
                    inputs = self.dataPreProcesser.gen_input(batch_x)
                    
                    with tf.GradientTape() as tape:
                        predict_y = self.net(inputs)
                        loss = (self.ComputeLoss(batch_y, predict_y) + self.ComputeMetrics(batch_y, predict_y)) / 2

                    grads = tape.gradient(loss, self.net.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
                    # print(loss.numpy())
                    # self.losses['train'].append(loss.numpy())
                    
                    with self.summary_writer.as_default():
                        tf.summary.scalar("train_loss", loss.numpy(), step=self.optimizer.iterations)
                        tf.summary.scalar("train_mae", self.ComputeMetrics.result(), step=self.optimizer.iterations)

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

                self.test((valid_x, valid_y), self.optimizer.iterations, True)
            
            print('############## test loss ###############')
            self.test((test_x, test_y), self.optimizer.iterations)
            

    def test(self, test_data: tuple, step_num: int, log=False):
        test_X, test_y = test_data

        inputs = self.dataPreProcesser.gen_input(test_X)
        predict_y = self.net(inputs)

        test_loss = self.ComputeLoss(test_y, predict_y)
        self.ComputeMetrics(test_y, predict_y)
        # self.losses['test'].append(test_loss.numpy())

        if log:
            with self.summary_writer.as_default():
                tf.summary.scalar("valid_loss", test_loss.numpy(), step=step_num)
                tf.summary.scalar("valid_mae", self.ComputeMetrics.result(), step=step_num)

        print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(test_loss.numpy(), self.ComputeMetrics.result()))



if __name__ == '__main__':
    for dim in [32, 64, 128]:
        recomendSystem = RecomendSystem(dim)
        recomendSystem.train()
    # dataPreProcesser = DataProcesser()
    # dataPreProcesser.load_data()
    # dataPreProcesser.process_data()
    # dataPreProcesser.gen_input()
    # net = Net()
    # net.build(input_shape=[(None, 1), (None, 1), (None, 1), (None, 1), (None, 1), (None, Config.generes_max_len), (None, Config.title_max_len)])
    # net(dataPreProcesser.inputs, training=True)