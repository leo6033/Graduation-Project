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
import tensorflow as tf
import tensorboard
from tensorflow import keras
from tensorflow.keras import Sequential, layers, optimizers, losses

from config import Config
class Net(keras.Model):
    def __init__(self, user_id_num: int, gender_num: int, age_num: int, occupation_num: int):
		super(Net, self).__init__()

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



    def call(self, inputs, training=None):
        user_id = self.user_id_embedding(inputs[0])
        gender = self.gender_embedding(inputs[1])
        age = self.age_embedding(inputs[2])
        occupation = self.occupation_embedding(inputs[3])

        x = keras.layers.concatenate([user_id, gender, age, occupation])[0]
        x = self.fc1(x)

        return x
