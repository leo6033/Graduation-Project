"""
@Description: 网络相关配置文件
@Author: 吕明伟
@Date: 2021-4-6
"""

class Config:
    user_id_embedding_output_dim = 32
    zip_code_embedding_output_dim = 16
    gender_embedding_output_dim = 16
    age_embedding_output_dim = 16
    occupation_embedding_output_dim = 16
    move_id_embedding_output_dim = 32
    movie_categories_output_dim = 32
    title_enbedding_output_dim = 64

    embedding_dim = 32

    movie_categories_num = 18
    move_id_num = 3953
    user_id_num = 6041
    gender_num = 2
    age_num = 7
    occupation_num = 21
    zip_code_num = None
    title_word_num = None

    title_max_len = 16
    generes_max_len = 6

    EPOCH = 5
    BATCH_SIZE = 256
    LEARNING_RATE = 0.01
    log_freq = 1000