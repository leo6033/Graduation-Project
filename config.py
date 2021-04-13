"""
@Description: 网络相关配置文件
@Author: 吕明伟
@Date: 2021-4-6
"""

class Config:
    user_id_embedding_output_dim = 32
    gender_embedding_output_dim = 16
    age_embedding_output_dim = 16
    occupation_embedding_output_dim = 16
    move_id_embedding_output_dim = 32
    movie_categories_output_num = 32

    embedding_dim = 32

    movie_categories_num = 18
    move_id_num = 3953

    title_max_len = 16
    generes_max_len = 6