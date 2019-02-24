# -*- coding:utf-8 -*-
# default word token
PAD_token = 0
SOS_token = 1
EOS_token = 2

# define max length of sentence
MAX_LENGTH = 10

# define word trim threshold
MIN_COUNT = 3

# define  teacher forcing ratio
teacher_forcing_ratio = 1.0

# path to store checkpoint file
save_dir = "./data/save"

# train data path
corpus_name = "cornell movie-dialogs corpus"
corpus = "data/" + corpus_name
datafile = corpus + "/formatted_movie_lines.txt"