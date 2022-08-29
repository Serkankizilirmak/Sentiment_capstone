# ===============================================================================================#

# Imports

# ===============================================================================================#

import streamlit as sl
import tensorflow
import nltk
import pandas as pd
import numpy as np
import altair as alt
import pickle

from nltk.tokenize import RegexpTokenizer
from keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from keras.preprocessing import sequence

# ===============================================================================================#

# Functions and Models Prepared

# ===============================================================================================#

word_index_dict = pickle.load(open(r'Data/Neural_Networks/Models/word_index_dict.pkl', 'rb'))

neural_net_model = load_model(r'Data/Neural_Networks/Models/Neural_Network.h5py')

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')


def index_review_words(text):
    review_word_list = []
    for word in text.lower().split():
        if word in word_index_dict.keys():
            review_word_list.append(word_index_dict[word])
        else:
            review_word_list.append(word_index_dict['<UNK>'])

    return review_word_list


def add_sum_suffix(text):
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        word = word + '_sum'
        new_text += word + ' '

    return new_text


def text_cleanup(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        new_text += word + ' '

    return new_text
##
def give_largest(col, n):

    largest = col.nlargest(n).reset_index(drop=True)

    data = [x for x in largest]

    index = [f'{i}_largest' for i in range(1, len(largest) + 1)]

    return pd.Series(data, columns=index)

def n_largest(df, axis, n):

    return df.apply(give_largest, axis=axis, n=n)
##


# ===============================================================================================#

# Streamlit

# ===============================================================================================#

sl.title("Otel Yorumları Sınıflandırıcı")

review_text = sl.text_area('Lütfen Yorumunuzu Giriniz (EN)')


if sl.button('Tahminle'):
    col1, col2 = sl.columns(2)

    result_review = review_text.title()

    review_text = text_cleanup(review_text)

    review_text = index_review_words(review_text)

    all_review_text = review_text

    all_review_text = pad_sequences([all_review_text], value=word_index_dict['<PAD>'], padding='post',
                                             maxlen=200)

    prediction = neural_net_model.predict(all_review_text)

    proba_df = pd.DataFrame(prediction, columns=['1', '2', '3', '4', '5'])

    prediction_num = np.argmax(prediction)

    with col1:

        sl.success("Prediction")
        a = n_largest(proba_df,axis=1, n=2)
        sl.success(a)
        sl.success(prediction_num + 1)
        sl.write("Text Padding")
        sl.write(all_review_text)

    with col2:

        sl.success("Prediction Probability")
        sl.write(proba_df.iloc[:,0])
        sl.write(proba_df.iloc[:,1])
        sl.write(proba_df.iloc[:, 2])
        sl.write(proba_df.iloc[:, 3])
        sl.write(proba_df.iloc[:, 4])



