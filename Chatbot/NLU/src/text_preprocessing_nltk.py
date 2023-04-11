#!/usr/bin/env python

'''

Text preprocessing to prepare it for machine learning model

'''
# Text Preprocessing using nltk

from string import punctuation
from itertools import chain
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import data_generation
# text preprocessing


class NLTK_Preprocessing:
    '''Preprocessing Library for the text implemented using mainly from nltk ,
    encoded using LabelEncoder,
    and padded using tensorflow pad_sequence class'''

    def __init__(self, *args):
        '''args [0] can be dataset filename in the first case or can also be dataframe object in the second case
            args[1] is the boolean value which represent if we want to preprocess dataframe or not.
        '''
        if len(args) == 1:
            if isinstance(args[0], str):
                self.data_file = args[0]
                self.dataframe = data_generation.import_dataframe(
                    self.data_file)
            elif isinstance(args[0], object):
                self.dataframe = args[0]
            else:
                raise Exception("INVALID_ARG_ERROR")
        elif len(args) == 2:
            if isinstance(args[0], object) and isinstance(args[1], bool):
                if args[1]:
                    self.dataframe = data_generation.preprocess_dataframe(
                        args[0])
                else:
                    self.dataframe = args[0]
            else:
                raise Exception("INVALID_ARG_ERROR")
        else:
            raise Exception("INVALID_ARG_ERROR")
        self.feature = self.dataframe['Questions']
        self.label = self.dataframe['Intent']
        self.train, self.test, self.val = self.split_data(
            self.feature, self.label)
        self.feature_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.tokens = None
        self.encoded_tokens = None
        self.padded_encoded_tokens = None
        self.train_labels, self.test_labels, self.val_labels = self.preprocess_labels()
        self.label_classes = None
        self.w2v = None

        # self.w2v=None
    def tokenize_text_sequence(self, column, uncase=True):
        '''
        tokenize the questions into text sequences using nltk
        '''
        if uncase:
            column = [x.lower() for x in list(column)]
        self.tokens = [word_tokenize(x) for x in list(column)]
        return self.tokens

    def remove_punctuation(self):
        '''
        remove the punctuation from the text sequences
        '''
        sentences = []
        for token in self.tokens:
            x = [word for word in token if word not in punctuation]
            sentences.append(x)
        self.tokens = sentences
        return self.tokens

    def lemmatize_sequence(self):
        '''
        lemmatize the text sequence using nltk wordnetlemmatizer library
        '''
        lm = WordNetLemmatizer()
        lem = []
        for token in self.tokens:
            x = [lm.lemmatize(word) for word in token]
            lem.append(x)
        self.tokens = lem
        return self.tokens

    def stem_sequence(self):
        '''stem the text sequences using porterstemmer'''
        stemmer = PorterStemmer()
        stemmed = []
        for token in self.tokens:
            x = [stemmer.stem(word) for word in token]
            stemmed.append(x)
        self.tokens = stemmed
        return self.tokens

    def remove_stopwords(self):
        '''
        remove the stopwords from the text sequences of english language
        '''
        stop_words = stopwords.words('english')
        sentences = []
        for token in self.tokens:
            x = [word for word in token if word not in stop_words]
            sentences.append(x)
        self.tokens = sentences
        return sentences

    def split_data(self, X, Y, test_size=0.2, val_size=0.1, random_state=42):
        '''split the data into training set, test set and validation set using the
        given parameter for the size'''
        # calculating the test size and val_size
        val_len = val_size*len(X)
        total_test_len = (test_size+val_size)*len(X)
        val_prp = val_len/total_test_len

        # training testing split
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=(test_size+val_size), random_state=random_state
        )
        x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size=val_prp)
        self.train = pd.DataFrame({X.name: x_train, Y.name: y_train})
        self.test = pd.DataFrame({X.name: x_test, Y.name: y_test})
        self.val = pd.DataFrame({X.name: x_val, Y.name: y_val})
        self.train.to_csv(
            '../data/intent_classification/train_set.csv', index=False)
        self.test.to_csv(
            '../data/intent_classification/test_set.csv', index=False)
        self.val.to_csv(
            '../data/intent_classification/val_set.csv', index=False)
        return self.train, self.test, self.val

    def get_train_set(self):
        '''Get the dataframe of the splited data '''
        return self.train

    def get_full_set(self):
        '''return dataframe'''
        return self.dataframe

    def get_test_set(self):
        '''Get the dataframe of the splited data '''
        return self.test

    def get_val_set(self):
        '''Get the dataframe of the splited data '''
        return self.val

    def get_tokens(self):
        '''Get the dataframe of the splited data '''
        return self.tokens

    def encode_train(self, X, Y):
        '''
            X is the data to be fitted and Y is the data we want to transform the fitting.
            It return the Label encoder object and the transformed list

            X and Y both are tokens in the form of ['hello', 'there', 'mates']
        '''
        le = self.feature_encoder
        flatX = list(chain.from_iterable(X))
        le.fit(flatX)
        #func=lambda s : '<oov>' if s not in le.classes_ else s
        Y_dash = []
        for sentence in Y:
            s = ['<oov>' if word not in le.classes_ else word for word in sentence]
            Y_dash.append(s)
        le.classes_ = np.append(le.classes_, '<oov>')
        Y = [list(le.transform(sen)) for sen in Y_dash]
        self.encoded_tokens = Y
        self.feature_encoder = le
        return self.encoded_tokens

    def word_index(self):
        '''return the word index of the feature/Questions'''
        return self.feature_encoder.classes_

    def get_encoded_train(self):
        ''' Return the fit train and transform train into it '''
        return self.encoded_tokens

    def pad_encoded(self, encoded_sequence, max_len):
        '''pad the encoded sequence with default of post truncating and post padding '''
        encoder = self.feature_encoder
        self.padded_encoded_tokens = pad_sequences(encoded_sequence,
                                                   padding='post',
                                                   maxlen=max_len,
                                                   truncating='post',
                                                   value=np.where(encoder.classes_ == '<oov>'))
        return self.padded_encoded_tokens

    def get_label_classes(self):
        '''get all the classes of the label'''
        return self.label_classes

    def train_word2vec_model(self, save=True):
        '''I will train the word 2 vec model for ML purpose'''
        prc = NLTK_Preprocessing(self.dataframe)
        prc.preprocess_sequence(self.get_train_set(), encode=False)
        tokens = prc.tokens
        w2v = Word2Vec(window=3, min_count=2, workers=4)
        w2v.build_vocab(corpus_iterable=tokens)
        w2v.train(tokens, total_examples=w2v.corpus_count, epochs=w2v.epochs)
        self.w2v = w2v
        if save:
            self.w2v.save('../models/w2v.model')
        return self.w2v

    def get_w2v_model(self):
        '''get w2v model '''
        return self.w2v

    def get_w2v_vectors(self):
        '''get all the pregenerated w2v vectors'''
        return self.w2v.wv

    def preprocess_labels(self):
        '''preprocess and encode the labels with sklearn LabelEncoder'''
        le = self.label_encoder
        le.fit(self.label)
        self.label_classes = self.label_encoder.classes_
        train_labels = np.asarray(le.transform(
            self.get_train_set()['Intent'])).astype('int32')
        test_labels = np.asarray(le.transform(
            self.get_test_set()['Intent'])).astype('int32')
        val_labels = np.asarray(le.transform(
            self.get_val_set()['Intent'])).astype('int32')
        return train_labels, test_labels, val_labels

    def get_encoded_train_labels(self):
        '''get encoded train labels'''
        return self.train_labels

    def get_encoded_test_labels(self):
        '''get encoded test labels'''
        return self.test_labels

    def get_encoded_val_labels(self):
        '''get encoded validation labels'''
        return self.val_labels

    def get_most_common(self, number=None):
        '''get most common number of words from the tokens '''
        flattokens = list(chain.from_iterable(self.tokens))
        freqdist = nltk.FreqDist(flattokens)
        most_common = freqdist.most_common(number)
        return most_common

    def preprocess_sequence(self,
                            encode_into,
                            max_vocab=185,
                            remove_punct=True,
                            lemmatize=True,
                            rem_stopwords=False,
                            stem=False,
                            pad=False,
                            encode=True,
                            max_pad_len=8):
        '''Pipeline for the preprocessing of the feature'''
        self.tokens = self.tokenize_text_sequence(self.train['Questions'])
        y_encode = NLTK_Preprocessing(self.dataframe)
        if remove_punct:
            self.tokens = self.remove_punctuation()
        if rem_stopwords:
            self.tokens = self.remove_stopwords()
        if lemmatize:
            self.tokens = self.lemmatize_sequence()
        if stem:
            self.tokens = self.stem_sequence()
        tokens = self.tokens
        # take out the most common words
        most_common = self.get_most_common(max_vocab-1)
        list_common = [x[0] for x in most_common]
        tokens = [[x for x in sent if x in list_common] for sent in tokens]
        self.tokens = tokens
        if encode:
            # doing the same for encoded into what we applied for the feature
            self.encode_train(self.tokens, self.tokens)
            y_tokens = y_encode.tokenize_text_sequence(encode_into)
            if remove_punct:
                y_tokens = y_encode.remove_punctuation()
            if rem_stopwords:
                y_tokens = y_encode.remove_stopwords()
            if lemmatize:
                y_tokens = y_encode.lemmatize_sequence()
            if stem:
                y_tokens = y_encode.stem_sequence()
            # take out the most common words
            tokens = y_encode.encode_train(self.tokens, y_tokens)
        if pad and encode:
            tokens = y_encode.pad_encoded(tokens, max_pad_len)
            # converting everything into numpy array
            tokens = np.asarray([np.array(x).astype('int32') for x in tokens])
        return tokens

    def generate_class_histgram(self):
        '''generate the label class frequency histogram'''
        label_count = self.label.value_counts()
        sns.barplot(label_count.index, label_count)
        plt.xticks(rotation=20)
        plt.gca().set_ylabel("no of samples")
        plt.gca().set_xlabel('classes')

# def encode_labels():

#def encode_labels():
if __name__ == '__main__':
    # nltk.download('all')
    #split_data(feature['Questions'], feature['Intent'])
    process = NLTK_Preprocessing("../data/final_data.csv")
    train = process.get_train_set()['Questions']
    test = process.get_test_set()['Questions']
    # print(train)
    preprocessed = process.preprocess_sequence(test, rem_stopwords=True)
    most_common = process.get_most_common()[:-100:-1]
    print(most_common)
    something = process.word_index()
    process.generate_class_histgram()
    print(something)
    print()
    print(preprocessed)
    print()
    print(str(len(preprocessed)) + " = "+str(len(test)))
