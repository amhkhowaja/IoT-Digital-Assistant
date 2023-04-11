#!/usr/bin/env python

'''
Intent Classification and their analysis
'''

import numpy as np
import pandas as pd
from tensorflow import keras
from keras import models, layers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import data_generation
import text_preprocessing_nltk
#from gensim.models import Word2Vec


class IntentClassifier:
    '''Intent CLassifier classifies the intent of the query asked by the user'''

    def __init__(self, dataframe):
        '''Setting up all the important variables and parameters for model training and analysis'''

        self.dataframe = data_generation.preprocess_dataframe(dataframe)
        self.preprocessor = text_preprocessing_nltk.NLTK_Preprocessing(
            dataframe)
        self.x_train_padded = self.preprocessor.preprocess_sequence(
            self.preprocessor.get_train_set()['Questions'], rem_stopwords=True, pad=True)
        self.x_test_padded = self.preprocessor.preprocess_sequence(
            self.preprocessor.get_test_set()['Questions'], rem_stopwords=True, pad=True)
        self.x_val_padded = self.preprocessor.preprocess_sequence(
            self.preprocessor.get_val_set()['Questions'], rem_stopwords=True, pad=True)
        self.y_train_encoded, self.y_test_encoded, self.y_val_encoded = self.preprocessor.preprocess_labels()
        vocab_size, embedding_dim, max_length = (185, 8, 8)
        self.cnn = models.Sequential(
            [
                layers.Embedding(vocab_size, embedding_dim,
                                 input_length=max_length),
                layers.Dropout(0.3),
                layers.Conv1D(16, 8, activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dense(5, activation="softmax")
            ]
        )
        self.bilstm = models.Sequential(
            [
                layers.Embedding(vocab_size, embedding_dim,
                                 input_length=max_length),
                layers.Bidirectional(layers.LSTM(embedding_dim)),
                layers.Dense(5, activation="softmax"),
            ]
        )
        self.models = {'CNN': self.cnn, 'BILSTM': self.bilstm}
        self.model_title = 'CNN'
        self.model = self.cnn

        self.training_history = None
        self.y_pred_test = None
        self.label_classes = self.preprocessor.label_classes

    def model_compile_train(self,
                            validation=False,
                            comp_loss='sparse_categorical_crossentropy',
                            comp_metrics="accuracy",
                            comp_optimizer='adam',
                            callbacks=callbacks.EarlyStopping(patience=5),
                            epochs=200,
                            validation_split=0.0,
                            batch_size=16,
                            shuffle=True,
                            save=False
                            ):
        '''
        Used for merging the model.compile and model.fit for cleaner and easy invokation.
        Returns : model and history of training.
        '''
        val = [None, (self.x_val_padded, self.y_val_encoded)]
        self.model.compile(
            loss=comp_loss, metrics=comp_metrics, optimizer=comp_optimizer)
        print(self.model.summary())
        self.training_history = self.model.fit(
            self.x_train_padded,
            self.y_train_encoded,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val[int(validation)],
            validation_split=validation_split,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        if save:
            models.save_model(self.model, "../models/"+self.model_title+".h5")
        return self.model

    def get_model(self):
        '''Return the model object'''
        return self.model

    def get_training_history(self):
        '''Returns the training history'''
        return self.training_history

    def plot_train_val_analysis(self, save=True):
        '''Will plot the training and validation accuracy and loss graph/curves'''
        epoch_range = range(len(self.training_history.history["loss"]))
        # validation loss and training loss
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.title(self.model_title + " Loss graph")
        plt.plot(
            epoch_range, self.training_history.history["loss"], label="Training Loss")
        plt.plot(
            epoch_range, self.training_history.history["val_loss"], label="Validation Loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title(self.model_title + " Accuracy graph")
        plt.plot(
            epoch_range, self.training_history.history["accuracy"], label="Training Accuracy")
        plt.plot(
            epoch_range, self.training_history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        if save:
            plt.savefig("../plots/" + self.model_title + "_loss_acc_graph")

    def predict_test_max(self):
        '''Predict the max argument of the all classes probability distribution'''
        y_pred = self.model.predict(self.x_test_padded)
        self.y_pred_test = [np.argmax(x) for x in y_pred]
        return self.y_pred_test

    def generate_conf_matrix(self, save=True):
        '''Generate the confusion matrix of the classes'''
        disp = confusion_matrix(self.y_test_encoded, self.y_pred_test)
        plt.figure(figsize=(8, 8))
        sns.heatmap(disp, annot=True, cmap="Blues").set(
            title="Confusion Matrix of " + self.model_title + "model"
        )
        if save:
            plt.savefig("../plots/" + self.model_title + "_confusion_matrix")

    def analyze_sentence(self, sentences):
        '''Analyze the sentences and give the predicted classes'''
        process = text_preprocessing_nltk.NLTK_Preprocessing(self.dataframe)
        padded = process.preprocess_sequence(sentences, pad=True)
        #sequence = tokenizer.texts_to_sequences(sentences)
        #padded = pad_sequences(sequence, maxlen=8, padding="post", truncating="post")
        y_pred = self.model.predict(padded)
        y_pred = [np.argmax(x) for x in y_pred]
        process.preprocess_labels()
        classes = process.get_label_classes()
        y_pred = [classes[int(x)] for x in y_pred]
        return y_pred

    def get_train_data(self):
        '''returns x_train, y_train(all paddded)'''
        return self.x_train_padded, self.y_train_encoded

    def get_test_data(self):
        '''returns x_test , y_test(all paddded)'''
        return self.x_test_padded, self.y_test_encoded

    def get_val_data(self):
        '''returns x_test , y_test(all paddded)'''
        return self.x_val_padded, self.y_val_encoded

    def evaluate_model(self):
        '''print the classification report of the model'''
        print(classification_report(
            self.get_test_data()[1], self.predict_test_max()))
        print("test_loss, test_accuracy :", self.model.evaluate(
            self.get_test_data()[0], self.get_test_data()[1]))


if __name__ == '__main__':
    df = pd.read_csv('../data/final_data.csv')
    print(type(df))
    classifier = IntentClassifier(df)
    classifier.model_compile_train(validation=True)
    sentences = [
        "how can i add filter?",
        "is my billing state active?",
        "can i customize my view?",
        "list all of my billing states",
        "Why network connectivity giving me error",
        "what is my name ?"
    ]
    classifier.plot_train_val_analysis()
    result = classifier.analyze_sentence(sentences)
    print(result)
    classifier.evaluate_model()
    process = text_preprocessing_nltk.NLTK_Preprocessing(df)
    #print(process.preprocess_sequence(sentences, encode=False))
