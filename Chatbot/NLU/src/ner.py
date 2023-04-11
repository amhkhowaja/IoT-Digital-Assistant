#!/usr/bin/env python

"""
Named Entity recogonition NER for NLU

"""
import json
import random
import sys
import subprocess
from spacy.training import Example
from spacy.language import Language
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
import data_generation
from text_preprocessing_nltk import NLTK_Preprocessing


class NER:

    '''Named Entity Recogonition for entity extraction in this domain'''

    def __init__(self, dataframe):
        '''Declaration of important variable for a class'''
        self.dataframe = data_generation.preprocess_dataframe(dataframe)
        self.model = spacy.load('en_core_web_lg')
        with open('../data/annotated.json') as json_file:
            self.annotated_data = json.loads(json_file.read())
        self.train, self.test = train_test_split(
            self.annotated_data, test_size=0.25, random_state=42)
        self.tokens = None
        self.prc = NLTK_Preprocessing(self.dataframe)
        self.w2v_model = self.prc.train_word2vec_model()
        self.wvs = self.prc.get_w2v_vectors()
        self.ner_model = None
        self.big_model = None
        self.big_model_name = "iota_model"

    def get_training_data(self):
        '''Returns the train data'''
        return self.train

    def get_test_data(self):
        '''Returns the test data'''
        return self.test

    def evaluate(self):
        '''Return list of examples of test'''
        examples = []
        for question, ant in self.get_test_data():
            doc = self.big_model.make_doc(question)
            example = Example.from_dict(doc, ant)
            examples.append(example)
            print(examples)
        return self.big_model.evaluate(examples)

    def train_ner(self, epochs, batch_size):
        '''Training the ner model , with the annotated data'''
        nlp = spacy.blank("en")
        data = self.train
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe("ner", last=True)
        for _, annot in data:
            for ent in annot['entities']:
                ner.add_label(ent[2])
        not_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*not_pipes):
            optimizer = nlp.begin_training()
            for _ in range(epochs):
                random.shuffle(data)
                losses = {}
                for batch in spacy.util.minibatch(data, batch_size):
                    for question, label in batch:
                        doc = nlp.make_doc(question)
                        example = Example.from_dict(doc, label)
                        nlp.update(
                            ([example]),
                            drop=0.25,
                            sgd=optimizer,
                            losses=losses
                        )
                print(losses)
        self.ner_model = nlp
        nlp.to_disk("../models/blank_ner")
        self.inject_wv_into_model("../models/blank_ner", "../models/w2v")
        print("Injected word vectors into the ner_model")
        self.ner_model = spacy.load("../models/blank_ner")
        return self.ner_model

    def load_ner(self, *args):
        '''Load the trained ner model, where args[0] is the filename'''
        if len(args) == 0:
            return self.ner_model
        elif len(args) == 1:
            return spacy.load(args[0])
        else:
            raise Exception("Too many arguments")

    def inject_wv_into_model(self, model_path, word_vectors_path):
        '''Run the subprocess with cmd commands to inject the word vector file in the model file'''
        subprocess.run(
            [
                sys.executable,
                "-m",
                "spacy",
                "init-model",
                "en",
                model_path,
                "--vector-loc",
                word_vectors_path
            ]
        )

    def generate_big_model(self, save=True):
        '''Generate the en_core_web_lg model with chatbot trained entities'''
        nlp_lg = spacy.load("en_core_web_lg")
        nlp_lg.meta["name"] = self.big_model_name
        nlp_ner = self.load_ner('../models/blank_ner')
        new_vocab = ['attribute', 'connection', 'identifier', 'inventory',
                     'location', 'option', 'page', 'stat', 'time', 'widget', 'toggle']
        for label in new_vocab:
            nlp_lg.vocab.strings.add(label)
        ner = nlp_ner.get_pipe('ner')

        @Language.component("chatbot_ner")
        def ner_component(doc):
            '''process the ner pipe'''
            return ner(doc)
        nlp_lg.add_pipe('ner', name="chatbot_ner", before="ner")
        chatbot_ner = nlp_lg.get_pipe("chatbot_ner")
        chatbot_ner.initialize("ner")
        self.big_model = nlp_lg

        if save:
            self.big_model.to_disk('../models/'+nlp_lg.meta["name"])
        return self.big_model

    def analyze_sentences(self, sentences):
        '''Analyze the sentences and return their entities along with its labels'''
        analyze = []
        for sentence in sentences:
            doc = self.big_model(sentence)
            entities = {}
            for ent in doc.ents:
                entities[ent.text] = ent.label_
            analyze.append({'text': sentence, 'entities': entities})
            # print(entities)
        return analyze


if __name__ == '__main__':
    dataframe = pd.read_csv('../data/final_data.csv')
    ner = NER(dataframe)
    #nlp = ner.load_ner('../models/main_ner_model')
    nlp = ner.train_ner(10, 5)
    nlp = ner.generate_big_model()
    sentences = [
        "I want to filter the billing state of imsi and sims",
        "Why I am getting the errors in the traffic control",
        "sim cards are being hacked.",
        "I want to visit Poland",
        "Mr. Harry Potter is the magician"
    ]
    print(ner.evaluate())
    print(ner.analyze_sentences(sentences))
