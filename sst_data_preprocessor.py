import re
import json
import os
import pytreebank
import numpy as np
import time
import pickle
import sys
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from official import nlp
from official.nlp import bert
import official.nlp.bert.tokenization


class SSTDataPreprocessor:
    """ Data preprocessing class """

    def __init__(self, config):
        self.config = config
        self.num_classes = config['data']['num_classes']
        self.sequence_length = config['data']['seq_length']

        self.load_data()
        self.get_tokenizer()

        self.train[0], self.train[1] = self.random_sample(self.train[0], self.train[1])
        self.dev[0], self.dev[1] = self.random_sample(self.dev[0], self.dev[1])
        self.test[0], self.test[1] = self.random_sample(self.test[0], self.test[1])

        self.train[0] = self.bert_encode(self.train[0])
        self.dev[0] = self.bert_encode(self.dev[0])
        self.test[0] = self.bert_encode(self.test[0])
        self.train[1] = self.train[1]
        self.dev[1] = self.dev[1]
        self.test[1] = self.test[1]
        
        data = {}
        data['train'] = self.train
        data['dev'] = self.dev
        data['test'] = self.test
        self.save_data(data)      


    def get_tokenizer(self):
        self.tokenizer = bert.tokenization.FullTokenizer(
            vocab_file='./vocab.txt',
            do_lower_case=True
        )


    def load_data(self):
        self.load_dataset()
        self.all_sentences = list(self.train[0]) + list(self.dev[0]) + list(self.test[0])
        self.vocab = self.get_vocab(self.all_sentences)


    def load_dataset(self):
        dataset = pytreebank.load_sst()

        if self.num_classes == 5:
            self.get_train_dev_test_fine(dataset)
        else:
            self.get_train_dev_test_course(dataset)


    def get_vocab(self, all_sentences):
        vocab = set()
        for sentence in all_sentences:
            for word in sentence.split(' '):
                vocab.add(word)
        return vocab


    def get_train_dev_test_fine(self, dataset):
        self.train = self.get_fine_grained_labeled_data(dataset['train'])
        self.dev = self.get_fine_grained_labeled_data(dataset['dev'])
        self.test = self.get_fine_grained_labeled_data(dataset['test'])


    def get_train_dev_test_course(self, dataset):
        self.train = self.get_course_grained_labeled_data(dataset['train'])
        self.dev = self.get_course_grained_labeled_data(dataset['dev'])
        self.test = self.get_course_grained_labeled_data(dataset['test'])


    def get_fine_grained_labeled_data(self, training_data):
        data_x = []
        data_y = []

        for data in training_data:
            label, sentence = data.to_labeled_lines()[0]
            sentence = self.clean_sentence(sentence)
            data_x.append(sentence)
            data_y.append(int(label))

        return [np.array(data_x), np.array(data_y)]


    def get_course_grained_labeled_data(self, training_data):
        data_x = []
        data_y = []

        for data in training_data:
            label, sentence = data.to_labeled_lines()[0]
            sentence = self.clean_sentence(sentence)
            if label < 2:
                data_x.append(sentence)
                data_y.append(0)
            elif label > 2:
                data_x.append(sentence)
                data_y.append(1)

        return [np.array(data_x), np.array(data_y)]


    def clean_sentence(self, sentence):
        rev = [sentence.strip()]
        rev = self.clean_str_sst(' '.join(rev))
        return rev
    

    def clean_str_sst(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
        string = re.sub(r"\s{2,}", " ", string)    
        return string.strip().lower()


    def random_sample(self, data_x, data_y):
        data_x, temp_x, data_y, temp_y = train_test_split(data_x, data_y, test_size=0.60, stratify=data_y, random_state=42)
        return data_x, data_y


    def bert_encode(self, sentences):
        input_word_ids = []

        for sentence in sentences:
            sentence_encoded = self.get_input_example(sentence)
            input_word_ids.append(sentence_encoded)
        
        input_word_ids = tf.ragged.constant(input_word_ids)
        input_mask = tf.ones_like(input_word_ids).to_tensor()
        input_type = tf.zeros_like(input_word_ids).to_tensor()
        
        inputs = [input_word_ids.to_tensor(), input_mask, input_type]
        return inputs


    def get_input_example(self, sentence):
        sentence_tokens = list(self.tokenizer.tokenize(sentence))
        tokens = ['[CLS]'] + sentence_tokens + (['[PAD]'] * (self.sequence_length - len(sentence_tokens))) + ['[SEP]'] 
        input_word_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_word_ids


    def save_data(self, dataset):
        pickle.dump(dataset, open(self.config['data']['output'], 'wb'))
        print('data dictionary w/ \'train\', \'dev\', \'test\' keys as {}'.format(
            self.config['data']['output']
        ))


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)
    preprocessor = SSTDataPreprocessor(config)
