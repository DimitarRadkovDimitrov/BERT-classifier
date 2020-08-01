import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks


class BERTClassifier:
    """ BERT Classifier """

    def __init__(self, train_data_size, config):
        self.config = config
        self.num_classes = config['data']['num_classes']
        self.sequence_length = config['data']['seq_length'] + 2
        self.bert_hub_url = config['BERT']['hub_url']
        self.trainable = not(config['BERT']['static'])
        self.batch_size = config['BERT']['batch_size']
        self.num_epochs = config['BERT']['num_epochs']
        self.model = self.build_model(train_data_size)


    def build_model(self, train_data_size):
        input_word_ids = Input(shape=(self.sequence_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(self.sequence_length,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(self.sequence_length,), dtype=tf.int32, name="segment_ids")
        bert_layer = hub.KerasLayer(
            self.bert_hub_url,
            trainable=self.trainable,
            name='bert_layer'
        )
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        output = layers.Lambda(lambda x: x[:,0,:])(sequence_output)
        output = layers.Dense(self.num_classes, activation='softmax')(output)

        steps_per_epoch = int(train_data_size / self.batch_size)
        num_train_steps = steps_per_epoch * self.num_epochs
        warmup_steps = int(self.num_epochs * train_data_size * 0.1 / self.batch_size)

        optimizer = nlp.optimization.create_optimizer(
            2e-5, 
            num_train_steps=num_train_steps, 
            num_warmup_steps=warmup_steps
        )
        model = Model(
            inputs=[input_word_ids, input_mask, segment_ids], 
            outputs=output, 
            name='bert_classifier'
        )
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        return model


    def train(self, train_x, train_y, dev_x=None, dev_y=None):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)      
        train_y = to_categorical(train_y, num_classes=self.num_classes)

        if dev_x is not None and dev_y is not None:
            dev_y = to_categorical(dev_y, num_classes=self.num_classes)
            history = self.model.fit(
                [train_x[0], train_x[1], train_x[2]],
                train_y,
                self.batch_size,
                self.num_epochs,
                validation_data=(dev_x, dev_y),
                callbacks=[early_stopping],
                shuffle=True
            )
        else:
            history = self.model.fit(
                [train_x[0], train_x[1], train_x[2]],
                train_y,
                self.batch_size,
                self.num_epochs,
                validation_split=0.20,
                callbacks=[early_stopping],
                shuffle=True
            )

        return history


    def evaluate(self, test_x, test_y):
        target_names = ['class {}'.format(i) for i in range(self.num_classes)]
 
        pred_y = self.model.predict(
            [test_x[0], test_x[1], test_x[2]],
            self.batch_size,
            verbose=1
        )
        pred_y = np.argmax(pred_y, axis=1)

        report = classification_report(
            test_y,
            pred_y,
            target_names=target_names,
            digits=4
        )
        return report
