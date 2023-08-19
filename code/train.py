import matplotlib.pyplot as plt
import nltk
import pandas as pd
import numpy as np
import sklearn
import gensim
from gensim.models import Word2Vec

import keras
import tensorflow
from keras.preprocessing.text import one_hot,Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input,LSTM
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split

from keras.initializers import Constant
from keras.layers import ReLU
from keras.layers import Dropout
import tensorflow as tf
tensorflow.random.set_seed(7)

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
import logging

class training_model:
    def __init__(self):
        print("Num GPUs Available for tf: ", len(tf.config.list_physical_devices('GPU')))
        print(f'PyTorch version: {torch.__version__}')
        print(f'CUDNN version: {torch.backends.cudnn.version()}')
        print(f'Available GPU devices for Torch: {torch.cuda.device_count()}')
        print(f'Device Name: {torch.cuda.get_device_name()}')

    def prepare_training(self):
        """
         Reads training and test data from CSV files, tokenizes the text data,
         trains a Word2Vec model on the tokenized data, and prepares the training
         and testing datasets for a machine learning model.
         Parameters:
         None
         Returns:
         None
         """

        self.df_train = pd.read_csv('../data/train_df.csv')
        self.df_test = pd.read_csv('../data/test_df.csv')
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        sentences = []
        sum = 0
        for pre in self.df_train['text']:
            sents = tokenizer.tokenize(pre.strip())
            sum += len(sents)
            for sent in sents:
                sentences.append(sent.split())

        model = gensim.models.Word2Vec(sentences=sentences, vector_size=250, window=10, min_count=1)

        model.train(sentences, epochs=10, total_examples=len(sentences))
        vocab = model.wv  # unique words i.e vocab in the dataset
        print("Vocab Length is: ", len(vocab))

        words = list(model.wv.index_to_key)
        word2vec_dict = {}
        for word in words:
            word2vec_dict[word] = model.wv.get_vector(word)

        max = -1
        for i, pre in enumerate(self.df_train['text']):
            tokens = pre.split()
            if (len(tokens) > max):
                max = len(tokens)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.df_train['text'])
        vocab_size = len(tokenizer.word_index) + 1
        encoded_vec = tokenizer.texts_to_sequences(self.df_train['text'])

        self.max_len = 47
        self.vocab_size = len(tokenizer.word_index) + 1
        self.embedding_dim = 250

        pad = pad_sequences(encoded_vec, maxlen=self.max_len, padding='post')

        # embedding matrix
        self.embed_matrix = np.zeros(shape=(vocab_size, self.embedding_dim))
        for word, i in tokenizer.word_index.items():
            vector = word2vec_dict.get(word)
            if vector is not None:
                self.embed_matrix[i] = vector

        Y = keras.utils.to_categorical(self.df_train['labels'])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(pad, Y, test_size=0.20, random_state=42)

    def train_rnn(self):
        pass

    def train_lstm(self):
        """
        Trains an LSTM model for text classification.
        Parameters:
            None
        Returns:
            None
        """
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_len,
                            embeddings_initializer=Constant(self.embed_matrix)))
        model.add(LSTM(128, return_sequences=False))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.20))
        model.add(Dense(2, activation='sigmoid'))

        model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        History = model.fit(self.x_train, self.y_train, epochs=20, batch_size=64, validation_split=0.2)

        model.save("../models/model_LSTM.h5")
        plt.plot(History.history['loss'])
        plt.plot(History.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("../figures/model_loss_RNN.png")

        plt.plot(History.history['accuracy'])
        plt.plot(History.history['val_accuracy'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show("../figures/model_accuracy_RNN.png")

    def train_bert(self):
        """
        Trains a BERT model on the provided training data.

        Returns:
            None
        """
        # logging.basicConfig(level=logging.INFO)
        # transformers_logger = logging.getLogger("transformers")
        # transformers_logger.setLevel(logging.WARNING)

        sample_df = self.df_train[1000:1500]
        eval_df = self.df_train[:1000]
        print(sample_df.head())

        model_args = ClassificationArgs(num_train_epochs=10, overwrite_output_dir=True, train_batch_size=16,
                                        evaluate_during_training=True)
        model = ClassificationModel("bert", "bert-base-cased", args=model_args, num_labels=2)

        model.train_model(train_df = sample_df, eval_df=eval_df, show_running_loss=True, acc=sklearn.metrics.accuracy_score)

        test_sample_df = self.df_train.sample(n=100)
        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(test_sample_df)
        print(result)
        precision = result['tp'] / (result['tp'] + result['fp'])
        recall = result['tp'] / (result['tp'] + result['fn'])
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(precision, recall, f1_score)

        model.model.save_pretrained('../models/bert-finetuned')
        model.tokenizer.save_pretrained('../models/bert-finetuned')
        model.config.save_pretrained('../models/bert-finetuned/')

    def train_roberta(self):
        sample_df = self.df_train[1000:1500]
        eval_df = self.df_train[:1000]
        print(sample_df.head())

        model_args = ClassificationArgs(num_train_epochs=10, overwrite_output_dir=True, train_batch_size=16,
                                        evaluate_during_training=True)
        model = ClassificationModel("roberta", "roberta-base", args=model_args, num_labels=2)

        model.train_model(train_df=sample_df, eval_df=eval_df, show_running_loss=True,
                          acc=sklearn.metrics.accuracy_score)

        test_sample_df = self.df_train.sample(n=100)
        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(test_sample_df)
        print(result)
        precision = result['tp'] / (result['tp'] + result['fp'])
        recall = result['tp'] / (result['tp'] + result['fn'])
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(precision, recall, f1_score)
        pass

    def train_gpt2(self):
        pass

