import pandas as pd
import tensorflow as tf

tf.random.set_seed(7)

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

import os
import torch
from tqdm import tqdm
from keras.models import load_model

import warnings

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"


class evaluation:
    def __init__(self):
        # checking for GPU and pytorch versions on device
        print("Num GPUs Available for tf: ", len(tf.config.list_physical_devices('GPU')))
        print(f'PyTorch version: {torch.__version__}')
        print(f'CUDNN version: {torch.backends.cudnn.version()}')
        print(f'Available GPU devices for Torch: {torch.cuda.device_count()}')
        print(f'Device Name: {torch.cuda.get_device_name()}')

        # Loading human behavior data from csv files for evaluation
        self.df_SPEEDED_RSVP = pd.read_csv('../data/human_behavior_data/SPEEDED_RSVP.csv')
        self.df_SPEEDED_SPR = pd.read_csv('../data/human_behavior_data/SPEEDED_SPR.csv')
        self.df_UNSPEEDED = pd.read_csv('../data/human_behavior_data/UNSPEEDED.csv')
        self.df_SPEEDED_RSVP['BERT'] = ''
        self.df_SPEEDED_SPR['BERT'] = ''
        self.df_UNSPEEDED['BERT'] = ''
        self.df_SPEEDED_RSVP['LSTM'] = ''
        self.df_SPEEDED_SPR['LSTM'] = ''
        self.df_UNSPEEDED['LSTM'] = ''
        self.df_SPEEDED_RSVP['RNN'] = ''
        self.df_SPEEDED_SPR['RNN'] = ''
        self.df_UNSPEEDED['RNN'] = ''

    def evaluate_lstm(self):
        model = load_model('../models/LSTM_model.keras')
        print("Model loaded successfully")

        # Inference from BERT model

        for i in tqdm(range(len(self.df_SPEEDED_RSVP['Preamble']))):
            text = self.df_SPEEDED_RSVP['Preamble'][i]
            self.df_SPEEDED_RSVP['LSTM'][i] = model.predict(text)

        for i in tqdm(range(len(self.df_SPEEDED_SPR['Preamble']))):
            text = self.df_SPEEDED_SPR['Preamble'][i]
            self.df_SPEEDED_SPR['LSTM'][i] = model.predict(text)
        for i in tqdm(range(len(self.df_UNSPEEDED['Preamble']))):
            text = self.df_UNSPEEDED['Preamble'][i]
            self.df_UNSPEEDED['LSTM'][i] = model.predict(text)

        # Saving the results in CSV files
        self.df_SPEEDED_RSVP.to_csv('../data/human_behavior_data/df_SPEEDED_RSVP.csv', index=False)
        self.df_SPEEDED_SPR.to_csv('../data/human_behavior_data/df_SPEEDED_SPR.csv', index=False)
        self.df_UNSPEEDED.to_csv('../data/human_behavior_data/df_UNSPEEDED.csv', index=False)

    def evaluate_rnn(self):
        model = load_model('../models/model_LSTM_2_epochs.h5')
        print("Model loaded successfully")

        # Inference from BERT model

        for i in tqdm(range(len(self.df_SPEEDED_RSVP['Preamble']))):
            text = self.df_SPEEDED_RSVP['Preamble'][i]
            self.df_SPEEDED_RSVP['RNN'][i] = model.predict(text)

        for i in tqdm(range(len(self.df_SPEEDED_SPR['Preamble']))):
            text = self.df_SPEEDED_SPR['Preamble'][i]
            self.df_SPEEDED_SPR['RNN'][i] = model.predict(text)
        for i in tqdm(range(len(self.df_UNSPEEDED['Preamble']))):
            text = self.df_UNSPEEDED['Preamble'][i]
            self.df_UNSPEEDED['RNN'][i] = model.predict(text)

        # Saving the results in CSV files
        self.df_SPEEDED_RSVP.to_csv('../data/human_behavior_data/df_SPEEDED_RSVP.csv', index=False)
        self.df_SPEEDED_SPR.to_csv('../data/human_behavior_data/df_SPEEDED_SPR.csv', index=False)
        self.df_UNSPEEDED.to_csv('../data/human_behavior_data/df_UNSPEEDED.csv', index=False)

    def evaluate_bert(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForSequenceClassification.from_pretrained("../models/training_model_bert_full_data")
        print("Model loaded successfully")

        # Inference from BERT model
        classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, device=0)
        for i in tqdm(range(len(self.df_SPEEDED_RSVP['Preamble']))):
            text = self.df_SPEEDED_RSVP['Preamble'][i]
            self.df_SPEEDED_RSVP['BERT'][i] = classifier(text)[0]['label']

        for i in tqdm(range(len(self.df_SPEEDED_SPR['Preamble']))):
            text = self.df_SPEEDED_SPR['Preamble'][i]
            self.df_SPEEDED_SPR['BERT'][i] = classifier(text)[0]['label']
        for i in tqdm(range(len(self.df_UNSPEEDED['Preamble']))):
            text = self.df_UNSPEEDED['Preamble'][i]
            self.df_UNSPEEDED['BERT'][i] = classifier(text)[0]['label']

        # Saving the results in CSV files
        self.df_SPEEDED_RSVP.to_csv('../data/human_behavior_data/df_SPEEDED_RSVP.csv', index=False)
        self.df_SPEEDED_SPR.to_csv('../data/human_behavior_data/df_SPEEDED_SPR.csv', index=False)
        self.df_UNSPEEDED.to_csv('../data/human_behavior_data/df_UNSPEEDED.csv', index=False)
