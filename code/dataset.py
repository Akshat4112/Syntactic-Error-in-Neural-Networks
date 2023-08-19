import re
import io
import requests
import os
import pandas as pd
class dataset:
    def __init__(self):
        pass
    def download_data(self):
        pass
        # URL = "https://raw.githubusercontent.com/TalLinzen/rnn_agreement/master/data/wiki.vocab"
        # download = requests.get(URL).content
        # df = pd.read_csv(io.StringIO(download.decode('utf-8')), sep='\t')
        # os.system("wget http://tallinzen.net/media/rnn_agreement/rnn_agr_simple.tar.gz")

    def preprocess_data(self):
        df_train = pd.read_csv('../data/rnn_agr_simple/numpred.train', sep='\t', names=['POS', "Preamble"])
        df_val = pd.read_csv('../data/rnn_agr_simple/numpred.val', sep='\t', names=['POS', "Preamble"])

        df_train["Preamble"] = df_train["Preamble"].apply(lambda s: ' '.join(re.sub("[.(),``!?:;-='...@_]", " ", s).split()))
        df_val["Preamble"] = df_val["Preamble"].apply(
            lambda s: ' '.join(re.sub("[.(),``!?:;-='...@_]", " ", s).split()))


        # VBZ is singular and VBP is plural
        df_train.loc[df_train["POS"] == "VBZ", "POS"] = 0
        df_train.loc[df_train["POS"] == "VBP", "POS"] = 1

        df_val.loc[df_val["POS"] == "VBZ", "POS"] = 0
        df_val.loc[df_val["POS"] == "VBP", "POS"] = 1

        train_df = pd.DataFrame(columns=["labels", "text"])
        train_df['labels'] = df_train['POS']
        train_df['text'] = df_train['Preamble']

        test_df = pd.DataFrame(columns=["labels", "text"])
        test_df['labels'] = df_val['POS']
        test_df['text'] = df_val['Preamble']

        train_df.to_csv('../data/train_df.csv')
        test_df.to_csv('../data/test_df.csv')

        print("Data Preprocessed and Saved...")




