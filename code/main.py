from dataset import dataset
from train import training_model

# data_obj = dataset()
# data_obj.download_data()
# data_obj.preprocess_data()

train_obj = training_model()
train_obj.prepare_training()

# train_obj.train_rnn()
train_obj.train_lstm()
# train_obj.train_bert()
# train_obj.train_roberta()
# train_obj.train_gpt2()

