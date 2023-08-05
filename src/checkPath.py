import os

BERT_PATH = "../input/bert_base_uncased/"
TRAINING_FILE = "../input/imdb.csv"
print(os.path.exists(BERT_PATH))
print(os.path.exists(TRAINING_FILE))
print(os.path.exists(os.path.join(BERT_PATH, "vocab.txt")))
print(os.path.exists(os.path.join(BERT_PATH, "pytorch_model.bin")))
