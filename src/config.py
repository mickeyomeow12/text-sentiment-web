import transformers

DEVICE = "cuda"
# 设置了模型输入的最大长度为 512。这是因为 BERT 模型的输入有长度限制，通常为 512 或 1024。如果输入长度超过这个限制，就需要截断。
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
# ACCUMULATION = 2
BERT_PATH = "../input/bert_base_uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/imdb.csv"
# 使用 transformers 模块中的 BertTokenizer 类从预训练的 BERT 模型创建了一个分词器实例，并设置了将输入文本转换为小写。在 BERT 模型中，首先需要将输入文本转换为 BERT 模型能理解的形式，也就是将每个单词转换为一个特定的数字（这个过程称为 tokenization）。
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
