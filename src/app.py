import config
import torch
import flask
from flask import Flask, render_template
from flask import request
import time

import torch.nn as nn
from model import BERTBaseUncased

app = Flask(__name__)

MODEL = None
DEVICE = config.DEVICE
PREDICTION_DICT = dict()


def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len,truncation=True
    )
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST','GET'])
def predict_sentiment():
    # sentence = request.args.get("sentence")
    # # print(sentence)
    # start_time = time.time()
    # positive_prediction = sentence_prediction(sentence)
    # negative_prediction = 1 - positive_prediction
    # response = {}
    # response["response"] = {
    #     "positive": str(positive_prediction),
    #     "negative": str(negative_prediction),
    #     "sentence": str(sentence),
    #     "time_taken": str(time.time() - start_time),
    # }
    # return flask.jsonify(response)
    # if request.method == 'POST':
    #     sentence = request.form.get("sentence")
    #     print(sentence)
    #     start_time = time.time()
    #     positive_prediction = sentence_prediction(sentence)
    #     negative_prediction = 1 - positive_prediction
    #     time_taken = str(time.time() - start_time)
    #
    #     return render_template('index.html', sentence=sentence, positive=positive_prediction,
    #                            negative=negative_prediction, time_taken=time_taken)
    # else:
    #     return render_template('index.html')

    sentence = request.form.get("sentence")
    positive = "None"
    negative = "None"
    time_taken = "None"
    result = "None"

    if sentence:  # check if sentence is not empty
        start_time = time.time()
        positive_prediction = sentence_prediction(sentence)
        negative_prediction = 1 - positive_prediction
        time_taken = str(time.time() - start_time)

        positive = str(positive_prediction)
        negative = str(negative_prediction)
        result = "Positive" if positive_prediction > 0.5 else "Negative"

    return render_template('index.html', sentence=sentence, positive=positive,
                           negative=negative, time_taken=time_taken, result=result)


if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host="0.0.0.0")
