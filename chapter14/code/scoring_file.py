import json
import os
from transformers import AutoModel
from azureml.core import Model

def init():
    global model
    model_path = Model.get_model_path('sentiment-analysis')
    model = AutoModel.from_pretrained(model_path, from_tf=True)

def run(request):
    try:
        data = json.loads(request)
        text = data['query']
        sentiment = model(text)
        result = {}
        result['sentiment'] = sentiment
        return result
    except Exception as e:
        error = str(e)
        return error
