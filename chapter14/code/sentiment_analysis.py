import json
from transformers import pipeline

def init():
    global sentiment_analysis
    sentiment_analysis = pipeline("sentiment-analysis")

def run(request):
    data = json.loads(request)
    text = data['query']
    sentiment = sentiment_analysis(text)
    result = {}
    result['sentiment'] = sentiment
    return result