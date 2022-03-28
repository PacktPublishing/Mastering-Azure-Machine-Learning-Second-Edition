import pandas as pd
import numpy as np

from transformers import pipeline

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

HF_MODEL = "sentiment-analysis"

PARAMS = ["query"]
IN_VALUES = ["My ML skills are quite good."]
OUT_VALUES = [np.array(["POSITIVE", 0.95])]

def init():
    global sentiment_analysis
    sentiment_analysis = pipeline(HF_MODEL)

input_sample = pd.DataFrame(data=[{PARAMS[0]: IN_VALUES[0]}])
output_sample = np.array(OUT_VALUES)

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    print(data)
    try:
        result = score(data)
        print(result)
        return result
    except Exception as e:
        error = str(e)
        return error

def score(df):
    assert isinstance(df, pd.DataFrame)
    result = df.apply(score_sentiment, axis=1)
    return result.values.tolist()

def score_sentiment(s):
    return sentiment_analysis(s[PARAMS[0]])[0]
