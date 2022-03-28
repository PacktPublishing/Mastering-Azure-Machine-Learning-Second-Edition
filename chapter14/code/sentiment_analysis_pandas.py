import json
import pandas as pd

from transformers import pipeline

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

INPUTS = 'Inputs'
DATA = 'data'
RESULTS = 'Results'

HF_MODEL = "sentiment-analysis"
HF_MODEL_PARAMS = ["label", "score"]
HF_MODEL_PREFIX = "sentiment"

IN_PARAMS = ["query"]
IN_VALUES = ["My ML skills are quite good."]

OUT_PARAMS = HF_MODEL_PARAMS
OUT_VALUES = ["POSITIVE", 0.95]

def init():
    global sentiment_analysis
    sentiment_analysis = pipeline(HF_MODEL)

input_sample = pd.DataFrame(data=[{IN_PARAMS[0]: IN_VALUES[0]}])
inputs = StandardPythonParameterType({DATA: PandasParameterType(input_sample)})

output_sample = pd.DataFrame(data=[{OUT_PARAMS[0]: OUT_VALUES[0], OUT_PARAMS[1]: OUT_VALUES[1]}])
outputs = PandasParameterType(output_sample)

@input_schema(INPUTS, inputs)
@output_schema(outputs)
def run(Inputs):
    try:
        data = Inputs[DATA]
        return score(data)
    except Exception as e:
        error = str(e)
        return error

def score(df):
    assert isinstance(df, pd.DataFrame)
    result = df.apply(score_sentiment, axis=1)
    return result.to_json(orient='records')

def score_sentiment(s):
    return sentiment_analysis(s[IN_PARAMS[0]])[0]
