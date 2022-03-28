from transformers import pipeline

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

def init():
    global sentiment_analysis
    sentiment_analysis = pipeline("sentiment-analysis")

sample_input = StandardPythonParameterType([
    "My ML skills are quite good.",
    "I didn't have good experience with ML."])
inputs = StandardPythonParameterType({'data': sample_input})

sample_output = StandardPythonParameterType([0.95, -0.95])
outputs = StandardPythonParameterType({'Results': sample_output})

@input_schema('Inputs', inputs)
@output_schema(outputs)
def run(Inputs):
    try:
        data = Inputs['data']
        return score(data)
    except Exception as e:
        error = str(e)
        return error

def score(data):
    assert isinstance(data, list)
    results = sentiment_analysis(data)
    return list(map(convert_score, results))

def convert_score(result):
    # Convert negative sentiments to negative scores
    if result['label'] == 'NEGATIVE':
        return -1.0 * result['score']
    return result['score']