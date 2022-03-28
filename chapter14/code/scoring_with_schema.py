import numpy as np
import pandas as pd

import os
from transformers import AutoModel

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    model_path = os.getenv("AZUREML_MODEL_DIR")
    model = AutoModel.from_pretrained(model_path, from_tf=True)

input_sample = pd.DataFrame(data=[{'query': "AzureML is quite good."}])
output_sample = np.array([np.array(["POSITIVE", 0.95])])

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        text = data['query']
        sentiment = model(text)
        result = {}
        result['sentiment'] = sentiment
        return result
    except Exception as e:
        error = str(e)
        return error

