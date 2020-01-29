import os
import json
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from azureml.core.model import Model


def init():
    global model
    global tokenizer

    try:
        model_name = "component-condition-check"
        model_version = "1"
        model_dir = f"azureml-models/{model_name}/{model_version}"

        with open(f"{model_dir}/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        model = load_model(f"{model_dir}/model.h5")
        
    except Exception as e:
        print(e)
        
# note you can pass in multiple rows for scoring
def run(raw_data):
    import time
    try:
        print("Received input: ", raw_data)
        
        inputs = json.loads(raw_data)

        sequences = tokenizer.texts_to_sequences(inputs["componentNotes"])
        data = pad_sequences(sequences, maxlen=100)

        results = model.predict(data)

        results = { "predictions": ["compliant" if int(m[0]) else "non-compliant" for m in results.tolist()] }
        return json.dumps(results)

    except Exception as e:
        error = str(e)
        print("ERROR: " + error + " " + time.strftime("%H:%M:%S"))
        return error