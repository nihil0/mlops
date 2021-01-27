import json
import os
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def init():
    global model
    global tokenizer

    print(os.getcwd())

    model_dir = os.environ["AZUREML_MODEL_DIR"]

    with open(f"{model_dir}/model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    model = load_model(f"{model_dir}/model/model.h5")


def run(raw_data):

    inputs = json.loads(raw_data)
    sequences = tokenizer.texts_to_sequences(inputs["componentNotes"])
    data = pad_sequences(sequences, maxlen=100)

    print(json.dumps(inputs["componentNotes"]))

    results = model.predict(data)

    results = {
        "model": "comp-condition-check-mcw2",
        "predictions": [
            "compliant" if int(m[0]) else "non-compliant" for m in results.tolist()
        ],
    }
    print(json.dumps(results))
    return results
