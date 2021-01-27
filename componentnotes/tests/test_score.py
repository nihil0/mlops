import json
import os

import yaml

import componentnotes.score as score

conf_file = os.path.join(os.path.dirname(__file__), "..", "conf.yaml")


def test_init(tmpdir):
    tmpdir.mkdir("model")

    with open(conf_file, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    model_name = conf["metadata"]["model_name"]
    model_version = conf["metadata"]["model_version"]

    os.environ[
        "AZUREML_MODEL_DIR"
    ] = f"/var/azureml-app/azureml-models/{model_name}/{model_version}"

    score.init()
    members = dir(score)
    assert "model" in members
    assert "tokenizer" in members


def test_run():
    sample_input = {
        "componentNotes": [
            "iron component manufactured in 1998 in good condition",
            "manufactured in 2017 made of steel in good condition",
        ]
    }

    res = score.run(json.dumps(sample_input))
    assert res == {
        "model": "comp-condition-check-mcw2",
        "predictions": ["compliant", "non-compliant"],
    }
