import score
import json

def test_init():
    score.init()

def test_run():
    sample_input = {
        "componentNotes": [
            "iron component manufactured in 1998 in good condition",
            "manufactured in 2017 made of steel in good condition"
        ]
    }

    res = score.run(json.dumps(sample_input))
    print(res)


