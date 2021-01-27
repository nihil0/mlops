"""
Downloads the appropriate version of a model
"""
import argparse

from azureml.core import Model, Run


def main(model_name, model_version, target_path):
    run = Run.get_context()
    models = Model.list(run.experiment.workspace, name=model_name)

    if model_version is not None:
        try:
            model = next(m for m in models if m.version == model_version)
        except StopIteration:
            raise ValueError("This version of the model was not found")
    else:
        print("Model version not specified. Using latest version.")
        model = max(models, key=lambda x: x.version)

    model.download(target_dir=target_path, exist_ok=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Location in which the downloaded model is stored",  # noqa
    )
    args = parser.parse_args()
    main(args.model_name, args.model_version, args.model_dir)
    print(f"Model written to {args.model_dir}")
