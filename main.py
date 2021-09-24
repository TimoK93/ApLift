"""
The main script to train the classifier and use the tracker.
To use the script a config.yaml needs to be specified.

Example usage:

    python3 main.py config/example_config.yaml

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPUs are not necessary for our framework!

from src.utilities.colors import PrintColors
from src.datasets import Data, get_sequences
from train import train

from src.evaluate import evaluate_sequence
from src.calculate_features import augment_features
from src.TrackingModel import load_model
from src.utilities.config_reader import main_function


@main_function
def main(**kwargs):
    """ Wraps the pipeline call and reads out the config file """
    run_pipeline(**kwargs)


def run_pipeline(working_dir, training_config, model_config, data_config):
    print("%s=== Run Aplift Pipeline ===%s\n" % (PrintColors.HEADER, PrintColors.ENDC))

    ''' Add global context normalizations to feature list '''
    edge_features = data_config["features"]
    valid_normalization = data_config["features_to_normalize"]
    augment_feature_names = augment_features(edge_features, valid_normalization)
    data_config["features"] = augment_feature_names
    model_config["features"] = augment_feature_names

    ''' Load Dataset and choose train / inference sequences'''
    dataset = Data(**data_config)
    dataset.sequences_for_training_loop = get_sequences(
        dataset.train_sequences, training_config.sequences_for_training)
    dataset.sequences_for_inference = get_sequences(
        dataset.train_sequences, training_config.sequences_for_inference, strict=False)
    if len(dataset.sequences_for_inference) == 0:
        dataset.sequences_for_inference = get_sequences(
            dataset.test_sequences, training_config.sequences_for_inference, strict=False)
    assert len(dataset.sequences_for_training_loop) >= 1
    assert len(dataset.sequences_for_inference) == 1

    ''' Determine directory names '''
    sequence_name = dataset.sequences_for_inference[0]["name"]
    sequence_directory = os.path.join(working_dir, sequence_name)
    os.makedirs(sequence_directory, exist_ok=True)
    if len(dataset.sequences_for_training_loop) == len(dataset.train_sequences):
        training_directory = os.path.join(working_dir, "train_all")
    else:
        training_directory = os.path.join(working_dir, sequence_name)

    ''' Load model '''
    checkpoint_path = os.path.join(training_directory, "model.pkl")
    model_exists = os.path.exists(checkpoint_path)
    if model_exists:
        print("Use existing checkpoint", checkpoint_path)
        model_config["checkpoint_path"] = checkpoint_path
        model = load_model(**model_config)
    else:
        model = load_model(**model_config)
    model.cpu()

    ''' Train the network if no model exists '''
    if not model_exists:
        train(
            directory=training_directory,
            model=model,
            dataset=dataset,
            training_config=training_config
        )

    ''' Evaluate validation sequence '''
    result_path = os.path.join(training_directory, sequence_name + ".txt")
    if os.path.exists(result_path) and False:
        print("Result file is still existing! Skip inference...")
        metrics = {}
    else:
        metrics = evaluate_sequence(
            model=model,
            dataset=dataset,
            result_dir=sequence_directory,
        )
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
