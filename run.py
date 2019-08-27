
from input.SmallNorb import SmallNorb
from learning.MatrixCapsNetEstimator import MatrixCapsNetEstimator
import dill as pickle
import numpy as np
import os.path
import datetime
from Configuration import Configuration
import sys
import tensorflow as tf

def run(result_name, config, sn):
    if config["reduce_to_two_examples"]:
        sn.reduce_to_two_examples(.3, .3)

    total_processed_examples = config["example_count"]  # = the number of examples necessary to bring the spread loss to .9 more or less

    batch_size = config["batch_size"]

    epoch_count = int(total_processed_examples / sn.training_example_count())  # = 600
    max_steps = int(total_processed_examples / batch_size)  # = 1143529

    save_summary_steps = max_steps / config["save_summary_count"]

    mcne = MatrixCapsNetEstimator().init(
        architecture=config["architecture"],
        initialization=config["initialization"],
        regularization=config["regularization"],
        save_summary_steps=save_summary_steps,
        eval_steps=config["eval_steps"],
        dtype={"float32": tf.float32, "float16": tf.float16}[config["dtype"]]
    )

    results = mcne.train_and_test(sn, batch_size, epoch_count, max_steps, model_path=os.path.join(config["tf_model_dir"], result_name))

    result_file_path = os.path.join(config["output"], result_name + '.dill')

    if not os.path.exists(os.path.dirname(result_file_path)):
        os.makedirs(os.path.dirname(result_file_path))

    with open(result_file_path, 'wb') as f:
        pickle.dump(results, f)

    test_result, validation_result, test_predictions = results

    print("TEST RESULTS")
    print(test_result)

    print("VALIDATION RESULTS")
    print(validation_result)

    # Some basic testing
    extracted_test_predictions = list(test_predictions)

    activations_array = np.concatenate(list(map(lambda v: v["activations"], extracted_test_predictions)))
    activations_finite = np.isfinite(activations_array).all()

    poses_array = np.concatenate(list(map(lambda v: v["poses"], extracted_test_predictions)))
    poses_finite = np.isfinite(poses_array).all()

    output_finite = activations_finite and poses_finite

    print("TEST PREDICTIONS FINITE?:")
    print(output_finite)

if __name__ == '__main__':
    config = Configuration.load(sys.argv[1])

    result_name = os.path.basename(sys.argv[1])

    #result_name = config.parse_arguments()

    sn = SmallNorb.from_cache()
    run(result_name, config.config, sn)
