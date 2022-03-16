
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

    save_summary_steps = 500

    print("TEXTURE WEIGHT DECAY")
    print(config["texture_weight_decay"])

    print("CAPSULE WEIGHT DECAY")
    print(config["capsule_weight_decay"])

    mcne = MatrixCapsNetEstimator().init(
        architecture=config["architecture"],
        initialization=config["initialization"],
        regularization=config["regularization"],
        save_summary_steps=save_summary_steps,
        eval_steps=config["eval_steps"],
        dtype={"float32": tf.float32, "float16": tf.float16}[config["dtype"]],
        spread_loss_decay_factor=config["spread_loss_decay_factor"],
        encoding_convolution_weight_stddev=config["encoding_convolution_weight_stddev"] if "encoding_convolution_weight_stddev" in config else None,
        primary_pose_weights_stddev=config["primary_pose_weights_stddev"] if "primary_pose_weights_stddev" in config else None,
        primary_activation_weight_stddev=config["primary_activation_weight_stddev"] if "primary_activation_weight_stddev" in config else None,
        initial_learning_rate=config["initial_learning_rate"] if "initial_learning_rate" in config else 0.001,
        learning_rate_decay_rate=config["learning_rate_decay_rate"] if "learning_rate_decay_rate" in config else 1.0,
        learning_rate_decay_steps=config["learning_rate_decay_steps"] if "learning_rate_decay_steps" in config else 1,
        learning_rate_use_staircase=config["learning_rate_use_staircase"] if "learning_rate_use_staircase" in config else False,
        clip_learning_rate=config["clip_learning_rate"] if "clip_learning_rate" in config else 0.0,
        texture_weight_decay=config["texture_weight_decay"] if "texture_weight_decay" in config else 0.0,
        capsule_weight_decay=config["capsule_weight_decay"] if "capsule_weight_decay" in config else 0.0,
        detach_primary_poses=config["detach_primary_poses"] if "detach_primary_poses" in config else False,
        identity_primary_poses=config["identity_primary_poses"] if "identity_primary_poses" in config else False
    )

    print('max_steps')
    print(max_steps)

    print('epoch_count')
    print(epoch_count)

    print('batch_size')
    print(batch_size)

    if "no_validation" in config and config["no_validation"]:
        results = mcne.train_and_test_without_validation_split(sn, batch_size, epoch_count, max_steps,
                                      model_path=os.path.join(config["tf_model_dir"], result_name))
    else:
        results = mcne.train_and_test(sn, batch_size, epoch_count, max_steps, model_path=os.path.join(config["tf_model_dir"], result_name))

    result_file_path = os.path.join(config["output_path"], result_name + '.dill')

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
