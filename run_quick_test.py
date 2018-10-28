from input.SmallNorb import SmallNorb
from learning.MatrixCapsNetEstimator import MatrixCapsNetEstimator
import dill as pickle
import config
import numpy as np
import os
import os.path

result_name = 'quick_test'

sn = SmallNorb.from_cache()

sn.reduce_to_two_examples(.3, .3)

mcne = MatrixCapsNetEstimator().init()

batch_size = 17
epoch_count = 50
max_steps = 50

results = mcne.train_and_test(sn, batch_size, epoch_count, max_steps, model_path=os.path.join(config.TF_MODEL_PATH, result_name))

result_file_path = os.path.join(config.OUTPUT_PATH, result_name + '.dill')

if not os.path.exists(os.path.dirname(result_file_path)):
    os.makedirs(os.path.dirname(result_file_path))

with open(result_file_path) as f:
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
