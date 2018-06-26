import os

PROJECT_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
SMALL_NORB_ROOT = os.path.join(DATA_ROOT, 'smallNORB')
SMALL_NORB_TRAINING = os.path.join(SMALL_NORB_ROOT, 'smallnorb-5x46789x9x18x6x2x96x96-training-')
SMALL_NORB_TEST = os.path.join(SMALL_NORB_ROOT, 'smallnorb-5x01235x9x18x6x2x96x96-testing-')
CACHE_ROOT = os.path.join(PROJECT_ROOT, 'cache')
SMALL_NORB_CACHE = os.path.join(CACHE_ROOT, 'small_norb_cache.dill')

TF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'tf_models')
TF_DEBUG_MODEL_PATH = os.path.join(PROJECT_ROOT, 'trash', 'model_dir')
TF_DEBUG_TRAINING_TIMELINE_FILE_PATH = os.path.join(PROJECT_ROOT, 'trash', 'train_timeline.json')
TF_DEBUG_EVAL_TIMELINE_FILE_PATH = os.path.join(PROJECT_ROOT, 'trash', 'eval_timeline.json')

OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'output')

RESULT_FILE = os.path.join(OUTPUT_PATH, 'results.dill')
MINI_RUN_RESULT_FILE = os.path.join(OUTPUT_PATH, 'mini_run_results.dill')