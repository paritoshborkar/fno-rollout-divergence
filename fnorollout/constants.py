from enum import Enum


class DataConfigKey(Enum):
    PATH = "path"
    SPLIT_TRAIN = "split.train"
    SPLIT_VAL = "split.val"
    N_TRAIN = "n_train"
    BATCH_SIZE = "batch_size"
    N_TESTS = "n_tests"
    TEST_RESOLUTION = "test_resolution"
    TEST_BATCH_SIZES = "test_batch_sizes"