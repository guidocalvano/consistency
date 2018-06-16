import numpy as np
import tensorflow as tf
import os
from input.SmallNorb import SmallNorb
from input.MatFileReaderWriter import MatFileReaderWriter


class TestSmallNorb(tf.test.TestCase):

    def setUp(self):
        self.training_base_path = os.path.dirname(os.path.abspath(__file__)) + '/trash/training_'
        self.test_base_path = os.path.dirname(os.path.abspath(__file__)) + '/trash/test_'

        self.training_examples_file_path = self.training_base_path + 'dat.mat'
        self.test_examples_file_path = self.test_base_path + 'dat.mat'

        self.training_labels_file_path = self.training_base_path + 'cat.mat'
        self.test_labels_file_path = self.test_base_path + 'cat.mat'

        self.training_info_file_path = self.training_base_path + 'info.mat'
        self.test_info_file_path = self.test_base_path + 'info.mat'

        self.tearDown()  # invoke tear down to assure clean starting state in case previous test crashed

        self.meta_features = 5
        self.label_count = 3
        self.example_count = 10
        self.stereo_scopic_image_count = 2
        self.non_stereo_scopic_example_count = self.stereo_scopic_image_count * self.example_count
        self.width = 96
        self.height = 96

        def random_examples():
            return (np.random.random([
            self.example_count,
            self.stereo_scopic_image_count,
            self.width,
            self.height]) * 256).astype('uint8')

        def random_labels():

            labels = np.zeros([self.example_count])
            sections = (float(self.example_count) * np.arange(self.label_count + 1).astype('float32') / float(self.label_count)).astype('int32')

            for i in range(self.label_count):
                labels[sections[i]:sections[i+1]] = i

            np.random.shuffle(labels)

            return labels.astype('int32')

        def random_info():
            return (np.random.random([self.example_count, self.meta_features]) * 6).astype('int32')

        self.raw_training_examples = random_examples()
        self.raw_test_examples = random_examples()

        self.raw_training_labels = random_labels()
        self.raw_test_labels = random_labels()

        self.raw_training_info = random_info()
        self.raw_test_info = random_info()

        MatFileReaderWriter.write_mat_file(self.training_examples_file_path, self.raw_training_examples)
        MatFileReaderWriter.write_mat_file(self.training_labels_file_path, self.raw_training_labels)
        MatFileReaderWriter.write_mat_file(self.training_info_file_path, self.raw_training_info)

        MatFileReaderWriter.write_mat_file(self.test_examples_file_path, self.raw_test_examples)
        MatFileReaderWriter.write_mat_file(self.test_labels_file_path, self.raw_test_labels)
        MatFileReaderWriter.write_mat_file(self.test_info_file_path, self.raw_test_info)


    def tearDown(self):

        for file_path in [
            self.training_examples_file_path,
            self.training_info_file_path,
            self.training_labels_file_path,
            self.training_examples_file_path,
            self.training_info_file_path,
            self.training_labels_file_path
        ]:
            if os.path.isfile(file_path):
                os.remove(file_path)

    def test_internal_representation(self):
        sn = SmallNorb().init(self.training_base_path, self.test_base_path)

        training_examples = np.concatenate([sn.training["examples"], sn.validation["examples"]])
        test_examples = sn.test["examples"]

        training_info = np.concatenate([sn.training["meta"], sn.validation["meta"]])
        training_labels = np.concatenate([sn.training["labels"], sn.validation["labels"]])


        self.assertTrue(not np.allclose(training_examples, test_examples, 0.001), "training and test set must be different")

        # mean is 0

        self.assertAlmostEqual(np.mean(training_examples),
                               0.0,
                               2,
                               "normalized training set must have approximate mean of 0")

        # variance is close to 1
        self.assertAlmostEqual(np.var(training_examples),
                               1.0,
                               2,
                               "normalized training set must have approximate variance of 1")
        # dimensions are (2 * example_count) x 48 x 48
        self.assertAllEqual(
            training_examples.shape,
            [self.example_count * 2, 48, 48, 1],
            "shape must be loaded correctly")

        self.assertAllEqual(training_labels.shape, [self.example_count * 2], "labels must be loaded with correct shape")
        self.assertAllEqual(training_info.shape,
                            [self.example_count * 2, self.meta_features + 1],
                            "meta data must be loaded with correct shape")

        self.assertAllEqual(np.sort(np.unique(training_labels)), np.arange(self.label_count))

    def test_default_data_set_construction(self):
        sn = SmallNorb().init(self.training_base_path, self.test_base_path)

        training_set = sn.default_training_set()
        validation_set = sn.default_validation_set()
        test_set = sn.default_test_set()

        # test if training and validation split numbers add up to the right size

        self.assertTrue((training_set[0].shape[0] + validation_set[0].shape[0]) == self.non_stereo_scopic_example_count,
                        "examples in training set and validation set should add up to the original example count")

        # make sure validation and training do not share items
        training_hash = 100 * sn.training["labels"] + sn.training["meta"][:, 0]
        validation_hash = 100 * sn.validation["labels"] + sn.validation["meta"][:, 0]

        self.assertTrue((~np.isin(training_hash, validation_hash)).all(), "no training examples in validation set")
        self.assertTrue((~np.isin(validation_hash, training_hash)).all(), "no validation examples in training set")

        # test if outputted images are of correct dimensions

        self.assertTrue(list(map(lambda x: x.value, training_set[0].shape[1:])) == [32, 32, 1], "output images should be 32 by 32 by 1")
        self.assertTrue(list(map(lambda x: x.value, validation_set[0].shape[1:])) == [32, 32, 1], "output images should be 32 by 32 by 1")
        self.assertTrue(list(map(lambda x: x.value, test_set[0].shape[1:])) == [32, 32, 1], "output images should be 32 by 32 by 1")

        # test if test set has the right number of elements
        self.assertTrue(test_set[0].shape[0].value == self.non_stereo_scopic_example_count, "tests not implemented")


    def test_trackable_stream_data_set_construction(self):
        sn = SmallNorb().init(self.training_base_path, self.test_base_path)

        epoch_count = 2
        batch_size = 3

        training_set = sn.trackable_stream_training_set(epoch_count, batch_size)
        test_set = sn.trackable_stream_test_set(batch_size)

        # is the training set shape correct?

        # is the training epoch count correct

        # is the reset cue correct?

        # is the progress cue

        # is the test set shape correct

        # is the test epoch count correct

        # is the reset cue correctly always true?

        # is the progress cue correctly always 1?

        # are training and test set different?

        self.assertTrue(False, "tests not implemented")
