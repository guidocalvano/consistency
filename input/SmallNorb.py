import numpy as np
import tensorflow as tf
import sys
import pickle
import os.path
from input.MatFileReaderWriter import MatFileReaderWriter
import datetime
from matplotlib import pyplot as plt
import cv2
import config


class SmallNorb:

    INFO_SIZE_PER_TYPE = np.array([5, 8, 10, 5])

    @staticmethod
    def from_cache(cache_file_path=config.SMALL_NORB_CACHE):
        if not os.path.isfile(cache_file_path):
            sn = SmallNorb()
            sn.init()
            with open(cache_file_path, 'wb') as f:
                pickle.dump(sn, f)
            return sn

        with open(cache_file_path, 'rb') as f:
            sn = pickle.load(f)

        return sn

    def init(self, base_path_training=config.SMALL_NORB_TRAINING, base_path_test=config.SMALL_NORB_TEST):

        unsplit_training_data =  self.load_and_convert_to_internal_representation(base_path_training)

        split_training_data = self.split_by_objects(unsplit_training_data, .8)

        self.training = split_training_data["training"]
        self.validation = split_training_data["validation"]
        self.test = self.load_and_convert_to_internal_representation(base_path_test)

        return self

    def load_and_convert_to_internal_representation(self, file_path_base):
        data_set = self.load_set(file_path_base)

        normalized = self.normalize_data_set(data_set)
        print(str(datetime.datetime.now()) + ' NORMALIZATION DONE')

        no_stereo_scopic = self.represent_stereoscopy_as_meta_data(normalized)
        print(str(datetime.datetime.now()) + ' REMOVED STEREOSCOPY')

        downsampled = self.downsample(no_stereo_scopic)
        print(str(datetime.datetime.now()) + ' DOWNSAMPLED')

        return downsampled

    def split_by_objects(self, data, ratio):

        rows_per_split = self.split_objects_stratified_by_label(data, ratio)

        training_data = self.filter_data_on_row_ids(data, rows_per_split["training"])

        validation_data = self.filter_data_on_row_ids(data, rows_per_split["validation"])

        return {
            "training": training_data,
            "validation": validation_data
        }

    def split_objects_stratified_by_label(self, data, ratio):

        unique_labels = np.unique(data["labels"])

        training_row_index_blocks = []
        validation_row_index_blocks = []

        for label in unique_labels:

            label_rows_BI = (data["labels"] == label)  # BI means boolean index, NI means numeric index

            unique_objects_with_label = np.unique(data["meta"][label_rows_BI, 0].copy())
            np.random.shuffle(unique_objects_with_label)

            split_point = int(float(unique_objects_with_label.shape[0]) * ratio)

            [training_objects, validation_objects] = np.array_split(unique_objects_with_label, [split_point])

            label_polluted_training_object_row_BI = np.isin(data["meta"][:, 0], training_objects)
            training_object_row_BI = np.logical_and(label_polluted_training_object_row_BI, label_rows_BI)
            training_object_row_NI = np.where(training_object_row_BI)[0]
            training_row_index_blocks.append(training_object_row_NI)

            label_polluted_validation_object_row_BI = np.isin(data["meta"][:, 0], validation_objects)
            validation_object_row_BI = np.logical_and(label_polluted_validation_object_row_BI, label_rows_BI)
            validation_object_row_NI = np.where(validation_object_row_BI)[0]
            validation_row_index_blocks.append(validation_object_row_NI)

        training_row_indices = np.concatenate(training_row_index_blocks)
        validation_row_indices = np.concatenate(validation_row_index_blocks)

        return {
            "training": training_row_indices,
            "validation": validation_row_indices
        }

    def filter_data_on_row_ids(self, data, row_ids):

        filtered_data = {
            "meta": data["meta"][row_ids, :],
            "labels": data["labels"][row_ids],
            "examples": data["examples"][row_ids, :, :]
        }

        return filtered_data

    # image pre-processing logic
    def normalize_data_set(self, data_set):
        examples = data_set["examples"]
        labels = data_set["labels"]
        meta_data = data_set["meta"]

        mean_zero_examples = examples - np.mean(examples)
        # to get unit variance you must devide by std
        mean_zero_unit_variance_examples = mean_zero_examples / np.std(mean_zero_examples)

        return {
            "examples": mean_zero_unit_variance_examples,
            "labels": labels,
            "meta": meta_data
        }

    def represent_stereoscopy_as_meta_data(self, data_set):
        examples = data_set["examples"]
        labels = data_set["labels"]
        meta_data = data_set["meta"]

        dimensions = examples.shape
        stereo_count = dimensions[0]
        left_vector = np.zeros(shape=[stereo_count])
        right_vector = np.ones(shape=[stereo_count])

        examples = np.concatenate([examples[:, 0, :, :], examples[:, 0, :, :]])  # append left and
        labels = np.concatenate([labels, labels])

        stereo_column = np.concatenate([left_vector, right_vector]).reshape([2 * stereo_count, 1])
        meta_data = np.concatenate([meta_data, meta_data])  # copy all rows
        meta_data = np.concatenate([meta_data, stereo_column], axis=1)  # append stereo column

        return {
            "examples": examples,
            "labels": labels,
            "meta": meta_data
        }

    def downsample(self, data_set):

        examples = data_set["examples"]

        circumvent_tensor_proto_2GB_issue = tf.placeholder(dtype=tf.float32, shape=examples.shape)
        with tf.Session() as sess:
            downsampled_images = tf.image.resize_images(circumvent_tensor_proto_2GB_issue, size=[48, 48]).eval(
                feed_dict={
                    circumvent_tensor_proto_2GB_issue: examples
                },
                session=sess
            )

        return {
            "examples": downsampled_images,
            "labels": data_set["labels"],
            "meta": data_set["meta"]
        }

    def randomize_color_data(
            self,
            image_layer  # [batch_size, width, height, color_channels] WARNING; RESHAPE IS PROBABLY NECESSARY TO ADD CHANNELS
    ):

        image_layer = tf.image.random_brightness(image_layer, max_delta=32. / 255.)
        image_layer = tf.image.random_contrast(image_layer, lower=0.5, upper=1.5)

        return image_layer

    def crop_training_set(self, image_layer):
        (batch_size, width, height, channel_count) = image_layer.get_shape()
        cropped_image_layer = tf.random_crop(image_layer, size=[batch_size,32, 32, channel_count])

        return cropped_image_layer

    def crop_test_set(self, image_layer):
        cropped_image_layer = tf.image.crop_to_bounding_box(
            image_layer,
            offset_width=8,
            offset_height=8,
            target_width=32,
            target_height=32
        )
        return cropped_image_layer

    def default_preprocessed_test_input(self, image_layer):
        cropped = self.crop_test_set(image_layer)
        return cropped

    def default_preprocessed_training_input(self, image_layer):
        cropped_image_layer = self.crop_training_set(image_layer)
        randomized_colors = self.randomize_color_data(cropped_image_layer)

        return randomized_colors

    # presentation

    def mock_meta(self, batch_size):
        info_size = SmallNorb.INFO_SIZE_PER_TYPE.shape[0]

        mocked_meta = (np.random.random([batch_size, info_size]) * SmallNorb.INFO_SIZE_PER_TYPE).astype('int')

        assert (mocked_meta.max(axis=0) == (SmallNorb.INFO_SIZE_PER_TYPE - 1)).all()
        assert (mocked_meta.min(axis=0) == np.zeros(SmallNorb.INFO_SIZE_PER_TYPE.shape)).all()

        return mocked_meta

    def all_separate_property_animations(self, examples, meta_info):
        animatable_property_indexes = np.delete(np.arange(meta_info.shape[1]), 0)

        animations = []
        for property_index in animatable_property_indexes:
            animations.append(self.single_separate_property_animations(examples, meta_info, property_index))

        return np.concatenate(animations, axis=0)

    def single_separate_property_animations(self, examples, meta_info, frame_meta_index):

        all_indices = np.arange(len(meta_info.shape))
        animation_meta_indices = np.delete(all_indices, frame_meta_index)

        animation_index_ranges = SmallNorb.INFO_SIZE_PER_TYPE[animation_meta_indices]

        animation_id = self.hash_properties(meta_info[:, animation_meta_indices], animation_index_ranges)

        frame_index = meta_info[frame_meta_index]

        sequence_length = SmallNorb.INFO_SIZE_PER_TYPE[frame_meta_index]
        animation_count = np.product(animation_index_ranges)

        animation_and_sequence_to_frame = self.create_animation_sequences(examples, animation_count, sequence_length, animation_id, frame_index)

        return animation_and_sequence_to_frame

    def fractionally_tile(a, tile_count):
        integer_tile_count = int(tile_count)
        fractional_tile_count = tile_count - integer_tile_count

        integer_tiling = np.tile(a, [integer_tile_count])

        if fractional_tile_count == 0:
            return integer_tiling

        fractional_tile_element_count = int(a.shape[0] * fractional_tile_count)

        fractional_tile = a[:fractional_tile_element_count]

        complete_tiling = np.concatenate([integer_tiling, fractional_tile])

        return complete_tiling

    def create_reverse_N_ary_encoding(self, property_ranges, image_count):
        reverse_n_ary_encoding = np.array([], shape=[image_count, 0])  # note that n varies per digit!!

        repeat_count = 1

        for c in property_ranges:
            ascending_values = np.repeat(np.arange(c), repeat_count)
            descending_values = np.flip(ascending_values, axis=0)
            value_cycle = np.concatenate([ascending_values, descending_values])
            value_cycle_length = value_cycle.shape[0]
            value_cycle_tile_count = image_count / value_cycle_length
            next_column = self.fractionally_tile(value_cycle, value_cycle_tile_count)

            reverse_n_ary_encoding = np.concatenate([reverse_n_ary_encoding, next_column.reshape([-1, 1])])

            repeat_count *= ascending_values.shape[0]

        return reverse_n_ary_encoding

    def hash_properties(self, property_columns, property_ranges):
        digit_scale_per_property = np.cumprod(property_ranges).reshape([1, 3])

        sequence_id_to_hash_code = np.sum(property_columns * digit_scale_per_property, axis=1)
        return sequence_id_to_hash_code

    def concatened_property_animations(self, examples, labels, meta_info, animation_index):

        image_count = meta_info.shape[0]
        all_indices = np.arange(len(meta_info.shape))
        property_indices = np.delete(all_indices, animation_index)

        property_index_ranges = SmallNorb.INFO_SIZE_PER_TYPE[property_indices]
        animation_count = SmallNorb.INFO_SIZE_PER_TYPE[animation_index]

        reverse_n_ary_encoding = self.create_reverse_N_ary_encoding(property_index_ranges, image_count)

        # map gray encoding to hash, ie. sequence_id -> hash ( sequence id, the order, is the row number of the gray encoding)

        sequence_id_to_hash_code =  self.hash_properties(reverse_n_ary_encoding, property_index_ranges)

        hash_code_to_sequence_id = np.empty(sequence_id_to_hash_code.shape)
        hash_code_to_sequence_id[:] = float('nan')
        hash_code_to_sequence_id[sequence_id_to_hash_code] = np.arange(sequence_id_to_hash_code.shape[0])

        # map meta to hash, i.e meta -> hash

        meta_to_hash = self.hash_properties(meta_info[:, property_indices])

        # map meta to sequence_id, i.e. meta-> sequence, by combining meta-> hash and hash-> sequence_id

        meta_to_sequence_id = hash_code_to_sequence_id[meta_to_hash]

        # animation_to_sequence[meta.animation_id, meta.sequence_id] = example[:]
        animation_id = meta_info[animation_index]
        sequence_length = reverse_n_ary_encoding.shape[0]
        animation_and_sequence_to_frame = self.create_animation_sequences(examples, labels, animation_count, sequence_length, animation_id, meta_to_sequence_id)

        return animation_and_sequence_to_frame

    def create_animation_sequences(self, examples, labels, animation_count, sequence_length, animation_id, sequence_id):
        batch_size = examples.shape[0]
        width = examples.shape[1]
        height = examples.shape[2]

        animation_and_sequence_id_to_example = np.empty([animation_count, sequence_length])
        animation_and_sequence_id_to_example[:, :] == float('nan')

        animation_and_sequence_id_to_example[animation_id, sequence_id] = np.arange(examples.shape[0])

        animation_and_sequence_shape = animation_and_sequence_id_to_example.shape
        animation_and_sequence_to_frame_shape = np.concatenate([animation_and_sequence_shape, [width, height]])
        animation_and_sequence_to_frame = \
            examples[animation_and_sequence_id_to_example.reshape([-1]), :, :].reshape(animation_and_sequence_to_frame_shape)

        animation_and_sequence_to_label = labels[animation_and_sequence_id_to_example.reshape([-1])].reshape(animation_and_sequence_shape)

        return animation_and_sequence_to_frame, animation_and_sequence_to_label

    def convert_animation_sequences_to_datasets(self, animation_sequences, batch_size):
        # batches consist of a slice of frames for each animation

        animation_count = animation_sequences.shape[0]

        animation_increase_factor = int(float(batch_size) / float(animation_count))

        unconcatenated_animations = np.split(animation_sequences, animation_increase_factor)

        # this could cause a crash if animation_sequences cannot be split into equal batches
        new_animation_sequences = np.concatenate(unconcatenated_animations)

        shape_for_batching = np.concatenate([[-1], new_animation_sequences.shape[2:]])
        new_animation_sequences_sequenced_for_batching = new_animation_sequences.swapaxes(0, 1).reshape(shape_for_batching)
        ds = tf.data.Dataset().from_tensor_slices(new_animation_sequences_sequenced_for_batching).batch(batch_size)

        return ds

    def default_training_set(self):

        input_data = self.default_preprocessed_training_input(tf.constant(self.training["examples"]))
        output_data = tf.constant(self.training["labels"])

        return (input_data, output_data)

    def default_validation_set(self):

        input_data = self.default_preprocessed_test_input(tf.constant(self.validation["examples"]))
        output_data = tf.constant(self.validation["labels"])

        return (input_data, output_data)

    def default_test_set(self):

        input_data = self.default_preprocessed_test_input(tf.constant(self.test["examples"]))
        output_data = tf.constant(self.test["labels"])

        return (input_data, output_data)

    # data access
    def load_set(
            self,
            base_path
    ):
        image_file_path = base_path + 'dat.mat'
        label_file_path = base_path + 'cat.mat'
        info_file_path = base_path + 'info.mat'

        uint_example_tensor = MatFileReaderWriter.read_mat_file(image_file_path)
        uint_example_tensor = np.expand_dims(uint_example_tensor, len(uint_example_tensor.shape))

        int32_label_array = MatFileReaderWriter.read_mat_file(label_file_path)

        context_data_array = MatFileReaderWriter.read_mat_file(info_file_path)

        return {
            "examples": uint_example_tensor,
            "labels": int32_label_array,
            "meta": context_data_array
        }

    def training_example_count(self):
        return self.training["examples"].shape[0]

    def label_count(self):
        return np.unique(self.test["labels"]).shape[0]