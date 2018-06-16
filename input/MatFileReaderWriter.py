import numpy as np
import sys
import os
import os.path

class MatFileReaderWriter:

    MAGIC_TO_DATATYPE = {
        507333716: 'int32',
        507333717: 'uint8'
    }

    @staticmethod
    def read_mat_file(file_path):
        fid = open(file_path, 'rb')

        magic_data_type = MatFileReaderWriter._read_int(fid)
        rank = MatFileReaderWriter._read_int(fid)
        dimensions = MatFileReaderWriter._read_numpy_binary_array(fid, rank, 'int32')
        element_count = np.product(dimensions)
        dtype_string = MatFileReaderWriter.MAGIC_TO_DATATYPE[magic_data_type]

        np_raw_array = MatFileReaderWriter._read_numpy_binary_array(fid, element_count, dtype_string)
        np_nd_array = np_raw_array.reshape(dimensions)

        fid.close()

        return np_nd_array

    @staticmethod
    def _read_int(fid):
        return int.from_bytes(fid.read(4), byteorder='little')

    @staticmethod
    def _read_numpy_binary_array(fid, element_count, dtype_string):

        dt = np.dtype(dtype_string)
        dt = dt.newbyteorder('=')
        if sys.byteorder != 'little':
            print("WARNING DATA MIGHT BE INTERPRETED AS BIG ENDIAN, THIS IS NOT TESTED")
            dt = dt.newbyteorder('<')

        element_size = dt.itemsize

        raw_data = fid.read(element_count * element_size)

        np_array = np.fromstring(raw_data, dtype=dt)
        return np_array

    @staticmethod
    def data_type_to_magic(data_type):
        for k, v in MatFileReaderWriter.MAGIC_TO_DATATYPE.items():
            if v == data_type:
                return k

        raise Exception("cannot convert data_type to magic")

    @staticmethod
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def write_mat_file(file_path, numpy_array):
        MatFileReaderWriter.ensure_dir(file_path)
        fid = open(file_path, 'wb')

        magic_data_type_bytes = MatFileReaderWriter.data_type_to_magic(numpy_array.dtype.name).to_bytes(4, 'little')
        rank_bytes = len(numpy_array.shape).to_bytes(4, 'little')
        shape_bytes = np.array(numpy_array.shape, dtype='int32').tobytes()
        data_bytes = numpy_array.tobytes()

        fid.write(magic_data_type_bytes + rank_bytes + shape_bytes + data_bytes)

        fid.close()

