import pytest
from hdf5Lib import hdf5Lib

# @pytest.fixture(

# )

def test_single_file_string():
    hdf5_file = hdf5Lib.read_hdf5('../examples/single_file.file_hdf5',is_split=False)
    assert hdf5_file.file_list == ['../examples/single_file/file_hdf5']

def test_formatted_file_string():
    file_list = [f"../examples/split_files/subfile_{x:02d}.hdf5" for x in range(50)]
    hdf5_file = hdf5Lib.read_hdf5('../examples/split_files/subfile_%.2d.hdf5',is_split=True,number_subfiles=50)
    assert hdf5_file.file_list == file_list

def test_list_of_files_strings():
    file_list = [f"../examples/split_files/subfile_{x:02d}.hdf5" for x in range(50)]
    hdf5_file = hdf5Lib.read_hdf5(file_list,is_split=True)
    assert hdf5_file.file_list == file_list