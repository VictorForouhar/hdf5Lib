import pytest
from hdf5Lib import hdf5Lib

#=======================================================================
# Testing generation of self.file_list
#=======================================================================
def test_single_file_string():
    hdf5_file = hdf5Lib.read_hdf5('../examples/single_file/file.hdf5',is_split=False)
    assert hdf5_file.file_list == ['../examples/single_file/file.hdf5']


def test_formatted_file_string():
    file_list = [f"../examples/split_files/subfile_{x:02d}.hdf5" for x in range(50)]
    hdf5_file = hdf5Lib.read_hdf5('../examples/split_files/subfile_%.2d.hdf5',is_split=True,number_subfiles=50)
    assert hdf5_file.file_list == file_list

def test_list_of_files_strings():
    file_list = [f"../examples/split_files/subfile_{x:02d}.hdf5" for x in range(50)]
    hdf5_file = hdf5Lib.read_hdf5(file_list,is_split=True)
    assert hdf5_file.file_list == file_list

#=======================================================================
# Testing whether correct number of subfiles are being calculated
#=======================================================================

def test_get_number_subfiles_single_file():
    hdf5_file = hdf5Lib.read_hdf5('../examples/single_file/file.hdf5',is_split=False)
    assert hdf5_file.number_subfiles == 1

def test_get_number_subfiles_formatted_file_string():
    file_list = [f"../examples/split_files/subfile_{x:02d}.hdf5" for x in range(50)]
    hdf5_file = hdf5Lib.read_hdf5('../examples/split_files/subfile_%.2d.hdf5',is_split=True,number_subfiles=50)
    assert hdf5_file.number_subfiles == 50

def test_get_number_subfiles_list_of_files_strings():
    file_list = [f"../examples/split_files/subfile_{x:02d}.hdf5" for x in range(50)]
    hdf5_file = hdf5Lib.read_hdf5(file_list,is_split=True)
    assert hdf5_file.number_subfiles == 50

def test_unspecified_number_subfiles():
    with pytest.raises(ValueError):
        hdf5Lib.read_hdf5('../examples/split_files/subfile_%.2d.hdf5',is_split=True)
    

