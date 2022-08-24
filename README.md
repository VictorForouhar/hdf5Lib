hdf5Lib
=======
Small Python3 script I have written for personal use when loading data stored in multiple HDF5 subfiles, as is the case in many cosmological simulations I have encountered. It contains a single class [`Read`](https://github.com/VictorForouhar/hdf5Lib/blob/07ba32dcfc3eb546a94cfd6bb38628a3ea2a2262/hdf5Lib.py#L6) which creates an object from which one can access information stored in one or more HDF5 files. This allows for data to be loaded sequentially or in parallel (via [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module). Loading in parallel can drastically cut down on loading times, particularly when the files are not in cache.

Requirements
=======
This package requires the following modules:[`h5py`](https://docs.h5py.org/en/stable/)(3.1.0), [`numpy`](https://numpy.org/doc/stable/) (1.19.0), [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) and [`tqdm`](https://tqdm.github.io/) (4.62.0). This code has only been tested using those versions.


Usage
-------
Check the ./examples/ folder to access the test HDF5 files. An interactive, jupyter-notebook version of the tutorial is available in the same location.

```python
import hdf5Lib
path_to_file = './examples/single_file/file.hdf5'
file = hdf5Lib.Read(path_to_file)
```
Once the object has been created, we can check what data entries are in the file and its attributes.
```python

# Prints all entries accesible at the top of the tree.
file.print_entries()

# Prints entries accesible in dataset_a.
file.print_entries('dataset_a')

# Prints attributes of dataset_a.
file.print_attributes('dataset_a')
```

If we want to get the value of an attribute, we simply specify the dataset and associated attribute we are interested in retrieving.
```python
pi = file.get_attribute('dataset_a', 'pi')
h  = file.get_attribute('dataset_a/subdataset_1','hubbleParam')
```

The code is able to handle cases where data has been split across many different 
(sub)files. In such cases, file paths can be specified in two different ways:
```python

# List with each entry being the path to each individual file 
path_to_files = ['./examples/split_files/subfile_%.2d.hdf5'%i for i in range(50)]
file = hdf5Lib.Read(path_to_files)

# Alternatively, provide a string-formatted path and the number of files the 
# data has been split across (internally, it does the above)
path_to_files = './examples/split_files/subfile_%.2d.hdf5'
number_files  = 50
file = hdf5Lib.Read(path_to_files, number_files = number_files)
```

Loading data can be done serially or in parallel. The code handles cases where data 
is split across many different files.
```python

path_to_files = ['./examples/split_files/subfile_%.2d.hdf5'%i for i in range(50)]

# Reads sequentially using a single processor
file_serial_mode = hdf5Lib.Read(path_to_files, parallel=False)
data = file_serial_mode['dataset_a']

# Reads in parallel using as many processes as possible.
file_parallel_mode = hdf5Lib.Read(path_to_files, parallel=True)
data = file_parallel_mode['dataset_a']
```


