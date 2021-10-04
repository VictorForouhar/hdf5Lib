hdf5Lib
=======
Small Python3 script I have written for personal use when loading data stored in multiple HDF5 subfiles, as is the case in many cosmological simulations I have encountered. It contains a single class [`read_hdf5`](https://github.com/VictorForouhar/hdf5Lib/blob/07ba32dcfc3eb546a94cfd6bb38628a3ea2a2262/hdf5Lib.py#L7) which creates an object from which one can access information stored in one or more HDF5 files. This allows for data to be loaded sequentially or in parallel (via [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module). Loading in parallel can drastically cut down on loading times, particularly when the files are not in cache.

Requirements
=======
This package requires the following modules:[`h5py`](https://docs.h5py.org/en/stable/)(2.10.0), [`numpy`](https://numpy.org/doc/stable/) (1.21.1), [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) and [`tqdm`](https://tqdm.github.io/) (4.62.0). This code has only been tested using those versions.


Usage
-------
Check the ./examples/ folder to access the test HDF5 files. An interactive, jupyter-notebook version of the tutorial is available in the same location.

```python
import hdf5Lib
path_to_file = './examples/single_file/file.hdf5'
file = hdf5Lib.read_hdf5(path_to_file)
```
Once the object has been created, we can check what data entries are in the file and its attributes.
```python
file.print_entries()
# Prints all entries accesible at the top of the tree: dataset_a, dataset_b, dataset_c.

file.print_entries('dataset_a')
# Prints entries accesible in dataset_a:

file.print_attributes('dataset_a')
# Prints attributes of dataset_a
```

If we want to get the value of an attribute, we simply specify the dataset and associated attribute we are interested in retrieving.
```python
pi = file.get_attribute('dataset_a', 'pi')
h  = file.get_attribute('dataset_a/subdataset_1','hubbleParam')
```

Finally, loading data is as simple as specifying the dataset we want to load.
```python

# Reads sequentially using a single processor
data = file.get_data(dataset_a)

# Reads in parallel using specified number of processors (if blank, it defaults to maximum 
# available)
data = file.get_data_parallel(dataset_a, number_processors_to_use)
```


