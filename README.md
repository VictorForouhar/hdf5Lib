hdf5Lib
=======
Small Python3 script I have written for personal use when loading data stored in multiple HDF5 subfiles, as is the case in many cosmological simulations I have encountered. It contains a single class [`read_hdf5`](https://github.com/VictorForouhar/hdf5Lib/blob/07ba32dcfc3eb546a94cfd6bb38628a3ea2a2262/hdf5Lib.py#L7) which creates an object from which one can access information stored in one or more HDF5 files. This allows for data to be loaded sequentially or in parallel (via [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module). Loading in parallel can drastically cut down on loading times, particularly when the files are not in cache. This is because most of the time is spent idling waiting for the disk.

Usage
-------
Check the example sections to access the test HDF5 files.

```python
import hdf5Lib
file = hdf5Lib.read_hdf5(path_to_file)
```
Once the object has been created, we can check what data entries are in the file and its attributes.
```python
file.print_entries()
# Prints all entries accesible at the top of the tree: dataset_a, dataset_b, dataset_c.

file.print_entries('dataset_a')
# Prints entries accesible in dataset_a:

file.print_attributes('dataset_a')
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


