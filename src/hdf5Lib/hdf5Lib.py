import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

class read_hdf5:
    # TODO: consider adding a flag indicating whether parallel or serial 
    # loading is used in __getitem__ method.
    def __init__(self, base_path, number_files = None, parallel = True):
        """
        Initiates object used to load data from a single or multiple HDF5 files.

        Parameters 
        ----------
        base_path : str or list of str
            Path to the file(s) to load the data from. If data is split into several 
            files, one can use a str with %d formatter or list containing paths to all 
            individual files.
        number_files : int, optional
            Number of files the data to load is split across. Defaults to None
        parallel : Bool
            Flag indicating whether data is loaded serially or in parallel. Only
            data split over multiple files is supported by parallel loading.
        
        Returns 
        ----------
        object
            Object allowing the user to load data from via its methods.
        """

        #=====================================================
        # Determining number of files to read
        #=====================================================

        # If base_path is a list, use it directly to define self.file_list and
        # get number of files directly from it. 
        if isinstance(base_path, list):
            self.file_list    = base_path
            self.number_files = len(self.file_list)

        # If base_path is a string, generate file_list based on whether it allows for
        # string formatting or not.  
        else:
            if '%' in base_path:
                try: 
                    self.file_list = [base_path%i for i in range(number_files)]
                    self.number_files = number_files
                except:
                    raise ValueError('Number of files not specified, despite using string ' \
                     + 'formated base_path.')
            else: 
                self.file_list    = [base_path] 
                self.number_files = 1 
        
        #=====================================================
        # Settings regarding paralell loading
        #=====================================================
        
        # Number of available cores
        self._number_cores = mp.cpu_count()
        
        # Currently only split files are compatible with parallel 
        # loading. In future I might add support for single files
        # (e.g. via array slices)
        if self.number_files == 1: self._parallel = False
        else: self._parallel = parallel
        

        # Dict holding data retrieved by groups
        self.data = {} 

    def get_data_single_subfile(self, dataset, subfile_path):
        """
        Returns the specified dataset retrieved from a single
        subfile.

        Parameters 
        ----------
        dataset: str
            Name of the dataset to retrieve
        subfile_path : str
            Path to the file we want to retrive the data from.

        Returns
        ----------
        ndarray:
            Array containing all entries of the requested dataset.
        """

        #===============================================================
        # Open subfile and try loading the specified data 
        #===============================================================
        with h5py.File(subfile_path, 'r') as file:
            try: 
                return file[dataset][()]
            except:
                return None

    def get_data_parallel(self, dataset, number_workers = None):
        
        """
        Loads data from several HDF5 files in parallel. 

        Parameters
        ----------
        dataset : str
            Name of the dataset to get the data from.
        number_workers : int
            Number of processes used in parallel to load the data. If none are
            specified, it defaults to the maximum available number in current
            process. Note this can only be used for data that is split across
            multiple HDF5 files.

        Returns
        ----------
        ArrayType
            Array containing all entries of the requested dataset.
        """

        #===============================================================
        # Setting number of workers used to load data
        #===============================================================

        # Check if requested number of workers does not exceed number
        # of cores available.
        if number_workers is not None:
            if number_workers > self._number_cores:
                raise ValueError(f"Specified number of workers ({number_workers}) is larger \n" +\
                                 f"than currently available ({self._number_cores})")

        #===============================================================
        # Loading data in parallel
        #===============================================================
        
        # Create a worker pool
        with mp.Pool(number_workers) as pool:
            # Need to use starmap to employ multiple arguments in function 
            data_list = pool.starmap(self.get_data_single_subfile, 
                                     zip([dataset] * self.number_files, self.file_list))

        #===============================================================
        # Removing empty entries
        #===============================================================
        
        # Remove subfile entries which contain no data 
        data_list = [subfile_entry for subfile_entry in data_list if subfile_entry is not None]

        # Check if list is empty (i.e. it only contained None entries). If so, it may
        # mean that the dataset name is incorrect or that this particular datafile doesn't
        # contain the information we are interested in.
        if not data_list:
            raise KeyError(f"No data was found for specified dataset ({dataset}) in any of the files.")

        #===============================================================
        # Merging list of arrays into a single array
        #===============================================================
        data_array = np.concatenate(data_list)

        return data_array 

    def get_data(self, dataset):

        """
        Returns an array containing the specified data entries.

        Parameters
        ----------
        dataset : str
            Name of the dataset

        Returns
        ----------
        array_type
            An array holding all the data. This includes data that has been
            split up into several files.

        """

        #===============================================================
        # Loading data sequentially
        #===============================================================
        
        # Iteratively go over each available file
        data_list = []
        for file_path in self.file_list:
            data_list.append(self.get_data_single_subfile(dataset, file_path))

        #===============================================================
        # Removing empty entries
        #===============================================================
        
        # Remove subfile entries which contain no data 
        data_list = [subfile_entry for subfile_entry in data_list if subfile_entry is not None]

        # Check if list is empty (i.e. it only contained None entries). If so, it may
        # mean that the dataset name is incorrect or that this particular datafile doesn't
        # contain the information we are interested in.
        if not data_list:
            raise KeyError(f"No data was found for specified dataset ({dataset}) in any of the files.")

        #===============================================================
        # Merging list of arrays into a single array
        #===============================================================
        data_array = np.concatenate(data_list)

        return data_array


    def load(self, group):
        if group not in self.data:
            # TODO: I can add self.parallel_mode = 1 to 
            # determine whether get_data or get_data_parallel is used.
            # For now default to parallel.

            self.data[group] = self.get_data_parallel(group)
        return self.data[group]
    
    def __getitem__(self,group):
        return self.load(group)
    
    def print_entries(self, dataset = None, filenum = 0):
        """
        Prints available datasets/groups in the specified file. 

        Parameters
        -----------
        dataset : str, opt
            The name of the group we are interested in obtaining the 
            data entries from. If None, it returns the groups of the file.
        filenum : int, opt
            Selects which file of the list. Defaults to the first one. 

        Returns
        -----------
        int
            On sucessful execution, returns 0.
        """

        with h5py.File(self.file_list[filenum],'r') as file:

            # Handle case where no specific field is entered
            if dataset is None:
                key_list = file.keys()
            else:
                key_list = file[dataset].keys()
            
            # Sort in alphabetical order
            key_list = sorted(key_list)

            # Print
            print ('-----------------------------------')
            for key in key_list:
                print (key)

            print ('-----------------------------------')
            print ()
            print ('Number of entries: %d'%len(key_list))

        return 0

    def print_attributes(self, dataset, filenum = 0):
        """
        Prints a list of attributes available for the specified dataset/group. 

        Parameters
        -----------
        dataset : str
            The name of the group/dataset we are interested in obtaining the 
            attribute list from. 
        filenum : int, opt
            Selects which file of the list is used to get the attributes from. Defaults to the first one. 

        Returns
        -----------
        int
            Returns 0 on sucessful execution.
        """

        with h5py.File(self.file_list[filenum],'r') as file:

            attribute_list = file[dataset].attrs.keys()

            print ('-'*40)
            # Print
            for attribute in attribute_list:
                print (attribute)
            
            print ('-'*40)
            print ()
            print ('Number of attributes: %d'%len(attribute_list))

        return 0

    def get_attribute(self, dataset, attribute, filenum=0):
        """
        Returns the value of the specified attribute in the given dataset.

        Parameters
        -----------
        dataset : str
            Dataset/Group we are interest in obtaining the attribute from
        attribute : str
            Name of the attribute we want to get 
        filenum : int, optional
            Number of the file we use to obtain the information from

        Returns
        -----------
        AttributeType
            The attribute value/object. 
        """
        
        with h5py.File(self.file_list[filenum],'r') as file:
            attribute_value = file[dataset].attrs[attribute]
        return attribute_value