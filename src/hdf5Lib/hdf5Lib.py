import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

class Read:
    '''
    Initialises an object used to load data from single or multiple
    HDF5 files, with parallel capabilities in the latter case.

    Parameters 
    ----------
    base_path : str or list of str
        Path to the file(s) to load the data from. If data is split
        into several files, one can use a str with %d formatter or 
        list containing all paths to individual files.
    number_files : int, opt
        Number of files the data to load is split across. Defaults to None, 
        in which case the code tries to obtain it from the number of entries
        in base_path list, if possible.
    parallel : bool, opt
        Flag indicating whether data is loaded serially or in parallel. Only
        data split over multiple files is supported by parallel loading. 
        Defaults to True.
    progress_bar : bool, opt
        Whether a tqdm progress bar should be shown to indicate loading progress.
        Defaults to True.
    '''

    def __init__(self, base_path, number_files = None, parallel = True, 
                 progress_bar = True):

        #=====================================================
        # Determining number of files to read and their 
        # paths
        #=====================================================

        # If base_path is a list, use it as the file list and
        # get number of files from it. 
        if isinstance(base_path, list):
            self._file_list    = base_path
            self._number_files = len(self._file_list)

        # If base_path is a string, generate file_list based on 
        # whether it allows for string formatting or not.  
        else:
            if '%' in base_path:
                if number_files is not None: 
                    self._file_list = [base_path%i for i in range(number_files)]
                    self._number_files = number_files
                else: 
                    raise ValueError('Number of files not specified, despite using string ' \
                     + 'formated base_path.')
            else: 
                self._file_list    = [base_path] 
                self._number_files = 1 
        
        #=====================================================
        # Settings regarding paralell loading
        #=====================================================
        
        # Maximum number of available workers 
        self._max_number_workers = mp.cpu_count()
        
        # Only split files are compatible with parallel loading.
        self._parallel = False if (self._number_files == 1) else parallel

        #=====================================================
        # Determine whether tqdm progress bar is displayed.
        #=====================================================
        self._disable_progress_bar = not progress_bar
        
        #=====================================================
        # Settings regarding paralell loading
        #=====================================================

        # Dict holding data retrieved by read routines
        self._data = {} 
    
    #===============================================================
    # Methods to retrieve data
    #===============================================================

    def _get_data_single_subfile(self, dataset, subfile_path):
        '''
        Returns the specified dataset loaded from a single
        subfile.

        Parameters 
        ----------
        dataset: str
            Name of the dataset to retrieve
        subfile_path : str
            Path to the file we want to retrive the data from.

        Returns
        ----------
        np.ndarray or None
            Array containing all entries of the requested dataset.
            If no dataset of the same name exists in said file, it 
            returns None instead.
        '''

        #===============================================================
        # Open subfile and try loading the specified data 
        #===============================================================
        with h5py.File(subfile_path, 'r') as file:
            try: 
                return file[dataset][()]
            except:
                return None
    
    def _merge_data(self, data_list, dataset):
        '''
        Processes data loaded by the read routines. First
        removes empty entries and then concatenates all remaining
        ones to return a single np.ndarray

        Parameters
        ----------
        data_list : list
            List with each entry containing the np.ndarray with the
            numerical values of the associated file.
        dataset: str
            Name of the dataset we have just loaded.

        Returns
        -------
        np.ndarray
            All data merged into a single np.ndarray.  
        '''

        # Remove entries which contain no data 
        data_list = [entry for entry in data_list if entry is not None]

        # Check if list is empty (i.e. it only contained None entries). If so, it may
        # mean that the dataset name is incorrect or that this particular datafile doesn't
        # contain the information we are interested in.
        if not data_list:
            raise KeyError(f"No data was found for specified dataset ({dataset}) in any of the files.")
   
        # Merging list of arrays into a single array
        return np.concatenate(data_list)
    
    def _get_data_serial(self, dataset):
        '''
        Loads data from several HDF5 files serially.

        Parameters
        ----------
        dataset : str
            Name of the dataset

        Returns
        ----------
        np.ndarray
            Array containing all numerical values belonging to requested dataset.
        '''

        #===============================================================
        # Loading data sequentially
        #===============================================================
        
        # Iteratively go over each available file
        data_list = []
        for file_path in tqdm(self._file_list, disable=self._disable_progress_bar):
            data_list.append(self._get_data_single_subfile(dataset, file_path))
        
        # Process and merge list of np.ndarrays 
        return self._merge_data(data_list, dataset)

    def _get_data_parallel(self, dataset, number_workers = None):
        '''
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
        np.ndarray
            Array containing all numerical values belonging to requested dataset.
        '''

        #===============================================================
        # Setting number of workers used to load data
        #===============================================================

        # Check if requested number of workers does not exceed number
        # of cores available.
        if number_workers is None:
            number_workers = self._max_number_workers
        else:
            number_workers = min(number_workers, self._max_number_workers)

        #===============================================================
        # Loading data in parallel
        #===============================================================
        
        # Create a worker pool and use starmap to map multiple function
        # arguments
        with mp.Pool(number_workers) as pool:
            data_list = pool.starmap(self._get_data_single_subfile, 
                                     tqdm(zip([dataset] * self._number_files, self._file_list), 
                                          total=self._number_files, disable=self._disable_progress_bar))
        
        # Process and merge list of np.ndarrays
        return self._merge_data(data_list, dataset)
    
    def __getitem__(self, dataset):
        '''
        Returns data belonging to specified dataset. First
        it checks if it has been loaded previously, and if not,
        it loads it.

        Parameters
        -----------
        dataset : str
            Name of the dataset to load from the hdf5 file(s).
        
        Returns
        -----------
        np.ndarray
            Array containing all numerical values belonging to requested dataset.
        '''

        # Check if it has been loaded before. If not, load.
        if dataset not in self._data:
            self._data[dataset] = self._get_data_parallel(dataset) if self._parallel else self._get_data_serial(dataset)
            self._data[dataset].flags.writeable = False # Make it read only
        
        return self._data[dataset]

    #===============================================================
    # Enter and exit methods 
    #===============================================================
    def __enter__(self):
        return self
    def __exit__(self,*args, **kwargs):
        del self

    #===============================================================
    # Helper functions to inspect data structure
    #===============================================================

    def print_entries(self, dataset = None, filenum = 0):
        '''
        Prints available datasets/groups in the specified file. 

        Parameters
        -----------
        dataset : str, opt
            The name of the dataset we are interested in obtaining the 
            data entries from. If None, it returns the groups of the file.
        filenum : int, opt
            Selects which file of the list. Defaults to the first one. 
        '''

        with h5py.File(self._file_list[filenum],'r') as file:

            # Handle case where no specific field is entered
            key_list = file.keys() if dataset is None else file[dataset].keys()

            # Sort in alphabetical order
            key_list = sorted(key_list)

            print ('-'*40)
            for key in key_list: print (key)
            print ('-'*40)
            print ()
            print ('Number of entries: %d'%len(key_list))

    def print_attributes(self, dataset, filenum = 0):
        '''
        Prints a list of attributes available for the specified dataset/group. 

        Parameters
        -----------
        dataset : str
            The name of the group/dataset we are interested in obtaining the 
            attribute list from. 
        filenum : int, opt
            Selects which file of the list is used to get the attributes from. 
            Defaults to the first one. 
        '''

        with h5py.File(self._file_list[filenum],'r') as file:

            attribute_list = file[dataset].attrs.keys()

            print ('-'*40)
            for attribute in attribute_list: print (attribute)            
            print ('-'*40)
            print ()
            print ('Number of attributes: %d'%len(attribute_list))

    def get_attribute(self, dataset, attribute, filenum=0):
        '''
        Returns the value of the specified attribute in the given dataset.

        Parameters
        -----------
        dataset : str
            Dataset we are interest in obtaining the attribute from
        attribute : str
            Name of the attribute we want to get 
        filenum : int, opt
            Selects which file of the list is used to get the attributes 
            from. Defaults to the first one. 

        Returns
        -----------
        float or int
            The attribute value. 
        '''
        
        with h5py.File(self._file_list[filenum],'r') as file:
            return file[dataset].attrs[attribute]