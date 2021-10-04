import h5py
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

class read_hdf5:

    def __init__(self, base_path, is_split = True, number_subfiles = None,
                progress_bar = False):
        """
        Initiates object used to load data from a single or multiple HDF5 files.

        Parameters 
        ----------
        base_path : str or list of str
            Path to the file(s) to load the data from. If data is split into several 
            files, one can use a str with %d formatter or list containing paths to all 
            individual files.
        is_split : bool, optional
            Specifies whether the data is split across several files. Default value is
            True.
        number_subfiles : int, optional
            Number of files the data to load is split across. If None, it. 
            Defaults to None
        progress_bar : bool, optional
            Specifies whether a bar showing the progress of data loading is shown.
            Default value is False.
        
        Returns 
        ----------
        object
            Object allowing the user to load data from via its methods.
        """

        # Determine whether TQDM progress bar is displayed when loading.
        self.disable_progress_bar = not progress_bar

        # Number of available cores
        self.number_cores = mp.cpu_count()

        # If base_path is a list, use it to get the file_list and check how many files there are. 
        if isinstance(base_path, list):
            self.file_list       = base_path
            self.number_subfiles = len(self.file_list)

        # If base_path is a string, generate file_list based on whether it is split or not.  
        else:
            # If multiple files are to be loaded, generate the file_list with the number of 
            # entries equal to the number of subfiles specified by the user.
            if is_split:

                # Raise error if no subfiles have been specified
                if number_subfiles is None:
                    raise ValueError('No number of subfiles have been specifed even when \n' 
                                   + 'is supposedly split.')
                else:
                    self.number_subfiles = number_subfiles
            
            # If not split, then we have a single file by definition 
            else:
                self.number_subfiles = 1
            # Use list comprehension to generate file_list if possible (requires 
            # string formatting)
            if '%' in base_path:
                self.file_list = [base_path%i for i in range(self.number_subfiles)]
            # If not string formatting present in string (e.g. single file), then simply
            # add single subfile. 
            else:
                self.file_list = [base_path]
            
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
            if number_workers > self.number_cores:
                raise ValueError(f"Specified number of workers ({number_workers}) is larger \n" +\
                                 f"than currently available ({self.number_cores})")

        #===============================================================
        # Loading data in parallel
        #===============================================================
        
        # Create a worker pool
        with mp.Pool(number_workers) as pool:
            # Need to use starmap to employ multiple arguments in function 
            data_list = pool.starmap(self.get_data_single_subfile, 
                                     zip([dataset] * self.number_subfiles, self.file_list))

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
    
    def get_data_dimensions(self, dataset):
        '''Method goes over all (sub)files, collecting the number of entries
        associated to specified dataset, as well as the number of columns and 
        datatype. Returns and '''

        array_dimensions = None
        for i, file_name in enumerate(tqdm(self.file_list,disable=self.disable_progress_bar)):
            with h5py.File(file_name, 'r') as file:
                
                # Testing whether dataset exists in this subfile. If
                # not, skip.
                try: 
                    dims = file[dataset].shape
                except:
                    continue
                    
                # Just done for the first file
                if (array_dimensions == None):
                    array_dimensions = list(dims)
                    array_dtype      = file[dataset].dtype
                else:
                    array_dimensions[0] += dims[0]

        return (array_dimensions, array_dtype)

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
        # Get required dimensions of output array to create it 
        #===============================================================
        array_dimensions, array_dtype = self.get_data_dimensions(dataset)

        # Create the array to hold data
        out = np.ndarray(array_dimensions,dtype = array_dtype)
        
        #===============================================================
        # Load data sequentially
        #===============================================================
        
        # Cycle through opened files and add data in the positon they
        # should be located in
        offsets = [0,0]
        
        for file_name in tqdm(self.file_list, disable = self.disable_progress_bar):

            # Open file and get the dataset data
            with h5py.File(file_name, 'r') as file:

                # Testing whether dataset exists in this subfile. If
                # not, skip.
                try: 
                    file[dataset]
                except:
                    continue
                    
                # Update upper value of the offset
                offsets[1] = offsets[0] + file[dataset].shape[0]
               
                # Store data in the array to be returned
                file[dataset].read_direct(out[offsets[0]:offsets[1]])
                
                # Update lower value of the offset
                offsets[0] = offsets[1]  

        return out

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