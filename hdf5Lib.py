import h5py
import warnings
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

class read_hdf5:

    def __init__(self, base_path, is_split = True, number_subfiles = None,
                progress_bar = False):

        '''Creates object used to load data from an HDF5 file. This can handle single
        files or files that have been split up into several subfiles. base_path should
        contain an integer suffix %d to differentiate between subfiles (even if single
        subfile). This is used to generate list of paths to file(s).'''

        # Determine whether TQDM progress bar is displayed when loading.
        self.disable_progress_bar = not progress_bar

        # Number of available cores
        self.number_cores = mp.cpu_count()

        if is_split == True:

            # Check whether subfiles have been specified if there are supposed to be 
            # subfiles
            if number_subfiles is None:
                print('No number of subfiles have been specifed, despite \n',
                      'please do so now. Goodbye!')
            self.number_subfiles = number_subfiles
        else:
            # Special case of a single file.
            self.number_subfiles = 1

        # List of paths pointing to each file to load
        self.file_list = [base_path%i for i in range(self.number_subfiles)]
    
    def get_data_single_subfile(self, dataset, subfile_path):
        '''Method returns the specified dataset retrieved from a single
        subfile.'''

        #===============================================================
        # Open subfile and try loading the specified data 
        #===============================================================
        with h5py.File(subfile_path, 'r') as file:
            try: 
                return file[dataset][()]
            except:
                return None

    def get_data_parallel(self, dataset, number_workers = None):
        '''Load data from a specified dataset in parallel using several cores.
        The number of workers used for this defaults to the maximum available
        number of cores.'''

        #===============================================================
        # Setting number of workers used to load data
        #===============================================================

        # Check if requested number of workers does not exceed number
        # of cores available.
        if number_workers is not None:
            if number_workers > self.number_cores:
                raise ValueError('Specified number of workers (%d) is larger \n'%number_workers +\
                'than currently available (%d)'%self.number_cores)

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

        # Check if list is empty (i.e. it only contained  None entries). If so, it may
        # mean that the dataset name is incorrect or that this particular datafile doesn't
        # contain the information we are interested in.
        if not data_list:
            warnings.warn('No data was found for specified entry. This may be due to incorrect name or because the data is not available in the given subfiles.')
            return None

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

        '''Reads all of the data of the specified dataset, even if split
        into several subfiles.'''

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
        '''Prints the data entries available for the specified
        dataset, or if not, the base hdf5 tree. If file split over several
        subfiles it assumes first subfile contains the same attributes as 
        the rest.'''

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
        '''Prints the list of attributes available for the specified entry.'''

        with h5py.File(self.file_list[filenum],'r') as file:

            attribute_list = file[dataset].attrs.keys()

            print ('-----------------------------------')
            # Print
            for attribute in attribute_list:
                print (attribute)
            
            print ('-----------------------------------')
            print ()
            print ('Number of attributes: %d'%len(attribute_list))

        return 0

    def get_attribute(self, dataset, attribute, filenum=0):
        with h5py.File(self.file_list[filenum],'r') as file:
            attribute_value = file[dataset].attrs[attribute]
        return attribute_value