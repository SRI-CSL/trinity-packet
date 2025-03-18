import os
import sys
import time



import numpy as np
import pandas as pd 



from torch.utils.data import Dataset



from sklearn.model_selection import train_test_split



from netflow.data_structures.dataset import NetflowDataset



# global constant denoting length of header information 
HEADER_LENGTH = 23



class ContextPCAPDataset(NetflowDataset):
    class PCAPFile(object):
        def __init__(self, attributes):
            """
            Internal class representation for PCAP files. 

            @param attributes: a line of the format filename PCAP_type
            """
            # split the attributes
            attributes = attributes.split()
            # filename is first 
            self.filename = attributes[0]
            # file type is second 
            self.pcap_type = attributes[1]
            # only allow PCAP and PCAPNG formats 
            assert (self.pcap_type == 'PCAP' or self.pcap_type == 'PCAPNG')

    def __init__(self, filename):
        """
        Constructor for the pcap with context dataset class. This class serves
        as a wrapper for various file locations and attributes associated with a 
        PCAP network dataset.

        @param filename: location of the meta file that contains relevant info
        """
        super(ContextPCAPDataset, self).__init__(filename)

        # save relevant values as instance variables 
        self.filename = filename
        self.prefix = self.attributes['prefix'][0]
        self.raw_datasets = self.attributes['raw datasets']
        self.processed_datasets = self.attributes['processed datasets']
        self.flow_datasets = self.attributes['flow datasets']
        self.flow_preprocessor = self.attributes['flow preprocessor'][0]

        # create a category mapping if it exists 
        if 'category mapping' in self.attributes:
            self.category_mapping = {}
            for category_mapping in self.attributes['category mapping']:
                category = category_mapping.split(':')[0]
                mapping = category_mapping.split(':')[1].strip()
                # set the mapping
                self.category_mapping[category] = mapping

        # convert the raw dataset into PCAP files 
        for iv in range(len(self.raw_datasets)):
            self.raw_datasets[iv] = self.PCAPFile(self.raw_datasets[iv])

    def read_processed_data(self):
        """
        Read the processed data from disk and return as a pandas dataframe.
        """
        dataset = pd.read_pickle(self.processed_datasets[0])
        for index in range(1, len(self.processed_datasets)):
            dataset = pd.concat([dataset, pd.read_pickle(self.processed_datasets[index])])

        return dataset
    
    def create_train_test_splits(self, test_size = 0.5, balanced = True, nsplits = 10):
        """
        Create a series of training and testing datasets with random splits. 

        @param test_size: the size of the testing dataset as a fraction (default = 0.50)
        @param balanced: balance the number of benign and attack samples (default = True)
        @param nsplits: the number of splits to run (default = 10)
        """
        # read the processed data
        read_time = time.time()
        data = self.read_processed_data()
        print ('Read processed data in {:0.2f} seconds.'.format(time.time() - read_time))

        # drop all rows with empty payloads here since they are both malicious/benign
        data = data[data['payload_length'] > 0]
        data.reset_index(drop = True, inplace = True)

        # create several splits of the training/testing data
        for split_index in range(nsplits):
            start_time = time.time()
            
            # save the complete training and testing datasets
            temp_directory = '{}/train_test_splits/{:03d}'.format(self.temp_directory, split_index)
            if not os.path.exists(temp_directory):
                os.makedirs(temp_directory, exist_ok = True)

            # divide the data into benign and attack 
            benign_samples = data[data['packet_label'] == 0]
            attack_samples = data[data['packet_label'] == 1]
            
            # resample the benign samples to match the same number as attack packets
            if balanced:
                no_attack_samples = len(attack_samples.index)
                selected_samples = benign_samples.sample(n = no_attack_samples, random_state = split_index)
                dropped_samples = benign_samples.drop(selected_samples.index)
            
                # save the dropped (benign) packets
                dropped_filename = '{}/X_test-Benign_unbalanced.pkl.gz'.format(temp_directory)
                dropped_samples.reset_index(drop = True, inplace = True)
                dropped_samples.to_pickle(dropped_filename)
                
                benign_samples = selected_samples
                
            sampled_data = pd.concat([benign_samples, attack_samples], axis = 0, ignore_index = True)

            # split the data into training and testing, stratifying by category
            # allow for all training or all testing 
            if test_size == 0:
                # create an empty data frame with correct columns 
                X_train, X_test = sampled_data, pd.DataFrame(columns = sampled_data.columns)
            elif test_size == 1:
                # create an empty data frame with correct columns 
                X_train, X_test = pd.DataFrame(columns = sampled_data.columns), sampled_data
            else:
                # use a custom random state to have deterministic results
                X_train, X_test = train_test_split(
                    sampled_data, 
                    test_size = test_size, 
                    stratify = sampled_data.packet_category, 
                    random_state = split_index,
                )

            X_train.reset_index(drop = True, inplace = True)
            X_test.reset_index(drop = True, inplace = True)

            # save splits for each category
            for category in pd.unique(data.packet_category):
                X_train_category = X_train[X_train['packet_category'] == category]
                X_test_category = X_test[X_test['packet_category'] == category]

                X_train_category.reset_index(drop = True, inplace = True)
                X_test_category.reset_index(drop = True, inplace = True)
                
                train_category_filename = '{}/X_train-{}.pkl.gz'.format(temp_directory, category)
                test_category_filename = '{}/X_test-{}.pkl.gz'.format(temp_directory, category)

                # don't necessarily need training and testing data 
                if len(X_train_category.index):
                    X_train_category.to_pickle(train_category_filename)
                if len(X_test_category.index):
                    X_test_category.to_pickle(test_category_filename)

            print ('  Created split {:03d} in {:0.2f} seconds.'.format(split_index, time.time() - start_time))

    def construct_ports_table(data, column_label):
        """
        Construct a table with the results of our "one-hot encoding". Note that
        it is not a true one-hot encoding since some values are duplicated. 
        
        @param data: the dataframe that contains port in and attack information
        @param column_label: the column to consider ('src_port' or dest_port')
        """
        # these are the possible ports that we consider
        # note that this is not a true one hot encoding since 
        # some of the categories overlap
        one_hot_encoding = {
            '21': [21],
            '22': [22],
            '80/8080': [80, 8080],
            '443/444': [443, 444],
            'Well Known (0 - 1023)': list(range(0, 1024)),
            'Registered (1024 - 49151)': list(range(1024, 49152)),
            'Dynamic/Private (49152 - 65535)': list(range(49152, 65536)),
        }
        
        # keep count of the number of benign and attack labels
        one_hot_encoding_counts = {}
        for label in one_hot_encoding.keys():
            one_hot_encoding_counts[label] = [0, 0]
        
        # go through every port in the dataset
        for port, data_by_port in data.groupby(column_label):
            ninstances = len(data_by_port.index)
            nattacks = data_by_port['packet_label'].sum()
            nbenigns = ninstances - nattacks
            
            # get the relevant one hot encoding label
            for label, ports in one_hot_encoding.items():
                if not port in ports: continue 
                
                one_hot_encoding_counts[label][0] += nbenigns 
                one_hot_encoding_counts[label][1] += nattacks

        for label, (nbenigns, nattacks) in one_hot_encoding_counts.items():
            print ('{} & {} & {} \\\\'.format(
                label, 
                nbenigns,
                nattacks,
            ))

    def port_analysis(self):
        """
        Compute analysis on the possible ports for header context. 
        """
        # read the processed data
        read_time = time.time()
        data = self.read_processed_data()
        print ('Read processed data in {:0.2f} seconds.'.format(time.time() - read_time))

        src_ports = data[['src_port', 'packet_label']]
        dest_ports = data[['dest_port', 'packet_label']]

        print ('\\multicolumn{3}{c}{\\textbf{Source Port}} \\\\ \hline')
        self.analyze_ports(src_ports, 'src_port')
        print ('\\multicolumn{3}{c}{\\textbf{Destination Port}} \\\\ \hline')
        self.analyze_ports(dest_ports, 'dest_port')

    def generate_unbalanced_chunks(self, nsplits = 10, nchunks = 100):
        """
        Generate many small chunks of the unbalanced (Benign heavy) data. 

        @param nsplits: the number of splits to run (default = 10)
        @param nchunks: the number of chunks to divide the data into (default = 100)
        """
        for split_index in range(nsplits):
            read_time = time.time()
            sys.stdout.write('Reading unbalanced data for split {:03d}...'.format(split_index))
            input_filename = '{}/train_test_splits/{:03d}/X_test-Benign_unbalanced.pkl.gz'.format(
                # read just one example
                self.temp_directory, 
                split_index, 
            )
            X = pd.read_pickle(input_filename)
            sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - read_time))

            output_directory = '{}/train_test_splits/{:03d}/unbalanced_data'.format(
                self.temp_directory,
                split_index,
            )
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok = True)

            write_time = time.time()
            sys.stdout.write('Writing unbalanced data for split {:03d}...'.format(split_index))
            for index, X_chunk in enumerate(np.array_split(X, nchunks)):
                output_filename = '{}/X_test-Benign_unbalanced-{:03d}.pkl.gz'.format(
                    output_directory, 
                    index,
                )
                X_chunk.to_pickle(output_filename)
            sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - write_time))

class ContextPCAPTorchDataset(Dataset):
    def __init__(
            self, 
            data, 
            header_length,
            payload_length, 
            output_shape, 
            to_fit,
            cnn,
            vocab,
        ):
        """
        Initialize the data generator for pytorch.

        @param data: pandas dataframe of pre-loaded data 
        @param header_length: the number of header attributes for context
        @param payload_length: the number of bytes to include for the payload
        @param output_shape: the number of layers in the output
        @param to_fit: boolean for training or testing
        @param cnn: is the network trained a cnn
        @param vocab: True if training a language model that uses a vocabulary
        """
        self.data = data 
        self.header_length = header_length
        self.payload_length = payload_length 
        self.output_shape = output_shape
        self.to_fit = to_fit
        self.cnn = cnn
        self.vocab = vocab

        # run data generation once for speed improvements 
        self.X, self.y = self.__data_generation__(self.data)

    def __len__(self):
        """
        Dataset required function that returns the number of samples 
        in the dataset.
        """
        return len(self.data.index)

    def payload_transform(self, x, dims):
        """
        # Function to transform the payload column: 
        # hex --> bytes --> int (0-255) --> expand or truncate to # dims 
        # --> normalize each feature (if not a vocabulary)
        """
        # Get bytes from hex   
        byte_array = bytes.fromhex(x[0])
        # Gets bytes as a list, which python interprets as an integer from 0-255
        byte_lst = list(byte_array)
        # Pad or truncate
        if (len(byte_lst) < dims):
            output = np.pad(byte_lst, (0, dims-len(byte_lst)), 'constant')
        else:
            output = np.array(byte_lst[0:dims].copy())
        # force the output to be of type float32
        output = output.astype(np.float32)
        
        # Normalize each feature to be 0 <= x <= 1
        # should not normalize between 0 and 1 if this is a transformer/LSTM architecture
        # TODO ADD VOCAB PARAM
        if not self.vocab:
            output = np.abs(output) / 255
        
        # return a torch tensor of the output 
        return output

    def convert_port_numbers(self, ports):
        """
        Internal function to convert the port numbers into a feature vector. 
        Not quite a one-hot encoding since duplicate categories are allowed. 

        @param ports: the source and destination port values.
        """
        # create the category encodings as lists to use np.isin
        categories = [
            [21],
            [22],
            [80, 8080],
            [443, 444],
            list(range(0, 1024)),
            list(range(1024, 49152)),
            list(range(49152, 65536)),
        ]
        encoded_ports = []
        for category in categories:
            encoded_ports.append(np.isin(ports, category))
        # concatenate the elements 
        return np.concatenate(encoded_ports, axis = 1)

    def __data_generation__(self, df_temp):
        """
        Internal function used by __init__ that takes a dataframe, performs 
        the payload_transform on the payload column, and returns the 
        transformed payload and label. Run only once (at startup)

        @param df_temp: a temporary data frame with only the relevant index
        """
        # convert the payload by padding if necessary
        X = df_temp['payload'].to_numpy().reshape((-1, 1))
        X = np.apply_along_axis(self.payload_transform, 1, X, self.payload_length)
        
        # current batch size (may be less than self.batch_size)
        batch_size = df_temp.shape[0]

        # add header context if applicable
        if self.header_length:
            #protocol = (df_temp[['transport_layer_protocol']].values == 'TCP')
            flags = df_temp[['cwr_flag', 'ece_flag', 'urg_flag', 'ack_flag', 'psh_flag', 'rst_flag', 'syn_flag', 'fin_flag']].values
            # normalize total length between 0 and 1 
            total_length = df_temp[['total_length']].values / 65535
            # create the one-hot encoding for the ports 
            src_ports = self.convert_port_numbers(df_temp[['src_port']].values)
            dest_ports = self.convert_port_numbers(df_temp[['dest_port']].values)
            # concatenate the usable header information together with the payload
            header = np.concatenate([flags, total_length, src_ports, dest_ports], axis = 1)
            # convert the header into float32 before concatenating with X 
            header = header.astype(np.float32)
            X = np.concatenate([header, X], axis = 1)
            # assert that the header is the expected length used throughout context PCAP module 
            assert (header.shape[1] == HEADER_LENGTH)

        # unsqueeze if cnn, must complete after adding context so that we can 
        # concatenate the arrays above with axis = 1
        if self.cnn:
            X = np.expand_dims(X, axis = 1)

        # if running inference, don't return y
        # this is needed for OOD inputs that do not correspond
        # to any node, otherwise the y value will try to access 
        # an array element that doesn't exist
        if not self.to_fit:
            # create a dummy variable for the labels (cannot just return None)
            return X, np.zeros((batch_size, self.output_shape), dtype = np.float32)

        # for binary classification return a 1-D array of labels
        if self.output_shape == 1:
            # expand the dimensions so the outputs are of shape (batch_size, 1)
            y = np.expand_dims(df_temp['target'].values, axis = 1)
        # create a one-hot encoding for multi-class classification 
        else:
            y = np.zeros((batch_size, self.output_shape), dtype=bool)
            y[np.arange(batch_size), df_temp['target']] = 1
        
        # convert the labels array to float32 for torch 
        return X, y.astype(np.float32)

    def __getitem__(self, idx):
        """
        Return a single item from the dataset. 

        @param idx: the index of the item to return
        """
        # return both the payload and label
        return self.X[idx,:], self.y[idx,:]
