import os
import sys
import glob
import time
import pickle
import sklearn



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# create a better style for matplotlib 
plt.style.use('seaborn-dark')
from sklearn.covariance import EmpiricalCovariance



from netflow.data_structures.dataset import NetflowDataset
from netflow.data_structures.util import parse_meta_file
from netflow.detectors.iaf import InverseAutoregressiveFlow



class NetflowExperiment(object):
    class Example(object):
        def __init__(self, attributes):
            """
            Internal class for examples (dataset, category) pairs.

            @param attributes: a line of the format dataset_filename category
            """
            # split the attributes
            attributes = attributes.split()
            # dataset is first 
            self.dataset = NetflowDataset(attributes[0])
            # category is second
            self.category = attributes[1]

    def __init__(self, filename, split_index, gpu = 4):
        """
        Constructor for the experiment class. This class serves as a wrapper
        for various file locations and attributes associated with a PCAP 
        network experiment.

        @param filename: location of the meta file that contains relevant info
        @param split_index: which train/test split to read
        @param gpu: the gpu to run the experiment on (default = 4)
        """
        # save the filename as an instance variable
        self.filename = filename

        # get the attributes from the meta file
        self.attributes = parse_meta_file(self.filename)

        # save relevant values as instance variables
        self.prefix = self.attributes['prefix'][0]
        # make sure the prefix matches the filename to avoid overwriting 
        # don't assert equality since absolute path can be included
        assert (self.filename.split('/')[-1].split('.')[0] == self.prefix)
        self.ID_examples = self.attributes['ID examples']
        # there needs to be in-distribution examples 
        assert (len(self.ID_examples))
        # there does not need to be out-of-distribution examples
        self.OOD_examples = self.attributes['OOD examples']
        self.network_architecture = self.attributes['network architecture'][0]
        self.target = self.attributes['target'][0]
        self.nepochs = int(self.attributes['nepochs'][0])

        # take a pretrained model from another network 
        if 'pretrained model' in self.attributes:
            self.pretrained_model = self.attributes['pretrained model'][0]

        # set the split index (which data split to read)
        self.split_index = split_index

        # create a temporary directory based on unique prefix 
        self.temp_directory = 'temp/experiments/{}/{:03d}'.format(self.prefix, self.split_index)
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory, exist_ok = True)

        # create a model directory based on unique prefix 
        self.model_directory = 'models/{}/{:03d}'.format(self.prefix, self.split_index)
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory, exist_ok = True)

        # create a results directory based on unique prefix 
        self.results_directory = 'results/{}/{:03d}'.format(self.prefix, self.split_index)
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory, exist_ok = True)

        # create a timing directory based on unique prefix 
        self.timing_directory = '{}/timing'.format(self.temp_directory)
        if not os.path.exists(self.timing_directory):
            os.makedirs(self.timing_directory, exist_ok = True)

        # create a figures directory based on unique prefix
        self.figures_directory = '{}/figures'.format(self.results_directory)
        if not os.path.exists(self.figures_directory):
            os.makedirs(self.figures_directory, exist_ok = True)
        
        # convert the ID_examples and OOD_examples into Example class
        for iv in range(len(self.ID_examples)):
            self.ID_examples[iv] = self.Example(self.ID_examples[iv])
        for iv in range(len(self.OOD_examples)):
            self.OOD_examples[iv] = self.Example(self.OOD_examples[iv])
                
        # create an empty dictionary mapping target values to output neurons
        self.target_mapping = {}
        # read the examples to get the list of possible outputs
        X_train = self.read_examples(self.ID_examples, 'train')
        # make sure the target occurs in the data 
        assert (self.target in X_train.columns)
        # deterministic list of unique values
        target_values = sorted(pd.unique(X_train[self.target]))
        # one output for every unique output
        self.output_shape = len(target_values)
        # populate the dictionary so that we can add new attributes for testing
        for index, target_value in enumerate(target_values):
            self.target_mapping[target_value] = index
        # set the gpu to use (if using gpus)
        self.gpu = gpu

    def read_target_mapping(self):
        """
        Read the target mapping from disk that maps targets to label values.
        """
        # read the target mapping every time (less efficient but simpler code)
        mapping_filename = '{}/target-mapping.pkl'.format(self.temp_directory)
        with open(mapping_filename, 'rb') as fd:
            target_mapping = pickle.load(fd)

        return target_mapping
    
    def read_examples(self, examples, split):
        """
        Read the examples into a pandas dataframe 

        @param examples: a list of examples to read by category
        @param split: which set to read ('train' or 'test') 
        """
        # must be either train or test 
        assert (split == 'train' or split == 'test')

        X_datasets = []
        
        # read the relevant training or testing datasets
        for example in examples:
            filename = '{}/train_test_splits/{:03d}/X_{}-{}.pkl.gz'.format(example.dataset.temp_directory, self.split_index, split, example.category)
            X_datasets.append(pd.read_pickle(filename))
        
        # concatenate the datasets
        X_datasets = pd.concat(X_datasets)
        
        # if the target_mapping variable is set, add to the datasets 
        if not len(self.target_mapping): return X_datasets

        # add new targets for any new OOD outputs 
        target_values = sorted(pd.unique(X_datasets[self.target]))
        for target_value in target_values:
            if not target_value in self.target_mapping:
                # add a new target mapping value 
                self.target_mapping[target_value] = len(self.target_mapping)

        X_datasets['target'] = X_datasets[self.target].map(self.target_mapping)
        
        # save the target mapping every time (less efficient but simpler code)
        mapping_filename = '{}/target-mapping.pkl'.format(self.temp_directory)
        with open(mapping_filename, 'wb') as fd:
            pickle.dump(self.target_mapping, fd)

        return X_datasets

    def no_unbalanced_chunks(self):
        """
        Return the number of chunks of unbalanced (Benign test) data generated previously.

        @param split_index: the split of this data to consider
        """
        directory = '{}/train_test_splits/{:03d}/unbalanced_data'.format(
            self.ID_examples[0].dataset.temp_directory, 
            self.split_index
        )

        return len(glob.glob('{}/*.pkl.gz'.format(directory)))

    def read_unbalanced_chunk(self, index):
        """
        Read the chunk of unbalanced data. 

        @param index: the chunk of data to read from disk.
        """
        directory = '{}/train_test_splits/{:03d}/unbalanced_data'.format(
            self.ID_examples[0].dataset.temp_directory, 
            self.split_index
        )
        input_filename = '{}/X_test-Benign_unbalanced-{:03d}.pkl.gz'.format(directory, index)

        X_chunk = pd.read_pickle(input_filename)
        return X_chunk

    def read_outputs(self, label, split = 'test', epoch = 'opt', unbalanced = False):
        """
        Read the outputs for the neural network.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        filename = '{}/{}_{}-outputs-{}.npy'.format(self.results_directory, label, split, epoch)
        assert (os.path.exists(filename))
        outputs = np.load(filename)

        if unbalanced:
            outputs = [outputs]

            input_directory = '{}/unbalanced_data'.format(self.results_directory)
            no_chunks = self.no_unbalanced_chunks()
            for index in range(no_chunks):
                filename = '{}/{}_{}-outputs_unbalanced-{:03d}-{}.npy'.format(input_directory, label, split, index, epoch)
                outputs.append(np.load(filename))

            outputs = np.concatenate(outputs)

        return outputs

    def read_predictions(self, label, split = 'test', epoch = 'opt', unbalanced = False):
        """
        Read the predictions for the neural network.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        filename = '{}/{}_{}-predictions-{}.npy'.format(self.results_directory, label, split, epoch)
        assert (os.path.exists(filename))
        predictions = np.load(filename)

        if unbalanced:
            predictions = [predictions]

            input_directory = '{}/unbalanced_data'.format(self.results_directory)
            no_chunks = self.no_unbalanced_chunks()
            for index in range(no_chunks):
                filename = '{}/{}_{}-predictions_unbalanced-{:03d}-{}.npy'.format(input_directory, label, split, index, epoch)
                predictions.append(np.load(filename))

            predictions = np.concatenate(predictions)

        return predictions

    def read_features(self, label, layer, split = 'test', epoch = 'opt', unbalanced = False):
        """
        Read the extracted features from an intermediate layer.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param layer: the extracted layer to read 
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        filename = '{}/{}_{}-features_{}-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        assert (os.path.exists(filename))
        features = np.load(filename)

        if unbalanced:
            features = [features]

            input_directory = '{}/unbalanced_data'.format(self.temp_directory)
            no_chunks = self.no_unbalanced_chunks()
            for index in range(no_chunks):
                filename = '{}/{}_{}-features_{}_unbalanced-{:03d}-{}.npy'.format(input_directory, label, split, layer, index, epoch)
                features.append(np.load(filename))

            features = np.concatenate(features)

        return features

    def read_mahalanobis_distances(self, label, layer, split = 'test', epoch = 'opt', unbalanced = False):
        """
        Read the mahalanobis distances from the extracted intermediate layer. 

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param layer: the extracted layer to read 
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        if unbalanced: filename = '{}/{}_{}-mahalanobis_{}_unbalanced-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        else: filename = '{}/{}_{}-mahalanobis_{}-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        assert (os.path.exists(filename))
        distances = np.load(filename)

        return distances

    def read_normalizing_flows(self, label, layer, split = 'test', epoch = 'opt', unbalanced = False):
        """
        Read the normalizing flows results from the extracted intermediate layer. 

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param layer: the extracted layer to read 
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        if unbalanced: filename = '{}/{}_{}-normalizing_flows_{}_unbalanced-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        else: filename = '{}/{}_{}-normalizing_flows_{}-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        assert (os.path.exists(filename))
        losses = np.load(filename)

        return losses

    def compile_model(self):
        """
        Placeholder function that must get overwritten.
        """
        pass

    def run_training(self):
        """
        Train the model for this experiment.
        """
        # make sure that this experiment is not using a pretrained model 
        assert (not hasattr(self, 'pretrained_model'))

        # construct the model 
        self.compile_model()
        
        # create the training dataset
        X_train = self.read_examples(self.ID_examples, 'train')
        # train the model with default number of epochs
        model_filename = '{}/model'.format(self.model_directory)
        self.model.train(X_train, nepochs = self.nepochs, model_filename = model_filename)

    def load_weights(self, epoch = 'opt'):
        """
        Load the weights from the trianed model. Throw assert if not available.
        
        @param epoch: the epoch to load from (default = 'opt')
        """
        # load the pretrained model weights if given 
        if hasattr(self, 'pretrained_model'):
            experiment = NetflowExperiment(self.pretrained_model, split_index = self.split_index, gpu = self.gpu)
            model_filename = '{}/model-{}'.format(experiment.model_directory, epoch)
        else:
            model_filename = '{}/model-{}'.format(self.model_directory, epoch)
        # load the model for inference
        assert (os.path.exists(model_filename))
        self.model.load_weights(model_filename)


    def infer(self, examples, label, split = 'test', fd = sys.stdout, epoch = 'opt', unbalanced = False):
        """
        Run inference on these examples and save the predictions to disk.

        @param examples: the list of example datasets
        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test') 
        @param fd: location to write results (default = stdout)
        @param epoch: the epoch to run inference on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        # skip if no examples
        if not len(examples): return

        # read the examples since it is required regardless on inference
        X = self.read_examples(examples, split)

        outputs_filename = '{}/{}_{}-outputs-{}.npy'.format(self.results_directory, label, split, epoch)
        predictions_filename = '{}/{}_{}-predictions-{}.npy'.format(self.results_directory, label, split, epoch)
        if not os.path.exists(predictions_filename):
            # run inference for these examples and split
            outputs = self.model.infer(X)
            # convert the outputs into a vector of predictions
            if self.output_shape > 1:
                predictions = np.argmax(outputs, axis = 1)
            else:
                predictions = np.squeeze(outputs) > 0.5
            
            np.save(outputs_filename, outputs)
            np.save(predictions_filename, predictions)

        if unbalanced:
            # create a new output directory 
            output_directory = '{}/unbalanced_data'.format(self.results_directory)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok = True)

            # break up X into small batches for inference
            no_chunks = self.no_unbalanced_chunks()
            for index in range(no_chunks):
                outputs_unbalanced_filename = '{}/{}_{}-outputs_unbalanced-{:03d}-{}.npy'.format(output_directory, label, split, index, epoch)
                predictions_unbalanced_filename = '{}/{}_{}-predictions_unbalanced-{:03d}-{}.npy'.format(output_directory, label, split, index, epoch)
                labels_unbalanced_filename = '{}/{}_{}-labels_unbalanced-{:03d}.npy'.format(output_directory, label, split, index)

                if os.path.exists(labels_unbalanced_filename): continue

                read_time = time.time()
                sys.stdout.write('Reading chunk {} of {}...'.format(index + 1, no_chunks))
                X_chunk = self.read_unbalanced_chunk(index)
                X_chunk['target'] = X_chunk[self.target].map(self.target_mapping)
                sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - read_time))

                infer_time = time.time()
                sys.stdout.write('Inferring on chunk {} of {}...'.format(index + 1, no_chunks))

                outputs = self.model.infer(X_chunk)
                # convert the outputs into a vector of predictions
                if self.output_shape > 1:
                    predictions = np.argmax(outputs, axis = 1)
                else:
                    predictions = np.squeeze(outputs) > 0.5

                # save the predictions and outputs, moved here so that testing
                # for predictions filename means inference completed successfully
                np.save(outputs_unbalanced_filename, outputs)
                np.save(predictions_unbalanced_filename, predictions)
                np.save(labels_unbalanced_filename, X_chunk['target'].values)

                sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - infer_time))

        # get the predictions and labels for the balanced dataset
        outputs = np.load(outputs_filename)
        predictions = np.load(predictions_filename)
        labels = X['target'].values
        
        if unbalanced:
            outputs = [outputs]
            predictions = [predictions]
            labels = [labels]
            for index in range(no_chunks):
                # read the outputs, predictions, and labels 
                outputs_filename = '{}/{}_{}-outputs_unbalanced-{:03d}-{}.npy'.format(output_directory, label, split, index, epoch)
                outputs.append(np.load(outputs_filename))

                predictions_filename = '{}/{}_{}-predictions_unbalanced-{:03d}-{}.npy'.format(output_directory, label, split, index, epoch)
                predictions.append(np.load(predictions_filename))
                
                labels_filename = '{}/{}_{}-labels_unbalanced-{:03d}.npy'.format(output_directory, label, split, index)
                labels.append(np.load(labels_filename))

            # concatenate into arrays for analysis 
            outputs = np.concatenate(outputs)
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)

        # compute the accuracy 
        accuracy = sum(predictions == labels) / labels.size

        fd.write('{} acc: {:0.8f}\n'.format(label, accuracy))

        # calculate advanced statistics for in-distribution examples 
        if label == 'ID':
            # for non-binary classification, weight f1-scores by number of samples 
            if self.output_shape == 1: average = 'binary'
            else: average = 'weighted'
            # calculate the f-1 score and write to disk 
            f1_score = sklearn.metrics.f1_score(labels, predictions, average = average)
            fd.write('{} f-1: {:0.8f}\n'.format(label, f1_score))

            # convert the labels into a (nsamples, nclasses) array for auc_score 
            nsamples = len(labels)
            if self.output_shape > 1:
                y_true = np.zeros((nsamples, self.output_shape), dtype=bool)
                y_true[np.arange(nsamples), labels] = 1
            else:
                y_true = labels
            # calculate the auc score and write to disk 
            auc_score = sklearn.metrics.roc_auc_score(y_true, outputs)
            fd.write('{} auc: {:0.8f}\n'.format(label, auc_score))

        # write the number of exmaples to disk
        fd.write('No. {} ex.: {}\n'.format(label, labels.size))

    def run_inference(self, epoch = 'opt', unbalanced = False):
        """
        Run all inference for this experiment. 

        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        # compile the model and load model weights 
        self.compile_model()
        self.load_weights(epoch)

        # open a file to write results 
        if unbalanced: results_filename = '{}/inference_unbalanced-{}.txt'.format(self.results_directory, epoch)
        else: results_filename = '{}/inference-{}.txt'.format(self.results_directory, epoch)
        fd = open(results_filename, 'w') 

        # get the results for the in-distribution examples
        self.infer(self.ID_examples, 'ID', 'test', fd, epoch, unbalanced)
        self.infer(self.OOD_examples, 'OOD', 'test', fd, epoch)
            
        # close the open file
        fd.close()

    def extract_features(self, examples, label, split, layer, epoch = 'opt', unbalanced = False):
        """
        Extract features from an intermediate layer.

        @param examples: the list of example datasets
        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test') 
        @param layer: the layer of the network to extract
        @param epoch: the epoch to run inference on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        # skip if no examples 
        if not len(examples): return

        feature_filename = '{}/{}_{}-features_{}-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        if not os.path.exists(feature_filename):
            # read the feature vector and extract the intermediate features 
            X = self.read_examples(examples, split)
            features = self.model.extract_features(X, layer)
            # save the features to the temp directory for later use
            np.save(feature_filename, features)

        if unbalanced:
            # create a new output directory 
            output_directory = '{}/unbalanced_data'.format(self.temp_directory)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok = True)

            # break up X into small batches for inference
            no_chunks = self.no_unbalanced_chunks()
            for index in range(no_chunks):
                # skip if this feature already was calculated
                feature_filename = '{}/{}_{}-features_{}_unbalanced-{:03d}-{}.npy'.format(output_directory, label, split, layer, index, epoch)
                if os.path.exists(feature_filename): continue

                read_time = time.time()
                sys.stdout.write('Reading chunk {} of {}...'.format(index + 1, no_chunks))
                X_chunk = self.read_unbalanced_chunk(index)
                X_chunk['target'] = X_chunk[self.target].map(self.target_mapping)
                sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - read_time))

                extract_time = time.time()
                sys.stdout.write('Extracting features on chunk {} of {}...'.format(index + 1, no_chunks))

                features = self.model.extract_features(X_chunk, layer)
                # save the features to the temp directory for later use
                np.save(feature_filename, features)

                sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - extract_time))

            # create a dummy file so we know to not run this code block again 
            output_filename = '{}/{}_{}-features_{}_unbalanced-{}.txt'.format(self.temp_directory, label, split, layer, epoch)
            with open(output_filename, 'w') as fd:
                fd.write('Completed.\n')

    def run_feature_extraction(self, layer, epoch = 'opt', unbalanced = False):
        """
        Run the intermediate algorithm that extracts features for training data, in-distribution
        and out-of-distribution testing data. Save features to disk for further analysis.
       
        @param layer: the layer of the network to extract
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        # compile the model and load model weights 
        self.compile_model()
        self.load_weights(epoch)

        # extract features from the intermediate layer 
        self.extract_features(self.ID_examples, 'ID', 'train', layer, epoch)
        self.extract_features(self.ID_examples, 'ID', 'test', layer, epoch, unbalanced)
        if len(self.OOD_examples):
            self.extract_features(self.OOD_examples, 'OOD', 'test', layer, epoch)

    def compute_robust_covariance_matrix(self, layer, epoch = 'opt'):
        """
        Compute the robust covariance matrix for each class (benign/attack).

        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')
        """
        # read the training dataset
        X_train = self.read_examples(self.ID_examples, 'train')
        
        # get the training features from disk
        features = self.read_features('ID', layer, 'train', epoch)
        
        # compute covariance for each label 
        categories = pd.unique(X_train['target'])
        for category in categories:
            # extract only the features for this label
            category_features = features[X_train['target'] == category]
                
            # compute robust convariance matrix
            robust_covariance = EmpiricalCovariance().fit(category_features)
            
            # compute max and min distance
            distance_from_covariance = robust_covariance.mahalanobis(category_features)
            max_distance = max(distance_from_covariance)
            min_distance = min(distance_from_covariance)

            # save the column that was used for computing this covariance matrix
            covariance_filename = '{}/covariance_{}-{}-{}.pickle'.format(self.temp_directory, layer, category, epoch)
            with open(covariance_filename, 'wb') as fd:
                # pickle the data using highest protocol available
                pickle.dump(robust_covariance, fd, pickle.HIGHEST_PROTOCOL)
                pickle.dump(min_distance, fd, pickle.HIGHEST_PROTOCOL)
                pickle.dump(max_distance, fd, pickle.HIGHEST_PROTOCOL)

    def mahalanobis_distance(self, predictions, features, layer, epoch = 'opt'):
        """
        Find the distance to the target labels for each feature.

        @param predictions: the label predicted for each feature 
        @param features: the features extracted from an intermediate layer
        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')
        """
        # read the training dataset to get the target labels
        X_train = self.read_examples(self.ID_examples, 'train')

        # read the covariance matricies 
        robust_covariances = {}
        max_distances = {}
        min_distances = {}
        for category in pd.unique(X_train['target']):
            covariance_filename = '{}/covariance_{}-{}-{}.pickle'.format(self.temp_directory, layer, category, epoch)
            with open(covariance_filename, 'rb') as fd:
                # read the pickled data
                robust_covariances[category] = pickle.load(fd)
                min_distances[category] = pickle.load(fd)
                max_distances[category] = pickle.load(fd)

        # quicker to batch compute the distances and throwaway unused computations
        # set distances values to infinity since we eventually will take the minimum for 
        # any categories that map to the same target value
        distances = np.inf * np.ones(features.shape[0])

        for category, X_train_by_target in X_train.groupby('target'):
            # get the distances for all examples 
            distances_from_category = (robust_covariances[category].mahalanobis(features) - min_distances[category]) / max_distances[category]

            # update the distances only for the predictions that match this category
            # np.minimum returns the elementwise min (distances starts at infinity)
            # we only update predictions that have this label (so if prediction is attack and this category is attack)
            distances[predictions == category] = np.minimum(distances[predictions == category], distances_from_category[predictions == category])
                
        return distances

    def compute_mahalanobis_distance(self, layer, epoch = 'opt', unbalanced = False):
        """
        Compute the Mahalanobis distance for the in-distribution and out-of-distribution
        test samples.

        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param unbalanced: include unbalanced benign data (default = False)
        """
        # only consider experiments with OOD examples 
        if not len(self.OOD_examples): return

        # start timing statistics 
        start_time = time.time()

        ID_features = self.read_features('ID', layer, 'test', epoch, unbalanced)
        OOD_features = self.read_features('OOD', layer, 'test', epoch)

        ID_predictions = self.read_predictions('ID', 'test', epoch, unbalanced)
        OOD_predictions = self.read_predictions('OOD', 'test', epoch)

        ID_mahalanobis = self.mahalanobis_distance(ID_predictions, ID_features, layer, epoch)
        OOD_mahalanobis = self.mahalanobis_distance(OOD_predictions, OOD_features, layer, epoch)

        # save the distances 
        if unbalanced:
            ID_filename = '{}/ID_test-mahalanobis_{}_unbalanced-{}.npy'.format(self.temp_directory, layer, epoch)
            OOD_filename = '{}/OOD_test-mahalanobis_{}_unbalanced-{}.npy'.format(self.temp_directory, layer, epoch)
        else:
            ID_filename = '{}/ID_test-mahalanobis_{}-{}.npy'.format(self.temp_directory, layer, epoch)
            OOD_filename = '{}/OOD_test-mahalanobis_{}-{}.npy'.format(self.temp_directory, layer, epoch)
        np.save(ID_filename, ID_mahalanobis)
        np.save(OOD_filename, OOD_mahalanobis)

        # we look at the bottom 95% of in-distribution examples 
        # these are considered in-distribution. anything in the bottom 
        # 5% is considered out-of-distribution (false positives)
        # we want to find the number of true positives, OOD inputs that 
        # have scores higher than this threshold
        true_negative_threshold = sorted(ID_mahalanobis)[19 * ID_mahalanobis.size // 20]
        
        # what proportion of OOD inputs have a score higher than the threshold 
        true_positives = sum(OOD_mahalanobis > true_negative_threshold)
        true_positive_rate = true_positives / OOD_mahalanobis.size
 
        # create an AUC ROC value for detecting OOD inputs
        mahalanobis_score = np.concatenate((ID_mahalanobis, OOD_mahalanobis))
        mahalanobis_truth = np.concatenate((np.zeros(ID_mahalanobis.size), np.ones(OOD_mahalanobis.size)))
        roc_auc_score = sklearn.metrics.roc_auc_score(mahalanobis_truth, mahalanobis_score)

        true_negative_threshold = sorted(ID_mahalanobis)[17 * ID_mahalanobis.size // 20]
        true_positives = sum(OOD_mahalanobis > true_negative_threshold)
        true_positive_rate_85 = true_positives / OOD_mahalanobis.size

        if unbalanced:
            timing_filename = '{}/mahalanobis_{}_unbalanced-{}.txt'.format(self.timing_directory, layer, epoch)
            output_filename = '{}/mahalanobis_{}_unbalanced-{}.txt'.format(self.results_directory, layer, epoch)
        else:
            timing_filename = '{}/mahalanobis_{}-{}.txt'.format(self.timing_directory, layer, epoch)
            output_filename = '{}/mahalanobis_{}-{}.txt'.format(self.results_directory, layer, epoch)
        
        # output the timing results
        with open(timing_filename, 'w') as fd:
            fd.write('No. ID Packets: {}\n'.format(ID_features.shape[0]))
            fd.write('No. OOD Packets: {}\n'.format(OOD_features.shape[0]))
            fd.write('Time: {:.04f} seconds\n'.format(time.time() - start_time))

        # output the results 
        with open(output_filename, 'w') as fd:
            fd.write('TPR (TNR = 95%): {:0.8f}%\n'.format(100 * true_positive_rate))
            fd.write('TPR (TNR = 85%): {:0.8f}%\n'.format(100 * true_positive_rate_85))
            fd.write('AUC: {:0.8f}\n'.format(roc_auc_score)) 

    def train_normalizing_flows(self, layer, stratify = 'packet_category', epoch = 'opt'):
        """
        Train a normalizing flow for each class (benign/attack).

        @param layer: the extracted layer to read 
        @param stratify: the column to stratify training and validation (default = 'packet_category')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        """
        # read the training dataset
        X_train = self.read_examples(self.ID_examples, 'train')
        
        # get the training features from disk
        features = self.read_features('ID', layer, 'train', epoch)

        # hardcoded legend for this batch of features
        legend = {
            0: 'Benign',
            1: 'Attack',
        }

        # compute covariance for each label 
        labels = pd.unique(X_train['target'])
        for label in labels:
            start_time = time.time()
            
            # extract only the features for this label
            label_features = features[X_train['target'] == label]
            label_categories = X_train.loc[X_train['target'] == label, stratify]

            # save the model to disk after training
            model_filename = '{}/normalizing_flows_{}-{}-{}'.format(self.model_directory, layer, label, epoch)
            loss_filename = '{}/normalizing_flows_{}-{}.png'.format(self.figures_directory, layer, epoch)

            # create a pandas dataframe for training 
            training_data = {
                'features': label_features,
                'stratify': label_categories,
            }

            # create and train normalizing flow model 
            flow = InverseAutoregressiveFlow(label_features.shape[1], gpu = self.gpu)
            flow.train(training_data, model_filename, loss_filename, legend[label])

            print ('Trained normalizing flow in {:0.2f} seconds for target {}.'.format(time.time() - start_time, label))
        
        # close the figures so that any future runs will not append to the same canvas 
        plt.close()

    def normalizing_flow(self, predictions, features, layer, epoch = 'opt'):
        """
        Load a normalizing flow model and calculate probabilities for this sample.

        @param predictions: the label predicted for each feature 
        @param features: the features extracted from an intermediate layer
        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')    
        """
        # read the training dataset to get the target labels
        X_train = self.read_examples(self.ID_examples, 'train')

        # create a zero vector for the log probailities 
        log_probabilities = np.zeros(features.shape[0], dtype = np.float32)

        # iterate over all input labels 
        for label in pd.unique(X_train['target']):
            # save the model to disk after training
            model_filename = '{}/normalizing_flows_{}-{}-{}'.format(self.model_directory, layer, label, epoch)

            # create and load model weights for normalizing flows
            flow = InverseAutoregressiveFlow(features.shape[1], gpu = self.gpu)
            flow.load_weights(model_filename)

            log_probabilities_from_label = flow.inverse(features)

            log_probabilities[predictions == label] = log_probabilities_from_label[predictions == label]

        return log_probabilities

    def calculate_normalizing_flows(self, layer, epoch = 'opt', unbalanced = False):
        """
        Run the inverse normalizing flow for all test features to find OOD inputs.

        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')
        """
        # only consider experiments with OOD examples 
        if not len(self.OOD_examples): return

        # start timing statistics 
        start_time = time.time()

        ID_features = self.read_features('ID', layer, 'test', epoch, unbalanced)
        OOD_features = self.read_features('OOD', layer, 'test', epoch)

        ID_predictions = self.read_predictions('ID', 'test', epoch, unbalanced)
        OOD_predictions = self.read_predictions('OOD', 'test', epoch)

        if unbalanced:
            ID_filename = '{}/ID_test-normalizing_flows_{}_unbalanced-{}.npy'.format(self.temp_directory, layer, epoch)
            OOD_filename = '{}/OOD_test-normalizing_flows_{}_unbalanced-{}.npy'.format(self.temp_directory, layer, epoch)
        else:
            ID_filename = '{}/ID_test-normalizing_flows_{}-{}.npy'.format(self.temp_directory, layer, epoch)
            OOD_filename = '{}/OOD_test-normalizing_flows_{}-{}.npy'.format(self.temp_directory, layer, epoch)    
        
        if not os.path.exists(OOD_filename):
            # get the losses for ID and OOD
            ID_losses = self.normalizing_flow(ID_predictions, ID_features, layer, epoch)
            OOD_losses = self.normalizing_flow(OOD_predictions, OOD_features, layer, epoch)

            # save the distances 
            np.save(ID_filename, ID_losses)   
            np.save(OOD_filename, OOD_losses)
        else:
            ID_losses = np.load(ID_filename)
            OOD_losses = np.load(OOD_filename)

        # we look at the bottom 95% of in-distribution examples 
        # these are considered in-distribution. anything in the bottom 
        # 5% is considered out-of-distribution (false positives)
        # we want to find the number of true positives, OOD inputs that 
        # have scores higher than this threshold. since low values are better 
        # under this metric, we take the top 5% as the cutoff. Lower values 
        # are better since we are returning the loss
        true_negative_threshold = sorted(ID_losses)[19 * ID_losses.size // 20]
        
        # what proportion of OOD inputs have a score lower than the threshold 
        # lower scores mean less likely to belong to the distribution
        true_positives = sum(OOD_losses > true_negative_threshold)
        true_positive_rate = true_positives / OOD_losses.size

        # create an AUC ROC value for detecting OOD inputs
        normalizing_flow_score = np.concatenate((ID_losses, OOD_losses))
        normalizing_flow_truth = np.concatenate((np.zeros(ID_losses.size), np.ones(OOD_losses.size)))
        roc_auc_score = sklearn.metrics.roc_auc_score(normalizing_flow_truth, normalizing_flow_score)

        true_negative_threshold = sorted(ID_losses)[17 * ID_losses.size // 20]
        true_positives = sum(OOD_losses > true_negative_threshold)
        true_positive_rate_85 = true_positives / OOD_losses.size

        if unbalanced:
            timing_filename = '{}/normalizing_flows_{}_unbalanced-{}.txt'.format(self.timing_directory, layer, epoch)
            output_filename = '{}/normalizing_flows_{}_unbalanced-{}.txt'.format(self.results_directory, layer, epoch)
        else:
            timing_filename = '{}/normalizing_flows_{}-{}.txt'.format(self.timing_directory, layer, epoch)
            output_filename = '{}/normalizing_flows_{}-{}.txt'.format(self.results_directory, layer, epoch)
            
        # output the timing results
        with open(timing_filename, 'w') as fd:
            fd.write('No. ID Packets: {}\n'.format(ID_features.shape[0]))
            fd.write('No. OOD Packets: {}\n'.format(OOD_features.shape[0]))
            fd.write('Time: {:.04f} seconds\n'.format(time.time() - start_time))

        # output the results 
        with open(output_filename, 'w') as fd:
            fd.write('TPR (TNR = 95%): {:0.8f}%\n'.format(100 * true_positive_rate))
            fd.write('TPR (TNR = 85%): {:0.8f}%\n'.format(100 * true_positive_rate_85))
            fd.write('AUC: {:0.8f}\n'.format(roc_auc_score))