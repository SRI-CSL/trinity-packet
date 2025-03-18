import os
import sys
import glob
import time
import pickle
import sklearn



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from pytorch_ood.detector import EnergyBased, ODIN


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

    def read_unbalanced_chunk(self, label, index):
        """
        Read the chunk of unbalanced data. 

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param index: the chunk of data to read from disk.
        """
        if label == 'ID': temp_directory = self.ID_examples[0].dataset.temp_directory
        elif label == 'OOD': temp_directory = self.OOD_examples[0].dataset.temp_directory
        else: assert ('Unkown label: {}'.format(label))

        directory = '{}/train_test_splits/{:03d}/unbalanced_data'.format(
            temp_directory,
            self.split_index
        )
        input_filename = '{}/X_test-Benign_unbalanced-{:03d}.pkl.gz'.format(directory, index)

        X_chunk = pd.read_pickle(input_filename)
        
        return X_chunk

    def no_unbalanced_chunks(self, label):
        """
        Return the number of chunks of unbalanced (Benign test) data generated previously.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        """

        if label == 'ID': examples = self.ID_examples
        elif label == 'OOD': examples = self.OOD_examples
        else: assert ('Unkown label: {}'.format(label))

        temp_directory = examples[0].dataset.temp_directory

        # make sure benign data is present in the OOD examples
        benign = False 
        for example in examples:
            if example.category == 'Benign': benign = True

        # if there is no benign data, don't run on any unbalanced chunks
        if not benign: return 0

        directory = '{}/train_test_splits/{:03d}/unbalanced_data'.format(
            temp_directory,
            self.split_index
        )

        return len(glob.glob('{}/*.pkl.gz'.format(directory)))

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

    def read_target_mapping(self):
        """
        Read the target mapping from disk that maps targets to label values.
        """
        # read the target mapping every time (less efficient but simpler code)
        mapping_filename = '{}/target-mapping.pkl'.format(self.temp_directory)
        with open(mapping_filename, 'rb') as fd:
            target_mapping = pickle.load(fd)

        return target_mapping

    def infer(self, examples, label, split = 'test', fd = sys.stdout, epoch = 'opt', balanced = False):
        """
        Run inference on these examples and save the predictions to disk.

        @param examples: the list of example datasets
        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test') 
        @param fd: location to write results (default = stdout)
        @param epoch: the epoch to run inference on (default = 'opt')
        @param balanced: only use balanced benign data (default = False)
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

        if not balanced:
            # create a new output directory 
            output_directory = '{}/unbalanced_data'.format(self.results_directory)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok = True)

            # break up X into small batches for inference
            no_chunks = self.no_unbalanced_chunks(label)
            for index in range(no_chunks):
                outputs_unbalanced_filename = '{}/{}_{}-outputs_unbalanced-{:03d}-{}.npy'.format(output_directory, label, split, index, epoch)
                predictions_unbalanced_filename = '{}/{}_{}-predictions_unbalanced-{:03d}-{}.npy'.format(output_directory, label, split, index, epoch)
                labels_unbalanced_filename = '{}/{}_{}-labels_unbalanced-{:03d}.npy'.format(output_directory, label, split, index)

                if os.path.exists(labels_unbalanced_filename): continue

                read_time = time.time()
                sys.stdout.write('Reading chunk {} of {}...'.format(index + 1, no_chunks))
                X_chunk = self.read_unbalanced_chunk(label, index)
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
        
        if not balanced:
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

    def run_inference(self, epoch = 'opt', balanced = False):
        """
        Run all inference for this experiment. 

        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param balanced: only use balanced benign data (default = False)
        """
        # compile the model and load model weights 
        self.compile_model()
        self.load_weights(epoch)

        # open a file to write results 
        if not balanced: results_filename = '{}/inference_unbalanced-{}.txt'.format(self.results_directory, epoch)
        else: results_filename = '{}/inference-{}.txt'.format(self.results_directory, epoch)
        fd = open(results_filename, 'w') 

        # get the results for the in- and out-of-distribution examples
        self.infer(self.ID_examples, 'ID', 'test', fd, epoch, balanced)
        self.infer(self.OOD_examples, 'OOD', 'test', fd, epoch, balanced)
            
        # close the open file
        fd.close()

    def run_exhaustive_inference(self, balanced = False):
        """
        Run all inference for this experiment for all epochs. 

        @param balanced: only use balanced benign data (default = False)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.run_inference('{:03d}'.format(epoch), balanced)

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.run_inference('opt', balanced)

    def extract_features(self, examples, label, split, layer, epoch = 'opt', balanced = False):
        """
        Extract features from an intermediate layer.

        @param examples: the list of example datasets
        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test') 
        @param layer: the layer of the network to extract
        @param epoch: the epoch to run inference on (default = 'opt')
        @param balanced: only use balanced benign data (default = False)
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

        if not balanced:
            # create a new output directory 
            output_directory = '{}/unbalanced_data'.format(self.temp_directory)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok = True)

            # break up X into small batches for inference
            no_chunks = self.no_unbalanced_chunks(label)
            for index in range(no_chunks):
                # skip if this feature already was calculated
                feature_filename = '{}/{}_{}-features_{}_unbalanced-{:03d}-{}.npy'.format(output_directory, label, split, layer, index, epoch)
                if os.path.exists(feature_filename): continue

                read_time = time.time()
                sys.stdout.write('Reading chunk {} of {}...'.format(index + 1, no_chunks))
                X_chunk = self.read_unbalanced_chunk(label, index)
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

    def run_feature_extraction(self, layer, epoch = 'opt', balanced = False):
        """
        Run the intermediate algorithm that extracts features for training data, in-distribution
        and out-of-distribution testing data. Save features to disk for further analysis.
       
        @param layer: the layer of the network to extract
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param balanced: only use balanced benign data (default = False)
        """
        # compile the model and load model weights 
        self.compile_model()
        self.load_weights(epoch)

        # extract features from the intermediate layer 
        # training data is always balanced
        self.extract_features(self.ID_examples, 'ID', 'train', layer, epoch, True)
        self.extract_features(self.ID_examples, 'ID', 'test', layer, epoch, balanced)
        if len(self.OOD_examples):
            self.extract_features(self.OOD_examples, 'OOD', 'test', layer, epoch, balanced)

    def run_exhaustive_feature_extraction(self, layer, balanced = False):
        """
        Run all feature extraction for this experiment for all epochs. 

        @param layer: the layer of the network to extract
        @param balanced: only use balanced benign data (default = False)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.run_feature_extraction(layer, '{:03d}'.format(epoch), balanced)

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.run_feature_extraction(layer, 'opt', balanced)

    def extract_feature_attributions(self, examples, label, split, layer, method, epoch = 'opt'):
        """
        Extract attributions features from an intermediate layer.

        @param examples: the list of example datasets
        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test') 
        @param layer: the layer of the network to extract
        @param method: the attribution method to perform (str)
        @param epoch: the epoch to run inference on (default = 'opt')
        """
        # skip if no examples 
        if not len(examples): return

        # read the feature vector and extract the intermediate features 
        X = self.read_examples(examples, split)
        attributions = self.model.extract_attributions(X, layer, method)
        # save the features to the temp directory for later use
        attribution_filename = '{}/{}_{}-{}_{}-{}.npy'.format(self.temp_directory, label, split, method, layer, epoch)
        np.save(attribution_filename, attributions)
        
    def run_feature_attribution_extraction(self, layer, method, epoch = 'opt'):
        """
        Extract the feature attributes using integrated gradients from the captum 
        library. 

        @param layer: the layer of the network to extract attributes
        @param method: the attribution method to perform (str)
        @param epoch: the epoch to run the experiment on (default = 'opt')
        """
        # compile the model and load model weights 
        self.compile_model()
        self.load_weights(epoch)

        # extract attribute features from the intermediate layer 
        self.extract_feature_attributions(self.ID_examples, 'ID', 'train', layer, method, epoch)
        self.extract_feature_attributions(self.ID_examples, 'ID', 'test', layer, method, epoch)
        self.extract_feature_attributions(self.OOD_examples, 'OOD', 'test', layer, method, epoch)

    def run_exhaustive_feature_attribution_extraction(self, layer, method):
        """
        Run all attribution extraction for this experiment for all epochs. 

        @param layer: the layer of the network to extract
        @param method: the attribution method to perform (str)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.run_feature_attribution_extraction(layer, method, '{:03d}'.format(epoch))

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.run_feature_attribution_extraction(layer, method, 'opt')

    def ood_infer(self, detector, examples, label, split = 'test', epoch = 'opt', balanced = False):
        """
        Run inference with an OOD detector on these examples and save the predictions to disk.

        @param detector: the OOD detector to run the tests for
        @param examples: the list of example datasets
        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test') 
        @param epoch: the epoch to run inference on (default = 'opt')
        @param balanced: only use balanced benign data (default = False)
        """
        # skip if no examples
        if not len(examples): return

        if detector == 'ODIN': 
            ood_detector =  ODIN(
                self.model.model,
                temperature = 1000.0,
            )
        elif detector == 'EnergyBased':
            ood_detector = EnergyBased(
                self.model.model,
            )
        else:
            assert ('Unrecognized OOD method: {}'.format(detector))

        # read the examples since it is required regardless on inference
        X = self.read_examples(examples, split)

        OOD_scores_filename = '{}/{}_{}-OOD_{}_scores-{}.npy'.format(self.temp_directory, label, split, detector, epoch)
        if not os.path.exists(OOD_scores_filename):
            # run inference for these examples and split
            scores = self.model.ood_infer(X, ood_detector)
            
            np.save(OOD_scores_filename, scores)

        ood_scores = np.load(OOD_scores_filename)

        if not balanced:
            ood_scores = [ood_scores]

            # create a new output directory 
            OOD_directory = '{}/unbalanced_data'.format(self.temp_directory)
            if not os.path.exists(OOD_directory):
                os.makedirs(OOD_directory, exist_ok = True)

            # break up X into small batches for inference
            no_chunks = self.no_unbalanced_chunks(label)
            for index in range(no_chunks):
                OOD_scores_unbalanced_filename = '{}/{}_{}-OOD_{}_scores_unbalanced-{:03d}-{}.npy'.format(OOD_directory, label, split, detector, index, epoch)

                if os.path.exists(OOD_scores_unbalanced_filename): 
                    read_time = time.time()
                    sys.stdout.write('Reading {}...'.format(OOD_scores_unbalanced_filename))
                    ood_scores.append(np.load(OOD_scores_unbalanced_filename))
                    sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - read_time))
                    continue

                read_time = time.time()
                sys.stdout.write('Reading chunk {} of {}...'.format(index + 1, no_chunks))
                X_chunk = self.read_unbalanced_chunk(label, index)
                X_chunk['target'] = X_chunk[self.target].map(self.target_mapping)
                sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - read_time))

                infer_time = time.time()
                sys.stdout.write('Inferring on chunk {} of {}...'.format(index + 1, no_chunks))

                scores = self.model.ood_infer(X_chunk, ood_detector)
                ood_scores.append(scores)

                # save the predictions and outputs, moved here so that testing
                # for predictions filename means inference completed successfully
                np.save(OOD_scores_unbalanced_filename, scores)

                sys.stdout.write('done in {:0.2f} seconds.\n'.format(time.time() - infer_time))

            ood_scores = np.concatenate(ood_scores)

        return ood_scores

    def run_ood_inference(self, detector, epoch = 'opt', balanced = False):
        """
        Run inference using an OOD detector for this experiment. 

        @param detector: the OOD detector to run the tests for
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param balanced: only use balanced benign data (default = False)
        """
        # compile the model and load model weights 
        self.compile_model()
        self.load_weights(epoch)

        start_time = time.time()

        # get the results for the in- and out-of-distribution examples
        ID_scores = self.ood_infer(detector, self.ID_examples, 'ID', 'test', epoch, balanced)
        OOD_scores = self.ood_infer(detector, self.OOD_examples, 'OOD', 'test', epoch, balanced)

        if detector == 'EnergyBased':
            # we look at the bottom 95% of in-distribution examples 
            # these are considered in-distribution. anything in the bottom 
            # 5% is considered out-of-distribution (false positives)
            # we want to find the number of true positives, OOD inputs that 
            # have scores higher than this threshold. since low values are better 
            # under this metric, we take the top 5% as the cutoff. Lower values 
            # are better since we are returning the loss
            true_negative_threshold = sorted(ID_scores)[19 * ID_scores.size // 20]

            # what proportion of OOD inputs have a score lower than the threshold 
            # higher scores mean less likely to belong to the distribution
            true_positives = sum(OOD_scores > true_negative_threshold)
            true_positive_rate = true_positives / OOD_scores.size

            # create an AUC ROC value for detecting OOD inputs
            ODIN_score = np.concatenate((ID_scores, OOD_scores))
            # since higher scores should indicate OOD-ness, OOD receives labels of 1
            ODIN_truth = np.concatenate((np.zeros(ID_scores.size), np.ones(OOD_scores.size)))
            roc_auc_score = sklearn.metrics.roc_auc_score(ODIN_truth, ODIN_score)

            true_negative_threshold = sorted(ID_scores)[17 * ID_scores.size // 20]
            true_positives = sum(OOD_scores > true_negative_threshold)
            true_positive_rate_85 = true_positives / OOD_scores.size
        elif detector == 'ODIN':
            # we look at the top 95% of in-distribution examples 
            # these are considered in-distribution. anything in the bottom 
            # 5% is considered out-of-distribution (false positives)
            # we want to find the number of true positives, OOD inputs that 
            # have scores lower than this threshold
            true_negative_threshold = sorted(ID_scores)[ID_scores.size // 20]
            # what proportion of OOD inputs have a score lower than the threshold 
            # lower scores mean more likely to belong to the distribution
            true_positives = sum(OOD_scores < true_negative_threshold)
            true_positive_rate = true_positives / OOD_scores.size

            # create an AUC ROC value for detecting OOD inputs
            ODIN_score = np.concatenate((ID_scores, OOD_scores))
            # since lower scores should indicate OOD-ness, OOD receives labels of 0
            ODIN_truth = np.concatenate((np.ones(ID_scores.size), np.zeros(OOD_scores.size)))
            roc_auc_score = sklearn.metrics.roc_auc_score(ODIN_truth, ODIN_score)

            true_negative_threshold = sorted(ID_scores)[3 * ID_scores.size // 20]
            true_positives = sum(OOD_scores < true_negative_threshold)
            true_positive_rate_85 = true_positives / OOD_scores.size

        if not balanced:
            timing_filename = '{}/{}_unbalanced-{}.txt'.format(self.timing_directory, detector, epoch)
            output_filename = '{}/{}_unbalanced-{}.txt'.format(self.results_directory, detector, epoch)
        else:
            timing_filename = '{}/{}-{}.txt'.format(self.timing_directory, detector, epoch)
            output_filename = '{}/{}-{}.txt'.format(self.results_directory, detector, epoch)
            
        # output the timing results
        with open(timing_filename, 'w') as fd:
            fd.write('No. ID Packets: {}\n'.format(ID_scores.size))
            fd.write('No. OOD Packets: {}\n'.format(OOD_scores.size))
            fd.write('Time: {:.04f} seconds\n'.format(time.time() - start_time))

        # output the results 
        with open(output_filename, 'w') as fd:
            fd.write('TPR (TNR = 95%): {:0.8f}%\n'.format(100 * true_positive_rate))
            fd.write('TPR (TNR = 85%): {:0.8f}%\n'.format(100 * true_positive_rate_85))
            fd.write('AUC: {:0.8f}\n'.format(roc_auc_score))

    def run_exhaustive_ood_inference(self, detector, balanced = False):
        """
        Run inference using an OOD detector for this experiment for all epochs. 

        @param detector: the OOD detector to run the tests for
        @param balanced: only use balanced benign data (default = False)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.run_inference('{:03d}'.format(epoch), balanced)

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.run_ood_inference(detector, 'opt', balanced)

    def read_outputs(self, label, split = 'test', epoch = 'opt', balanced = True):
        """
        Read the outputs for the neural network.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param balanced: only use balanced benign data (default = True)
        """
        filename = '{}/{}_{}-outputs-{}.npy'.format(self.results_directory, label, split, epoch)
        assert (os.path.exists(filename))
        outputs = np.load(filename)

        if not balanced:
            outputs = [outputs]

            input_directory = '{}/unbalanced_data'.format(self.results_directory)
            no_chunks = self.no_unbalanced_chunks(label)
            for index in range(no_chunks):
                filename = '{}/{}_{}-outputs_unbalanced-{:03d}-{}.npy'.format(input_directory, label, split, index, epoch)
                outputs.append(np.load(filename))

            outputs = np.concatenate(outputs)

        return outputs

    def read_predictions(self, label, split = 'test', epoch = 'opt', balanced = True):
        """
        Read the predictions for the neural network.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param balanced: only use balanced benign data (default = True)
        """
        filename = '{}/{}_{}-predictions-{}.npy'.format(self.results_directory, label, split, epoch)
        assert (os.path.exists(filename))
        predictions = np.load(filename)

        if not balanced:
            predictions = [predictions]

            input_directory = '{}/unbalanced_data'.format(self.results_directory)
            no_chunks = self.no_unbalanced_chunks(label)
            for index in range(no_chunks):
                filename = '{}/{}_{}-predictions_unbalanced-{:03d}-{}.npy'.format(input_directory, label, split, index, epoch)
                predictions.append(np.load(filename))

            predictions = np.concatenate(predictions)

        return predictions

    def read_features(self, label, layer, split = 'test', epoch = 'opt', balanced = True):
        """
        Read the extracted features from an intermediate layer.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param layer: the extracted layer to read 
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param balanced: only use balanced benign data (default = True)
        """
        filename = '{}/{}_{}-features_{}-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        assert (os.path.exists(filename))
        features = np.load(filename)

        if not balanced:
            features = [features]

            input_directory = '{}/unbalanced_data'.format(self.temp_directory)
            no_chunks = self.no_unbalanced_chunks(label)
            for index in range(no_chunks):
                filename = '{}/{}_{}-features_{}_unbalanced-{:03d}-{}.npy'.format(input_directory, label, split, layer, index, epoch)
                features.append(np.load(filename))

            features = np.concatenate(features)

        return features

    def read_feature_attributions(self, label, method, layer, split = 'test', epoch = 'opt'):
        """
        Read the feature attributions from an intermediate layer. 

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param method: the attribution method to perform (str)
        @param layer: the extracted layer to read 
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt') 
        """
        filename = '{}/{}_{}-attributions_{}_{}-{}.npy'.format(self.temp_directory, label, split, method, layer, epoch)
        assert (os.path.exists(filename))
        attributions = np.load(filename)

        return attributions

    def read_mahalanobis_distances(self, label, layer, split = 'test', epoch = 'opt', stratify = 'target', balanced = True):
        """
        Read the mahalanobis distances from the extracted intermediate layer. 

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param layer: the extracted layer to read 
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param stratify: the column to compute covariance matrices (default = 'target')
        @param balanced: only use balanced benign data (default = True)
        """
        if not balanced: filename = '{}/{}_{}-mahalanobis_{}_{}_unbalanced-{}.npy'.format(self.temp_directory, label, split, stratify, layer, epoch)
        else: filename = '{}/{}_{}-mahalanobis_{}_{}-{}.npy'.format(self.temp_directory, label, split, stratify, layer, epoch)
        assert (os.path.exists(filename))
        distances = np.load(filename)

        return distances

    def read_normalizing_flows(self, label, layer, split = 'test', epoch = 'opt', balanced = True):
        """
        Read the normalizing flows results from the extracted intermediate layer. 

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param layer: the extracted layer to read 
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param balanced: only use balanced benign data (default = True)
        """
        if not balanced: filename = '{}/{}_{}-normalizing_flows_{}_unbalanced-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        else: filename = '{}/{}_{}-normalizing_flows_{}-{}.npy'.format(self.temp_directory, label, split, layer, epoch)
        assert (os.path.exists(filename))
        losses = np.load(filename)

        return losses

    def stack_features(self, label, split = 'test', epoch = 'opt'):
        """
        Stack the features and save the augmented feature space. 

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param split: the split of the data (default = 'test')
        @param epoch: the epoch to stack features (default = 'opt')
        """
        # create an empty set of features and return concatenated version 
        features = []

        for layer in self.model.model.extractable_layers:
            features.append(self.read_features(label, layer, split, epoch))

        features = np.concatenate(features, axis = 1)

        # save the features to disk 
        feature_filename = '{}/{}_{}-features_stacked-{}.npy'.format(self.temp_directory, label, split, epoch)
        np.save(feature_filename, features)

    def stack_feature_layers(self, epoch = 'opt'):
        """
        Stack the extracted features together. 

        @param epoch: the epoch to stack features (default = 'opt')
        """
        # compile the model 
        self.compile_model()

        self.stack_features('ID', 'train', epoch)
        self.stack_features('ID', 'test', epoch)
        self.stack_features('OOD', 'test', epoch)

    def apply_feature_attributions(self, features, attributions):
        """
        Apply the feature attributions to the features themselves. 

        @param features: the features extracted from an intermediate layer
        @param attributions: the extracted feature attributions from an intermediate layer (default = None)
        """
        attributions = np.abs(attributions)
        
        attributions = sklearn.preprocessing.normalize(attributions, norm = 'l2', axis = 1)

        return np.multiply(features, attributions)

    def recall(self, label, epoch = 'opt'):
        """
        Calculate the recall for each category.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param epoch: the epoch to calculate the recall (default = 'opt')
        """
        if label == 'ID':
            X_test = self.read_examples(self.ID_examples, 'test')
        else:
            X_test = self.read_examples(self.OOD_examples, 'test')
        categories = sorted(pd.unique(X_test['packet_category']))

        # read the predictions 
        predictions = self.read_predictions(label, 'test', epoch)
        labels = X_test[self.target].map(self.target_mapping).values

        # open the filename
        results_filename = '{}/{}_recall-{}.txt'.format(self.results_directory, label, epoch)
        fd = open(results_filename, 'w')

        # go through each category
        for category in categories:
            category_predictions = predictions[X_test['packet_category'] == category]
            category_labels = labels[X_test['packet_category'] == category]

            # create the recall for this category 
            recall = np.sum(category_predictions == category_labels) / category_predictions.shape[0]
            fd.write('{}: {:0.8f}\n'.format(category, recall))

        # close the file
        fd.close()

    def maximum_softmax_probability(self, epoch = 'opt', balanced = False):
        """
        Compute the MSP for the in-distribution and out-of-distribution test samples.

        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param balanced: only use balanced benign data (default = False)
        """
        # only consider experiments with OOD examples 
        if not len(self.OOD_examples): return

        # start timing statistics 
        start_time = time.time()
        # cast to np.float128 to avoid overflow when calculating softmax
        ID_outputs = self.read_outputs('ID', 'test', epoch, balanced).astype(np.float128)
        OOD_outputs = self.read_outputs('OOD', 'test', epoch, balanced).astype(np.float128)

        # compute softmax from outputs
        ID_outputs = np.exp(ID_outputs) / np.sum(np.exp(ID_outputs), axis=1, keepdims=True)
        OOD_outputs = np.exp(OOD_outputs) / np.sum(np.exp(OOD_outputs), axis=1, keepdims=True)

        ID_max_softmax = np.max(ID_outputs, axis = 1)
        OOD_max_softmax = np.max(OOD_outputs, axis = 1)

        # we look at the top 95% of in-distribution examples 
        # these are considered in-distribution. anything in the bottom 
        # 5% is considered out-of-distribution (false positives)
        # we want to find the number of true positives, OOD inputs that 
        # have scores lower than this threshold
        true_negative_threshold = sorted(ID_max_softmax)[ID_max_softmax.size // 20]
        
        # what proportion of OOD inputs have a score lower than the threshold 
        true_positives = sum(OOD_max_softmax < true_negative_threshold)
        true_positive_rate = true_positives / OOD_max_softmax.size
        
        # create an AUC ROC value for detecting OOD inputs
        softmax_score = np.concatenate((ID_max_softmax, OOD_max_softmax))
        # since lower scores should indicate OOD-ness, OOD receives labels of 0
        softmax_truth = np.concatenate((np.ones(ID_max_softmax.size), np.zeros(OOD_max_softmax.size)))
        roc_auc_score = sklearn.metrics.roc_auc_score(softmax_truth, softmax_score)

        true_negative_threshold = sorted(ID_max_softmax)[3 * ID_max_softmax.size // 20]
        true_positives = sum(OOD_max_softmax < true_negative_threshold)
        true_positive_rate_85 = true_positives / OOD_max_softmax.size

        if not balanced:
            timing_filename = '{}/maximum_softmax_probability_unbalanced-{}.txt'.format(self.timing_directory, epoch)
            output_filename = '{}/maximum_softmax_probability_unbalanced-{}.txt'.format(self.results_directory, epoch)
        else:
            timing_filename = '{}/maximum_softmax_probability-{}.txt'.format(self.timing_directory, epoch)
            output_filename = '{}/maximum_softmax_probability-{}.txt'.format(self.results_directory, epoch)
        
        # output the timing results
        with open(timing_filename, 'w') as fd:
            fd.write('No. ID Packets: {}\n'.format(ID_outputs.shape[0]))
            fd.write('No. OOD Packets: {}\n'.format(OOD_outputs.shape[0]))
            fd.write('Time: {:.04f} seconds\n'.format(time.time() - start_time))

        # output the results 
        with open(output_filename, 'w') as fd:
            fd.write('TPR (TNR = 95%): {:0.8f}%\n'.format(100 * true_positive_rate))
            fd.write('TPR (TNR = 85%): {:0.8f}%\n'.format(100 * true_positive_rate_85))
            fd.write('AUC: {:0.8f}\n'.format(roc_auc_score)) 

    def compute_exhaustive_maximum_softmax_probability(self, balanced = False):
        """
        Run all MSP for this experiment for all epochs. 

        @param balanced: only use balanced benign data (default = False)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.maximum_softmax_probability('{:03d}'.format(epoch), balanced)

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.maximum_softmax_probability('opt', balanced)

    def compute_robust_covariance_matrix(self, layer, epoch = 'opt', stratify = 'target', attribution_method = ''):
        """
        Compute the robust covariance matrix for each class (benign/attack).

        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param stratify: the column to compute covariance matrices (default = 'target')
        @param attribution_method: include feature attributes (default = '', i.e., None)
        """
        # read the training dataset
        X_train = self.read_examples(self.ID_examples, 'train')
        
        # get the training features from disk
        features = self.read_features('ID', layer, 'train', epoch)

        # read attribution if requested 
        if len(attribution_method):
            attributions = self.read_feature_attributions('ID', attribution_method, layer, 'train', epoch)
            # add underscore to follow correct naming conventions
            attribution_method = '_{}'.format(attribution_method)
        
        # compute covariance for each label 
        categories = pd.unique(X_train[stratify])
        for category in categories:
            # extract only the features for this label
            category_features = features[X_train[stratify] == category]
            
            # if attribution is requested multiply the category_features by the relevant attribute values 
            if len(attribution_method):
                # get the attributions corresponding to this category and only select the rows where the training
                # target equals the category in question. Perform element-wise multiplication
                category_features = self.apply_feature_attributions(
                    category_features,
                    attributions[X_train[stratify] == category],
                )
                
            # compute robust convariance matrix
            robust_covariance = EmpiricalCovariance().fit(category_features)
            
            # compute max and min distance
            distance_from_covariance = robust_covariance.mahalanobis(category_features)
            max_distance = max(distance_from_covariance)
            min_distance = min(distance_from_covariance)

            # save the column that was used for computing this covariance matrix
            covariance_filename = '{}/covariance{}_{}_{}-{}-{}.pickle'.format(self.temp_directory, attribution_method, stratify, layer, category, epoch)
            with open(covariance_filename, 'wb') as fd:
                # pickle the data using highest protocol available
                pickle.dump(robust_covariance, fd, pickle.HIGHEST_PROTOCOL)
                pickle.dump(min_distance, fd, pickle.HIGHEST_PROTOCOL)
                pickle.dump(max_distance, fd, pickle.HIGHEST_PROTOCOL)
    
    def compute_exhaustive_robust_covariance_matrix(self, layer, stratify = 'target', attribution_method = ''):
        """
        Run all covariance computations for this experiment for all epochs. 

        @param layer: the extracted layer to read 
        @param stratify: the column to compute covariance matrices (default = 'target')
        @param attribution_method: include feature attributes (default = '', i.e., None)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.compute_robust_covariance_matrix(layer, '{:03d}'.format(epoch), stratify, attribution_method)

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.compute_robust_covariance_matrix(layer, 'opt', stratify, attribution_method)

    def mahalanobis_distance(self, predictions, features, layer, epoch = 'opt', stratify = 'target', attributions = None, attribution_method = ''):
        """
        Find the distance to the target labels for each feature.

        @param predictions: the label predicted for each feature 
        @param features: the features extracted from an intermediate layer
        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param stratify: the column to compute covariance matrices (default = 'target')
        @param attributions: the extracted feature attributions from an intermediate layer (default = None)
        @param attribution_method: include feature attributes (default = '', i.e., None)
        """
        # read the training dataset to get the target labels
        X_train = self.read_examples(self.ID_examples, 'train')
      
        # set the attribution boolean based on the input
        attribution = (attributions is not None)

        # read the covariance matricies 
        robust_covariances = {}
        max_distances = {}
        min_distances = {}
        for category in pd.unique(X_train[stratify]):
            covariance_filename = '{}/covariance{}_{}_{}-{}-{}.pickle'.format(self.temp_directory, attribution_method, stratify, layer, category, epoch)
            with open(covariance_filename, 'rb') as fd:
                # read the pickled data
                robust_covariances[category] = pickle.load(fd)
                min_distances[category] = pickle.load(fd)
                max_distances[category] = pickle.load(fd)

        # quicker to batch compute the distances and throwaway unused computations
        # set distances values to infinity since we eventually will take the minimum for 
        # any categories that map to the same target value
        distances = np.inf * np.ones(features.shape[0])
        # divide the data by the target column so that we can then get all unique values in stratify column
        # n.b., this assumes that no category in stratify take multiple values in target
        for _, X_train_by_target in X_train.groupby('target'):
            # if stratify is target, only one label returned
            for category in pd.unique(X_train_by_target[stratify]):
                # assert that all categories fall to a unique target 
                labels = pd.unique(X_train[X_train[stratify] == category]['target'])
                assert (len(labels) == 1)
                label = labels[0]

                # get the attributions corresponding to this category and only select the rows where the training
                # target equals the category in question. Perform element-wise multiplication
                if attribution: category_features = self.apply_feature_attributions(features, attributions)
                else: category_features = features

                # get the distances for all examples 
                distances_from_category = (robust_covariances[category].mahalanobis(category_features) - min_distances[category]) / max_distances[category]

                # update the distances only for the predictions that match this label
                # np.minimum returns the elementwise min (distances starts at infinity)
                # we only update predictions that have this label (so if prediction is attack and this category is attack)
                # however, we can stratify based on types of attacks to create more distributions
                # in that case, we take the closest distribution for the corresponding category
                # (not used in work for DLSP 2023 since we stratify by target)
                distances[predictions == label] = np.minimum(distances[predictions == label], distances_from_category[predictions == label])
                
        return distances

    def compute_mahalanobis_distance(self, layer, epoch = 'opt', stratify = 'target', attribution_method = '', balanced = False):
        """
        Compute the Mahalanobis distance for the in-distribution and out-of-distribution
        test samples.

        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param stratify: the column to compute covariance matrices (default = 'target')
        @param attribution_method: include feature attributes (default = '', i.e., None)
        @param balanced: only use balanced benign data (default = False)
        """
        # only consider experiments with OOD examples 
        if not len(self.OOD_examples): return

        # start timing statistics 
        start_time = time.time()

        ID_features = self.read_features('ID', layer, 'test', epoch, balanced)
        OOD_features = self.read_features('OOD', layer, 'test', epoch, balanced)

        ID_predictions = self.read_predictions('ID', 'test', epoch, balanced)
        OOD_predictions = self.read_predictions('OOD', 'test', epoch, balanced)

        if len(attribution_method):
            ID_attributions = self.read_feature_attributions('ID', attribution_method, layer, 'test', epoch)
            OOD_attributions = self.read_feature_attributions('OOD', attribution_method, layer, 'test', epoch)
            # add underscore to follow correct naming conventions
            attribution_method = '_{}'.format(attribution_method)
        else:
            ID_attributions = None 
            OOD_attributions = None

        ID_mahalanobis = self.mahalanobis_distance(ID_predictions, ID_features, layer, epoch, stratify, ID_attributions, attribution_method)
        OOD_mahalanobis = self.mahalanobis_distance(OOD_predictions, OOD_features, layer, epoch, stratify, OOD_attributions, attribution_method)

        # save the distances 
        if not balanced:
            ID_filename = '{}/ID_test-mahalanobis{}_{}_{}_unbalanced-{}.npy'.format(self.temp_directory, attribution_method, stratify, layer, epoch)
            OOD_filename = '{}/OOD_test-mahalanobis{}_{}_{}_unbalanced-{}.npy'.format(self.temp_directory, attribution_method, stratify, layer, epoch)
        else:
            ID_filename = '{}/ID_test-mahalanobis{}_{}_{}-{}.npy'.format(self.temp_directory, attribution_method, stratify, layer, epoch)
            OOD_filename = '{}/OOD_test-mahalanobis{}_{}_{}-{}.npy'.format(self.temp_directory, attribution_method, stratify, layer, epoch)
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
        # since higher scores should indicate OOD-ness, OOD receives labels of 1
        mahalanobis_truth = np.concatenate((np.zeros(ID_mahalanobis.size), np.ones(OOD_mahalanobis.size)))
        roc_auc_score = sklearn.metrics.roc_auc_score(mahalanobis_truth, mahalanobis_score)

        true_negative_threshold = sorted(ID_mahalanobis)[17 * ID_mahalanobis.size // 20]
        true_positives = sum(OOD_mahalanobis > true_negative_threshold)
        true_positive_rate_85 = true_positives / OOD_mahalanobis.size

        if not balanced:
            timing_filename = '{}/mahalanobis{}_{}_{}_unbalanced-{}.txt'.format(self.timing_directory, attribution_method, stratify, layer, epoch)
            output_filename = '{}/mahalanobis{}_{}_{}_unbalanced-{}.txt'.format(self.results_directory, attribution_method, stratify, layer, epoch)
        else:
            timing_filename = '{}/mahalanobis{}_{}_{}-{}.txt'.format(self.timing_directory, attribution_method, stratify, layer, epoch)
            output_filename = '{}/mahalanobis{}_{}_{}-{}.txt'.format(self.results_directory, attribution_method, stratify, layer, epoch)
        
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

    def compute_exhaustive_mahalanobis_distance(self, layer, stratify = 'target', attribution_method = '', balanced = False):
        """
        Run all mahalanobis computations for this experiment for all epochs. 

        @param layer: the extracted layer to read 
        @param stratify: the column to compute covariance matrices (default = 'target')
        @param attribution_method: include feature attributes (default = '', i.e., None)
        @param balanced: only use balanced benign data (default = False)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.compute_mahalanobis_distance(layer, '{:03d}'.format(epoch), stratify, attribution_method, balanced)

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.compute_mahalanobis_distance(layer, 'opt', stratify, attribution_method, balanced)

    def train_normalizing_flows(self, layer, stratify = 'packet_category', epoch = 'opt', attribution_method = '', nblocks = 20, affine_clamping = 2.0):
        """
        Train a normalizing flow for each class (benign/attack).

        @param layer: the extracted layer to read 
        @param stratify: the column to stratify training and validation (default = 'packet_category')
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param attribution_method: include feature attributes (default = '', i.e., None)
        @param nblocks: the number of blocks in the flow (default = 20)
        @param affine_clamping: the clamping parameter (default = 2.0)
        """
        # read the training dataset
        X_train = self.read_examples(self.ID_examples, 'train')
        
        # get the training features from disk
        features = self.read_features('ID', layer, 'train', epoch)

        # read attribution if requested 
        if len(attribution_method):
            attributions = self.read_feature_attributions('ID', attribution_method, layer, 'train', epoch)
            # add underscore to follow correct naming conventions
            attribution_method = '_{}'.format(attribution_method)

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

            # if attribution is requested multiply the label_features by the relevant attribute values 
            if len(attribution_method):
                # get the attributions corresponding to this category and only select the rows where the training
                # target equals the category in question. Perform element-wise multiplication
                label_features = self.apply_feature_attributions(
                    label_features,
                    attributions[X_train['target'] == label],
                )

            # get the suffix (used for ablation studies)
            if nblocks == 20 and abs(affine_clamping - 2.0) < 1e-6: suffix = ''
            else: suffix = '-nblocks-{:03d}-affine_clamping-{:03d}'.format(nblocks, int(10 * affine_clamping))

            # save the model to disk after training
            model_filename = '{}/normalizing_flows{}_{}-{}-{}{}'.format(self.model_directory, attribution_method, layer, label, epoch, suffix)
            loss_filename = '{}/normalizing_flows{}_{}-{}{}.png'.format(self.figures_directory, attribution_method, layer, epoch, suffix)

            # make sure to not overwrite an existing model
            if os.path.exists(model_filename): continue

            # create a pandas dataframe for training 
            training_data = {
                'features': label_features,
                'stratify': label_categories,
            }

            # create and train normalizing flow model 
            flow = InverseAutoregressiveFlow(label_features.shape[1], gpu = self.gpu, nblocks = nblocks, affine_clamping = affine_clamping)
            flow.train(training_data, model_filename, loss_filename, legend[label])

            print ('Trained normalizing flow in {:0.2f} seconds for target {}.'.format(time.time() - start_time, label))
        
        # close the figures so that any future runs will not append to the same canvas 
        plt.close()

    def train_exhaustive_normalizing_flows(self, layer, attribution_method = ''):
        """
        Train all normalizing flows for this experiment for all epochs. 

        @param layer: the extracted layer to read 
        @param attribution_method: include feature attributes (default = '', i.e., None)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.train_normalizing_flows(layer, '{:03d}'.format(epoch), attribution_method = attribution_method)

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.train_normalizing_flows(layer, 'opt', attribution_method = attribution_method)

    def normalizing_flow(self, predictions, features, layer, epoch = 'opt', attributions = None, attribution_method = '', nblocks = 20, affine_clamping = 2.0):
        """
        Load a normalizing flow model and calculate probabilities for this sample.

        @param predictions: the label predicted for each feature 
        @param features: the features extracted from an intermediate layer
        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')    
        @param attributions: the extracted feature attributions from an intermediate layer (default = None)
        @param attribution_method: include feature attributes (default = '', i.e., None)
        @param nblocks: the number of blocks in the flow (default = 20)
        @param affine_clamping: the clamping parameter (default = 2.0)
        """
        # read the training dataset to get the target labels
        X_train = self.read_examples(self.ID_examples, 'train')

        # set the attribution boolean based on the input
        attribution = (attributions is not None)

        # create a zero vector for the log probailities 
        log_probabilities = np.zeros(features.shape[0], dtype = np.float32)

        # iterate over all input labels 
        for label in pd.unique(X_train['target']):
            # get the suffix (used for ablation studies)
            if nblocks == 20 and abs(affine_clamping - 2.0) < 1e-6: suffix = ''
            else: suffix = '-nblocks-{:03d}-affine_clamping-{:03d}'.format(nblocks, int(10 * affine_clamping))

            # load the model to disk from training
            model_filename = '{}/normalizing_flows{}_{}-{}-{}{}'.format(self.model_directory, attribution_method, layer, label, epoch, suffix)
            
            # get the attributions corresponding to this category and only select the rows where the training
            # target equals the category in question. Perform element-wise multiplication
            if attribution: category_features = self.apply_feature_attributions(features, attributions)
            else: category_features = features

            # create and load model weights for normalizing flows
            flow = InverseAutoregressiveFlow(category_features.shape[1], gpu = self.gpu, nblocks = nblocks, affine_clamping = affine_clamping)
            flow.load_weights(model_filename)

            log_probabilities_from_label = flow.inverse(category_features)

            log_probabilities[predictions == label] = log_probabilities_from_label[predictions == label]

        return log_probabilities

    def calculate_normalizing_flows(self, layer, epoch = 'opt', attribution_method = '', balanced = False, nblocks = 20, affine_clamping = 2.0):
        """
        Run the inverse normalizing flow for all test features to find OOD inputs.

        @param layer: the extracted layer to read 
        @param epoch: the epoch to run the experiment on (default = 'opt')
        @param attribution_method: include feature attributes (default = '', i.e., None)
        @param balanced: only use balanced benign data (default = False)
        @param nblocks: the number of blocks in the flow (default = 20)
        @param affine_clamping: the clamping parameter (default = 2.0)
        """
        # only consider experiments with OOD examples 
        if not len(self.OOD_examples): return

        # start timing statistics 
        start_time = time.time()

        ID_features = self.read_features('ID', layer, 'test', epoch, balanced)
        OOD_features = self.read_features('OOD', layer, 'test', epoch, balanced)

        ID_predictions = self.read_predictions('ID', 'test', epoch, balanced)
        OOD_predictions = self.read_predictions('OOD', 'test', epoch, balanced)

        # read attribution if requested 
        if len(attribution_method):
            ID_attributions = self.read_feature_attributions('ID', attribution_method, layer, 'test', epoch)
            OOD_attributions = self.read_feature_attributions('OOD', attribution_method, layer, 'test', epoch)
            # add underscore to follow correct naming conventions
            attribution_method = '_{}'.format(attribution_method)
        else:
            ID_attributions = None 
            OOD_attributions = None

        # get the suffix (used for ablation studies)
        if nblocks == 20 and abs(affine_clamping - 2.0) < 1e-6: suffix = ''
        else: suffix = '-nblocks-{:03d}-affine_clamping-{:03d}'.format(nblocks, int(10 * affine_clamping))

        if not balanced:
            ID_filename = '{}/ID_test-normalizing_flows{}_{}_unbalanced-{}{}.npy'.format(self.temp_directory, attribution_method, layer, epoch, suffix)
            OOD_filename = '{}/OOD_test-normalizing_flows{}_{}_unbalanced-{}{}.npy'.format(self.temp_directory, attribution_method, layer, epoch, suffix)
        else:
            ID_filename = '{}/ID_test-normalizing_flows{}_{}-{}{}.npy'.format(self.temp_directory, attribution_method, layer, epoch, suffix)
            OOD_filename = '{}/OOD_test-normalizing_flows{}_{}-{}{}.npy'.format(self.temp_directory, attribution_method, layer, epoch, suffix)    
        
        if not os.path.exists(OOD_filename):
            # get the losses for ID and OOD
            ID_losses = self.normalizing_flow(ID_predictions, ID_features, layer, epoch, ID_attributions, attribution_method, nblocks, affine_clamping)
            OOD_losses = self.normalizing_flow(OOD_predictions, OOD_features, layer, epoch, OOD_attributions, attribution_method, nblocks, affine_clamping)

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
        # since higher scores should indicate OOD-ness, OOD receives labels of 1
        normalizing_flow_truth = np.concatenate((np.zeros(ID_losses.size), np.ones(OOD_losses.size)))
        roc_auc_score = sklearn.metrics.roc_auc_score(normalizing_flow_truth, normalizing_flow_score)

        true_negative_threshold = sorted(ID_losses)[17 * ID_losses.size // 20]
        true_positives = sum(OOD_losses > true_negative_threshold)
        true_positive_rate_85 = true_positives / OOD_losses.size

        if not balanced:
            timing_filename = '{}/normalizing_flows{}_{}_unbalanced-{}{}.txt'.format(self.timing_directory, attribution_method, layer, epoch, suffix)
            output_filename = '{}/normalizing_flows{}_{}_unbalanced-{}{}.txt'.format(self.results_directory, attribution_method, layer, epoch, suffix)
        else:
            timing_filename = '{}/normalizing_flows{}_{}-{}{}.txt'.format(self.timing_directory, attribution_method, layer, epoch, suffix)
            output_filename = '{}/normalizing_flows{}_{}-{}{}.txt'.format(self.results_directory, attribution_method, layer, epoch, suffix)
            
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

    def calculate_exhaustive_normalizing_flows(self, layer, attribution_method = '', balanced = False):
        """
        Run all normalizing flows for this experiment for all epochs. 

        @param layer: the extracted layer to read 
        @param attribution_method: include feature attributes (default = '', i.e., None)
        @param balanced: only use balanced benign data (default = False)
        """
        for epoch in range(self.nepochs):
            start_time = time.time()
            # convert epoch into a formatted string and run inference 
            self.calculate_normalizing_flows(layer, '{:03d}'.format(epoch), attribution_method, balanced)

            print ('Epoch {}/{} - time {:0.2f}s'.format(
                epoch + 1, 
                self.nepochs,
                time.time() - start_time,
            ))
        # run on the optimal network 
        self.calculate_normalizing_flows(layer, 'opt', attribution_method, balanced)