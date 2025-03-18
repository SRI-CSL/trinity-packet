# removes warning message from umap
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")



import os 
import sys
import time 
import umap


import numpy as np
import pandas as pd 



from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture


import matplotlib.pyplot as plt 
plt.style.use('seaborn-dark')



class NetflowContinualLearning(object):
    def __init__(self, experiment):
        """
        Class for the continual learning component of the pipeline. 

        @param experiment: a NetflowExperiment to operate on
        """
        self.experiment = experiment 
        # create directories for the learning process
        self.model_directory = '{}/continual-learning'.format(experiment.model_directory)
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory, exist_ok = True)
        self.results_directory = '{}/continual-learning'.format(experiment.results_directory)
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory, exist_ok = True)
        self.temp_directory = '{}/continual-learning'.format(experiment.temp_directory)
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory, exist_ok = True)
        self.samples_directory = '{}/samples'.format(self.temp_directory)
        if not os.path.exists(self.samples_directory):
            os.makedirs(self.samples_directory, exist_ok = True)

        # get the ID and OOD train and test examples
        self.ID_examples = self.experiment.ID_examples 
        self.OOD_examples = self.experiment.OOD_examples 
        self.ID_train = self.experiment.read_examples(self.ID_examples, 'train')
        self.ID_test = self.experiment.read_examples(self.ID_examples, 'test')
        # note that we train on the OOD 'test' examples and infer on the 
        # OOD 'train' examples! this is not a mistake as we have already seen 
        # the OOD 'test' examples during experiment inference but have never
        # used the OOD 'train' examples so we can treat those as a withheld 
        # dataset. 
        self.OOD_train = self.experiment.read_examples(self.OOD_examples, 'test')
        self.OOD_test = self.experiment.read_examples(self.OOD_examples, 'train')

        # need to reset the indices here so that an index in features maps to here
        self.ID_train.reset_index(drop = True, inplace = True)
        self.ID_test.reset_index(drop = True, inplace = True)
        self.OOD_train.reset_index(drop = True, inplace = True)
        self.OOD_test.reset_index(drop = True, inplace = True)

        # finetune for the same number of epochs as original training 
        self.nepochs = self.experiment.nepochs
        # make sure each category appears at least 5 times
        self.min_samples_per_category = 5

    def random_selection(self, X, train_size):
        """
        Return a random selection of the ID and OOD examples for training.

        @param X: the data to select from
        @param train_size: the number of examples to include
        """
        # allow for 100% labeled data 
        eps = 1e-6
        if abs(1.0 - train_size) < eps: 
            return X
        
        # guarantee at least five examples from each category
        nexamples = len(X.index)
        if int(round(nexamples * train_size)) < self.min_samples_per_category:
            return X.sample(self.min_samples_per_category)

        # use train_test_split to return a random sample of the data
        X_sampled, _ = train_test_split(X, test_size = 1.0 - train_size)
        
        return X_sampled

    def minimum_gradients(self, X, gradients, train_size):
        """
        Return the instances with the largest magnitude gradients.

        @param X: the data to select from
        @param gradients: the gradients for this dataset
        @param train_size: the number of examples to include        
        """
        # allow for 100% labeled data 
        eps = 1e-6
        if abs(1.0 - train_size) < eps: 
            return X

        # reset the index here on a copy of X so that the indices line up with 
        # the gradients which are index from (0, len(X.index))
        X = X.reset_index(drop = True, inplace = False)

        gradients = np.linalg.norm(gradients, axis = 1)

        nexamples = max(
            self.min_samples_per_category, 
            int(round(train_size * gradients.size))
        )

        indices = sorted(np.argpartition(gradients, nexamples)[:nexamples])

        return X.iloc[indices]

    def maximum_gradients(self, X, gradients, train_size):
        """
        Return the instances with the largest magnitude gradients.

        @param X: the data to select from
        @param gradients: the gradients for this dataset
        @param train_size: the number of examples to include        
        """
        # allow for 100% labeled data 
        eps = 1e-6
        if abs(1.0 - train_size) < eps: 
            return X

        # reset the index here on a copy of X so that the indices line up with 
        # the gradients which are index from (0, len(X.index))
        X = X.reset_index(drop = True, inplace = False)

        gradients = np.linalg.norm(gradients, axis = 1)

        nexamples = max(
            self.min_samples_per_category, 
            int(round(train_size * gradients.size))
        )

        indices = sorted(np.argpartition(gradients, -1 * nexamples)[-1 * nexamples:])

        return X.iloc[indices]

    def least_ood(self, X, OODs, train_size):
        """
        Return the least OOD instances. 
        
        @param X: the data to select from
        @param OODs: the OOD samples for this dataset
        @param train_size: the number of examples to include
        """
        # allow for 100% labeled data 
        eps = 1e-6
        if abs(1.0 - train_size) < eps: 
            return X

        # reset the index here on a copy of X so that the indices line up with 
        # the gradients which are index from (0, len(X.index))
        X = X.reset_index(drop = True, inplace = False)

        nexamples = max(
            self.min_samples_per_category, 
            int(round(train_size * OODs.size))
        )

        indices = sorted(np.argpartition(OODs, nexamples)[:nexamples])

        return X.iloc[indices]

    def most_ood(self, X, OODs, train_size):
        """
        Return the least OOD instances. 
        
        @param X: the data to select from
        @param OODs: the OOD samples for this dataset
        @param train_size: the number of examples to include
        """
        # allow for 100% labeled data 
        eps = 1e-6
        if abs(1.0 - train_size) < eps: 
            return X

        # reset the index here on a copy of X so that the indices line up with 
        # the gradients which are index from (0, len(X.index))
        X = X.reset_index(drop = True, inplace = False)

        nexamples = max(
            self.min_samples_per_category, 
            int(round(train_size * OODs.size))
        )

        indices = sorted(np.argpartition(OODs, -1 * nexamples)[-1 * nexamples:])

        return X.iloc[indices]
  
    def generate_umap(self, layer):
        """
        Use the UMAP algorithm to create a 2D embedding of the latent space.
        """
        # save timing statistics
        timing_filename = '{}/OOD-umap-{}-timing.txt'.format(self.temp_directory, layer)
        fd = open(timing_filename, 'w')

        OOD_time = time.time()
        OOD_filename = '{}/OOD-features_2d-{}-umap.npy'.format(self.temp_directory, layer)
        if not os.path.exists(OOD_filename):
            # read the OOD features, reduce to 2D, and save
            features = self.experiment.read_features('OOD', layer, 'test')
            reducer = umap.UMAP(n_neighbors = 15, n_components = 2)
            features_2d = reducer.fit_transform(features)
            np.save(OOD_filename, features_2d)
        fd.write('OOD features shape: {}\n'.format(features.shape))
        fd.write('OOD time: {}\n'.format(time.time() - OOD_time))

        # plot the OOD points
        plt.figure(figsize = (6, 6))
        plt.scatter(features_2d[:,0], features_2d[:,1], color = '#149b5a88', s = 0.5)
        plt.axis('off')
        plt.tight_layout()
        plt.title('OOD UMAP {}'.format(layer))
        output_filename = '{}/OOD-features_2d-{}-umap.png'.format(self.temp_directory, layer)
        plt.savefig(output_filename)
        plt.close()

        # close file 
        fd.close()

    def read_umap_projection(self, label, layer):
        """
        Read the UMAP projection into 2D space for this label and layer.

        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param layer: the extracted layer to read 
        """
        filename = '{}/{}-features_2d-{}-umap.npy'.format(self.temp_directory, label, layer)
        assert (os.path.exists(filename))
        features_2d = np.load(filename)

        return features_2d

    def generate_stratified_samples(self, layer):
        """
        Generate sample points using the stratified sampling heuristic.

        @param layer: the extracted layer to consider
        """
        features_2d = self.read_umap_projection('OOD', layer)

        # cluster the points using a GaussianMixture model
        clustering = GaussianMixture(n_components = 18, n_init = 12, warm_start = True).fit_predict(features_2d)
        clusters, counts = np.unique(clustering, return_counts = True)

        # run for preset training sizes 
        train_sizes = [x / 100 for x in list(range(1, 20)) + list(range(20, 100, 10)) + [100]]

        for train_size in train_sizes:
            # do not overwrite existing point selections
            output_filename = '{}/OOD-stratified_sampling-umap-{}-{:05d}.npy'.format(self.samples_directory, layer, int(round(train_size * 10000)))
            if os.path.exists(output_filename): continue

            # get the indices of points to include for re-training
            indices = []
            for cluster, count in zip(clusters, counts):
                npoints = int(round(train_size * count))

                # sample points without replacement
                points = np.random.choice(np.where(clustering == cluster)[0], size = npoints, replace = False)
                indices += list(points)
            
            # convert to a numpy array and save
            indices = np.array(sorted(indices)).flatten()
            np.save(output_filename, indices)

    def stratified_sampling(self, X, train_size, layer):
        """
        Return the results from the stratified sampling method.

        @param X: the data to select from
        @param train_size: the number of examples to include
        @param layer: the extracted layer to consider
        """
        input_filename = '{}/OOD-stratified_sampling-umap-{}-{:05d}.npy'.format(
            self.samples_directory, 
            layer,
            int(10000 * train_size)
        )
        points = np.load(input_filename)
        
        # reset the index here on a copy of X so that the indices line up with 
        # the gradients which are index from (0, len(X.index))
        X = X.reset_index(drop = True, inplace = False)

        return X.iloc[points]

    def finetune(self, train_size, method = 'random'):
        """
        Finetune the model from this experiment using a fraction of OOD/ID examples.
        
        @param train_size: fraction of OOD examples to use
        @param method: way to select fraction of OOD examples (default = 'random')
        """
        # compile the appropriate model (run here to reset the learning parameters)
        self.experiment.compile_model()
        self.model = self.experiment.model

        # since we are finetuning, we need to start with the original weights
        pretrained_model_filename = '{}/model-opt'.format(self.experiment.model_directory)
        assert (os.path.exists(pretrained_model_filename))
        self.model.load_weights(pretrained_model_filename)

        sampled_data = []

        for _, X in self.ID_train.groupby('packet_category'):
            if method == 'random':
                sampled_data.append(
                    self.random_selection(X, train_size)
                )
        for _, X in self.OOD_train.groupby('packet_category'):
            if method == 'random':
                sampled_data.append(
                    self.random_selection(X, train_size)
                )

        X_train = pd.concat(sampled_data)

        # update the learning rate here for the model by reducing by a factor of 10
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

        model_filename = '{}/model-{}-{:05d}'.format(self.model_directory, method, int(10000 * train_size))
        self.model.train(X_train, nepochs = self.nepochs, model_filename = model_filename)

    def finetune_drift(self, train_size, ID_method = 'random', OOD_method = 'random', layer = None):
        """
        Finetune the model from this experiment using a fraction of OOD/ID examples.
        
        @param train_size: fraction of OOD examples to use
        @param ID_method: way to select fraction of ID examples (default = 'random')
        @param ID_method: way to select fraction of OOD examples (default = 'random')
        @param layer: the layer to take gradients/results from (default = None)
        """
        # compile the appropriate model (run here to reset the learning parameters)
        self.experiment.compile_model()
        self.model = self.experiment.model

        # since we are finetuning, we need to start with the original weights
        pretrained_model_filename = '{}/model-opt'.format(self.experiment.model_directory)
        assert (os.path.exists(pretrained_model_filename))
        self.model.load_weights(pretrained_model_filename)

        sampled_data = []

        for _, X in self.ID_train.groupby('packet_category'):
            if ID_method == 'random':
                sampled_data.append(
                    self.random_selection(X, train_size)
                )
            else:
                assert ('Unknown ID method: {}'.format(ID_method))

        # we need to look at the OOD data together since we are
        # deciding which data to label (cannot go by packet_category)
        X = self.OOD_train

        # as mentioned above, for finetuning we need to use the 'test' dataset as training for OOD
        if not layer is None and 'gradients' in OOD_method:
            OOD_gradients = self.experiment.read_feature_attributions('OOD', 'LGXA', layer, 'test')
        if not layer is None and 'mahalanobis' in OOD_method:
            OOD_values = self.experiment.read_mahalanobis_distances('OOD', layer, 'test')
        if not layer is None and 'normalizing_flows' in OOD_method:
            OOD_values = self.experiment.read_normalizing_flows('OOD', layer, 'test')

        if OOD_method == 'random':
            sampled_data.append(
                self.random_selection(X, train_size)
            )
        elif OOD_method == 'stratified_sampling':
            sampled_data.append(
                self.stratified_sampling(X, train_size, layer)
            )
        elif OOD_method == 'minimum_gradients':
            sampled_data.append(
                self.minimum_gradients(X, OOD_gradients[X.index], train_size)
            )
        elif OOD_method == 'maximum_gradients':
            sampled_data.append(
                self.maximum_gradients(X, OOD_gradients[X.index], train_size)
            )
        # the least and most OOD methods allow for both normalizing flows and mahalanobis distance
        elif 'least_ood' in OOD_method:
            sampled_data.append(
                self.least_ood(X, OOD_values[X.index], train_size)
            )
        elif 'most_ood' in OOD_method:
            sampled_data.append(
                self.most_ood(X, OOD_values[X.index], train_size)
        )
        else: 
            assert ('Unknown OOD method: {}'.format(OOD_method))
            
        X_train = pd.concat(sampled_data)

        # update the learning rate here for the model by reducing by a factor of 10
        for param_group in self.model.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

        if layer is None:
            model_filename = '{}/model-ID-{}-OOD-{}-{:05d}'.format(self.model_directory, ID_method, OOD_method, int(10000 * train_size))
        else:
            model_filename = '{}/model-ID-{}-OOD-{}_{}-{:05d}'.format(self.model_directory, ID_method, OOD_method, layer, int(10000 * train_size))

        self.model.train(X_train, nepochs = self.nepochs, model_filename = model_filename)

    def infer(self, X, label, suffix, fd = sys.stdout):
        """
        Run inference on these examples and save the predictions to disk.

        @param X: the dataset
        @param label: signifier of this set of examples (e.g., ID or OOD)
        @param suffix: signifier of the model parameters
        @param fd: location to write results (default = stdout)
        """
        # run inference for these examples and split 
        outputs = self.model.infer(X)
        
        if self.model.output_shape > 1:
            predictions = np.argmax(outputs, axis = 1)
        else:
            predictions = np.squeeze(outputs) > 0.5
        
        # get the labels for this dataset
        labels = X['target'].values
        
        # compute the accuracy 
        accuracy = sum(predictions == labels) / len(X.index)
        fd.write('{} acc: {:0.8f}\n'.format(label, accuracy))
        
        # write the number of exmaples to disk
        fd.write('No. {} ex.: {}\n'.format(label, len(X.index)))

        print ('{} acc: {:0.8f}'.format(label, accuracy))
        print ('No. {} ex.: {}'.format(label, len(X.index)))
        
        # save the predictions and outputs, moved here so that testing
        # for predictions filename means inference completed successfully
        outputs_filename = '{}/{}_test-outputs-{}.npy'.format(self.results_directory, label, suffix)
        np.save(outputs_filename, outputs)
        predictions_filename = '{}/{}_test-predictions-{}.npy'.format(self.results_directory, label, suffix)
        np.save(predictions_filename, predictions)

    def run_inference(self, train_size, method = 'random', epoch = 'opt'):
        """
        Perform inference on the finetuned model.

        @param train_size: fraction of OOD examples to use
        @param method: way to select fraction of OOD examples (default = 'random')
        @param epoch: the epoch to load from (default = 'opt')
        """
        # compile the appropriate model (run here to reset the learning parameters)
        self.experiment.compile_model()
        self.model = self.experiment.model

        suffix = '{}-{:05d}-{}'.format(method, int(10000 * train_size), epoch)

        # load the model weights for this epoch
        model_filename = '{}/model-{}'.format(self.model_directory, suffix)
        self.model.load_weights(model_filename)

        # open a file to write results 
        results_filename = '{}/inference-{}.txt'.format(self.results_directory, suffix)
        fd = open(results_filename, 'w')
        
        # get the results for both in-distribution and out-of-distribution examples 
        self.infer(self.ID_test, 'ID', suffix, fd)
        self.infer(self.OOD_test, 'OOD', suffix, fd)

        # close the open file 
        fd.close()

    def run_inference_drift(self, train_size, ID_method = 'random', OOD_method = 'random', layer = None, epoch = 'opt', ):
        """
        Perform inference on the finetuned model.

        @param train_size: fraction of OOD examples to use
        @param ID_method: way to select fraction of ID examples (default = 'random')
        @param ID_method: way to select fraction of OOD examples (default = 'random')
        @param layer: the layer to take gradients/results from (default = None)
        @param epoch: the epoch to load from (default = 'opt')
        """
        # compile the appropriate model (run here to reset the learning parameters)
        self.experiment.compile_model()
        self.model = self.experiment.model

        if layer is None:
            suffix = 'ID-{}-OOD-{}-{:05d}-{}'.format(ID_method, OOD_method, int(10000 * train_size), epoch)
        else:
            suffix = 'ID-{}-OOD-{}_{}-{:05d}-{}'.format(ID_method, OOD_method, layer, int(10000 * train_size), epoch)

        # load the model weights for this epoch
        model_filename = '{}/model-{}'.format(self.model_directory, suffix)
        self.model.load_weights(model_filename)
        
        # open a file to write results 
        results_filename = '{}/inference-{}.txt'.format(self.results_directory, suffix)
        fd = open(results_filename, 'w')
        
        # get the results for both in-distribution and out-of-distribution examples 
        self.infer(self.ID_test, 'ID', suffix, fd)
        self.infer(self.OOD_test, 'OOD', suffix, fd)

        # close the open file 
        fd.close()