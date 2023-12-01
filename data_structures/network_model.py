import time
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader



import numpy as np
import matplotlib.pyplot as plt
# create a better style for matplotlib 
plt.style.use('seaborn-dark')



from sklearn.model_selection import train_test_split



class NetworkModel(object):
    def __init__(
                self, 
                header_length,
                payload_length,
                output_shape, 
                network,
                dataset,
                gpu = 4):
        """
        Constructor for the Network Traffic Model class. 

        @param header_length: the number of header attributes for context 
        @param payload_length: the number of bytes to include for the payload
        @param output_shape: the number of layers in the output 
        @param network: a class corresponding to the neural network architecture 
        @param dataset: the torch dataset to use 
        @param gpu: the gpu to use for training/inference (default = 4)
        """
        # set the input shape instance variable
        self.header_length = header_length
        self.payload_length = payload_length
        self.output_shape = output_shape

        # create the model with the specified input/output shapes 
        self.model = network(self.header_length, self.payload_length, self.output_shape)
        self.dataset = dataset

        # compile the model and choose loss function based on number of output nodes
        if self.output_shape == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = self.model.learning_rate,
            betas = [0.9, 0.999],
            amsgrad = True,
        )

        # restrict tensorflow to one gpu
        self.device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

        # put the model onto the proper device 
        self.model.to(self.device)

    def train(self, X_train, nepochs = 20, model_filename = None):
        """
        Train the model given input data. 

        @param X_train: the training data
        @param epochs: the number of epochs to run (default = 20)
        @param model_filename: path to save model weights (default = None)
        """
        # create a deterministic validation set from the training data 
        X_train, X_val = train_test_split(
                    X_train, 
                    test_size = 0.25, 
                    stratify = X_train.packet_category, 
                    random_state = 0,
                )

        # create a new training dataset 
        training_data = self.dataset(
            X_train, 
            self.header_length,
            self.payload_length,
            self.output_shape,
            to_fit = True,
            cnn = self.model.cnn,
            vocab = self.model.vocab,
        )

        validation_data = self.dataset(
            X_val, 
            self.header_length, 
            self.payload_length, 
            self.output_shape, 
            # need to include to_fit here, otherwise the dataset can 
            # return any value for the labels (to_fit used for data without labels)
            # for validation, we need accurate labels. 
            to_fit = True, 
            cnn = self.model.cnn, 
            vocab = self.model.vocab,
        )

        # create a dataloader for the dataset
        training_dataloader = DataLoader(
            training_data, 
            batch_size = self.model.batch_size, 
            shuffle = True, 
            num_workers = 4,
        )

        validation_dataloader = DataLoader(
            validation_data, 
            batch_size = self.model.batch_size, 
            # no need to shuffle validation data
            shuffle = False, 
            num_workers = 4,
        )

        # keep track of the weights with the lowest validation loss
        best_validation_loss = np.inf 

        running_train_losses = []
        running_validation_losses = []
        for epoch in range(nepochs):
            # keep a tally of the the loss and time 
            running_train_loss = 0.0
            start_time = time.time()

            # set for training 
            self.model.train()

            # get the inputs and labels for training seprately
            for index, (inputs, targets) in enumerate(training_dataloader):
                # load the inputs and labels to the device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # zero the parameter gradients 
                self.optimizer.zero_grad()

                # forward pass, backward pass, optimize 
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # keep track of running loss
                running_train_loss += loss.item()

            # normalize to make comparable to validation 
            running_train_loss = running_train_loss / (index + 1)
            # keep track of running losses
            running_train_losses.append(running_train_loss)

            # save every 2nd epoch model weight
            if not (epoch + 1) % 2 and not model_filename is None:
                torch.save(self.model.state_dict(), '{}-{:03d}'.format(model_filename, epoch + 1))

            # keep a tally of the loss 
            running_validation_loss = 0.0

            # set for validation 
            self.model.eval()

            # get the input and labels for validation separately 
            for index, (inputs, targets) in enumerate(validation_dataloader):
                # load the inputs and labels to the device 
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # forward pass only 
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # keep track of the running loss
                running_validation_loss += loss.item()

            # normalize to make comparable to training 
            running_validation_loss = running_validation_loss / (index + 1)
            # keep track of running losses
            running_validation_losses.append(running_validation_loss)

            # save if this is the best loss 
            if running_validation_loss < best_validation_loss:
                best_validation_loss = running_validation_loss
                # save the state_dict into a variable to decrease training time
                opt_weights = self.model.state_dict()
            
            # print statistics for this epoch 
            print ('Epoch {}/{} - train loss: {:0.4f} - val loss: {:0.4f} - time: {:0.2f}s'.format(
                epoch + 1, 
                nepochs, 
                running_train_losses[-1],
                running_validation_losses[-1],
                time.time() - start_time,
            ))

        # save the best weights after all epochs to decrease storage costs         
        opt_filename = '{}-opt'.format(model_filename)
        torch.save(opt_weights, opt_filename)

        if not model_filename is None:
            # save the loss filename 
            train_loss_filename = '{}-training_losses.npy'.format(model_filename)
            np.save(train_loss_filename, np.array(running_train_losses))
            validation_loss_filename = '{}-validation_losses.npy'.format(model_filename)
            np.save(validation_loss_filename, np.array(running_validation_losses))

            # plot the training loss 
            plt.plot(running_train_losses, label = 'train loss')
            plt.plot(running_validation_losses, label = 'val loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig('{}.png'.format(model_filename))
            plt.close()

    def load_weights(self, model_filename):
        """
        Load pre-trained weights into the model.

        @param model_filename: path to model weights 
        """
        self.model.load_state_dict(torch.load(model_filename, map_location = self.device))

    def infer(self, X_test, layer = None):
        """
        Run inference on the test dataset and return predictions.

        @param X_test: the dataset to run inference on
        @param layer: the layer of the network to extract (default = None, i.e., model inference)
        """
        # set model to evaluation mode 
        self.model.eval()
        
        # create a new testing dataset 
        testing_data = self.dataset(
            X_test, 
            self.header_length,
            self.payload_length,
            self.output_shape,
            to_fit = False,
            cnn = self.model.cnn,
            vocab = self.model.vocab,
        )
        
        # create a dataloader for the dataset
        testing_dataloader = DataLoader(
                                testing_data, 
                                batch_size = self.model.batch_size, 
                                shuffle = False, 
                                num_workers = 1
                            )
        
        # create a list of features (probabilities if extract = None) 
        features = []

        # get the inputs seprately
        for inputs, _ in testing_dataloader:
            # load the inputs to the device
            inputs = inputs.to(self.device)

            # run the forward pass
            outputs = self.model(inputs, extract = layer)
            features.append(outputs.detach().cpu().numpy())
            
               
        # features will be the output probabilities if layer = None which 
        # is the default for calls to self.infer(X_test); for calls to 
        # extract_features, self.infer(X_test, layer) will return the output
        # of a given layer. extract_features function left for simpler naming
        # convention and backwards compatability
        return np.concatenate(features)

    def extract_features(self, X_test, layer):
        """
        Return weights on inference at intermediate layer.

        @param X_test: the dataset to run inference on 
        @param layer: the layer of the network to extract
        """
        # call inference with the layer which returns extracted features
        return self.infer(X_test, layer)