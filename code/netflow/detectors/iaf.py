import time 



import numpy as np
import torch 
import torch.nn as nn 



import FrEIA.framework as Ff
import FrEIA.modules as Fm



import matplotlib.pyplot as plt 



from torch.utils.data import Dataset, DataLoader



from sklearn.model_selection import train_test_split



class IAFDataset(Dataset):
    def __init__(self, data):
        """
        Simple Dataset class that allows for batch learning on IAF. 

        @param data: the data to train/infer a normalizing flow on 
        """
        self.data = data

    def __len__(self):
        """
        Dataset required function that returns the number of samples 
        in the dataset.
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Return a single item from the dataset. 

        @param idx: the index of the item to return
        """
        # return the payload and 0 for a non existent label
        return self.data[idx,:], 0

def subnet_fc(dims_in, dims_out):
    return nn.Sequential(
        nn.Linear(dims_in, 128), 
        nn.LeakyReLU(negative_slope = 0.01), 
        nn.Linear(128, 128),
        nn.LeakyReLU(negative_slope = 0.01),
        nn.Linear(128, dims_out),
    )

class InverseAutoregressiveFlow(object):
    def __init__(self, input_shape, gpu = 4, nblocks = 20, affine_clamping = 2.0):
        """
        Initialize an Inverse Autoregressive flow for the model safeguard.

        @param input_shape: the dimensionality of the feature vector 
        @parma gpu: the gpu to run the experiment on (default = 4)
        @param nblocks: the number of blocks in the flow (default = 20)
        @param affine_clamping: the clamping parameter (default = 2.0)
        """
        self.input_shape = input_shape 
        # get the device 
        self.gpu = gpu
        self.device = torch.device('cuda:{}'.format(self.gpu))

        # create the normalizing flow network         
        self.model = Ff.SequenceINN(
            self.input_shape,
        )
        
        # add learnable blocks 
        nblocks = nblocks
        for _ in range(nblocks):
            self.model.append(
                Fm.AllInOneBlock,
                subnet_constructor = subnet_fc, 
                permute_soft = True,
                affine_clamping = affine_clamping,
                reverse_permutation = True,
            )

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr = 1e-4, 
            betas = [0.8, 0.9],
            weight_decay = 2e-5,
        )

    def train(self, X_train, model_filename, loss_filename = None, label = None):
        """
        Train a normalizing flow model on the provided data. 

        @param X_train: the training data (dictionary) with features and attributes
        @param model_filename: path to save model weights
        @param loss_filename: path to save training losses 
        @param label: which category (attack or benign)
        """
        print ('Training {}'.format(model_filename))
        
        # create a deterministic validation set from the training data
        X_train, X_val = train_test_split(
            X_train['features'], 
            test_size = 0.25, 
            stratify = X_train['stratify'],
            random_state = 0,
        )

        # use a large batch size for quicker training
        BATCH_SIZE = 131072
        # set number of training epochs
        nepochs = 512

        # create a dataloader for the training data 
        # get the features attribute and extract values for numpy array
        training_data = IAFDataset(X_train)
        training_dataloader = DataLoader(
            training_data, 
            batch_size = BATCH_SIZE, 
            shuffle = True, 
            num_workers = 8,
        )

        validation_data = IAFDataset(X_val)
        validation_dataloader = DataLoader(
            validation_data, 
            batch_size = BATCH_SIZE,
            shuffle = True, 
            num_workers = 8,
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

            # divide the training data into large batch sizes
            for index, (inputs, _) in enumerate(training_dataloader):
                # put on correct device 
                inputs = inputs.to(self.device)
                # add in gaussian noise to create more stable training 
                inputs = inputs + torch.randn(inputs.size(), device = self.device) * 0.05

                # zero the parameter gradients 
                self.optimizer.zero_grad()

                # pass to model and get transformed variable z and log Jacobian determinant
                z, log_jac_det = self.model(inputs)

                # calculate the negative log-likelihood of the model 
                loss = 0.5 * torch.sum(z ** 2, 1) - log_jac_det 
                loss = loss.mean() / self.input_shape
                # backpropagate and update the weights 
                loss.backward()

                self.optimizer.step()
                
                # keep track of running loss
                running_train_loss += loss.item()

            # normalize to make comparable to validation 
            running_train_loss = running_train_loss / (index + 1)
            # keep track of running losses
            running_train_losses.append(running_train_loss)

            # keep a tally of the loss 
            running_validation_loss = 0.0

            # set for validation 
            self.model.eval()

            # get the input and labels for validation separately 
            for index, (inputs, _) in enumerate(validation_dataloader):
                # put on correct device 
                inputs = inputs.to(self.device)
                # add in gaussian noise to create more stable training 
                inputs = inputs + torch.randn(inputs.size(), device = self.device) * 0.05

                # pass to model and get transformed variable z and log Jacobian determinant
                z, log_jac_det = self.model(inputs)
                
                # calculate the negative log-likelihood of the model 
                loss = 0.5 * torch.sum(z ** 2, 1) - log_jac_det 
                loss = loss.mean() / self.input_shape

                # keep track of running loss
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

        # save the best weights after all epochs to decrease training time
        torch.save(opt_weights, model_filename)

        if not loss_filename is None:
            plt.plot(
                running_train_losses, 
                label = 'Train Loss {} (min: {:0.4f})'.format(label, min(running_train_losses))
            )
            plt.plot(
                running_validation_losses, 
                label = 'Validation Loss {} (min: {:0.4f})'.format(label, min(running_train_losses))
            )
            # add the labels
            plt.title('Training Loss')
            plt.ylabel('Negative Log-Likelihood')
            plt.xlabel('Epoch')
            # plot the legend 
            plt.legend()
            # save the figure (gets overwritten each time)
            plt.savefig(loss_filename)

        # save the losses 
        train_loss_filename = '{}-training_losses.npy'.format(model_filename)
        np.save(train_loss_filename, np.array(running_train_losses))
        validation_loss_filename = '{}-validation_losses.npy'.format(model_filename)
        np.save(validation_loss_filename, np.array(running_validation_losses))

    def load_weights(self, model_filename):
        """
        Load pre-trained weights into the model.

        @param model_filename: path to model weights 
        """
        self.model.load_state_dict(torch.load(model_filename, map_location = self.device))

    def inverse(self, X_test):
        """
        Infer on the normalizing flow model on the provided data.

        @param X_train: the training data
        """
        # set model to evaluation mode
        self.model.eval()

        # create a dataloader for the training data 
        testing_data = IAFDataset(X_test)
        testing_dataloader = DataLoader(
            testing_data, 
            batch_size = 131072, 
            shuffle = False, 
            num_workers = 8,
        )

        # create an array of the testing losses (to concatenate later)
        testing_losses = []

        # run inference 
        with torch.no_grad():
            # divide the training data into large batch sizes
            for inputs, _ in testing_dataloader:
                # put on correct device 
                inputs = inputs.to(self.device)
                # add in gaussian noise to create more stable training 
                inputs = inputs + torch.randn(inputs.size(), device = self.device) * 0.05

                z, log_jac_det = self.model(inputs)        
                losses = 0.5 * torch.sum(z ** 2, 1) - log_jac_det 

                testing_losses.append(losses.detach().cpu().numpy())

        # return the losses, note that these do not correspond to 
        # probabilities or predictions but the loss if we were to model
        # the distribution. Thus, lower loss values are better 
        return np.concatenate(testing_losses)