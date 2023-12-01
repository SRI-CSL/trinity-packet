import torch 



import torch.nn as nn 



class ContextCNNBatchNormDropout(nn.Module):
    def __init__(self, header_length, payload_length, output_shape):
        """
        Constructor for the CNN class with header context, 
        with batch normalization and drop out.

        @param header_length: the number of header attributes for context 
        @param payload_length: the number of bytes to include for the payload
        @param output_shape: the number of layers in the output 
        """
        # run constructor for inherited class
        super(ContextCNNBatchNormDropout, self).__init__()
        
        self.header_length = header_length
        self.payload_length = payload_length
        self.output_shape = output_shape
        self.batch_size = 1024
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 16),
            nn.Conv1d(16, 16, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 16),
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.BatchNorm1d(num_features = 16),
            nn.Dropout(p = 0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 32),
            nn.Conv1d(32, 32, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 32),
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.BatchNorm1d(num_features = 32),
            nn.Dropout(p = 0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 64),
            nn.Conv1d(64, 64, 3, padding = 1),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 64),
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.BatchNorm1d(num_features = 64),
            nn.Dropout(p = 0.2),
        )
        self.flatten = nn.Flatten()
        self.dense1 = nn.Sequential(
            # take integer division to account for potential rounding in pooling
            # add the header here since it is different from the bytes (not usable with CNN) 
            nn.Linear(self.payload_length // 8 * 64 + self.header_length, 256),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 256),
            nn.Dropout(p = 0.2),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 128),
            nn.Dropout(p = 0.2),
        )
        self.dense3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 64),
            nn.Dropout(p = 0.2),
        )
        self.output = nn.Sequential(
            nn.Linear(64, self.output_shape),
            # remove the sigmoid activation to move to categorical cross-entropy
            # binary cases will use BCELossWithLogits
        )

        # set global parameters for the generator 
        self.cnn = True
        self.vocab = False
        self.learning_rate = 1e-4
        self.extractable_layers = {
            'dense1': self.dense1,
            'dense2': self.dense2,
            'dense3': self.dense3,
        }

    def forward(self, x, extract = None):
        """
        Define the forward operation for an input sample x.

        @param x: the tensor to conduct the forward operation on
        @param extract: layer to extract if not None (default = None)
        """
        x_header = torch.squeeze(x[:,:,:self.header_length])
        x_payload = x[:,:,self.header_length:]

        # pass the input through each defined layer 
        # only payload goes through convolution layers
        x = self.conv1(x_payload)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        # concatenate the header to the flattened CNN data
        x = torch.cat([x_header, x], axis = 1)
        x = self.dense1(x)
        if extract == 'dense1':
            return x
        x = self.dense2(x)
        if extract == 'dense2':
            return x
        x = self.dense3(x)
        if extract == 'dense3':
            return x
        x = self.output(x)

        # if no layer has been returned, assert none were given 
        assert (extract is None)

        return x