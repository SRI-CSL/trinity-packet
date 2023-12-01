import torch.nn as nn 



class ContextFNNBatchNormDropout(nn.Module):
    def __init__(self, header_length, payload_length, output_shape):
        """
        Constructor for the FNN class with header context, 
        with batch normalization and drop out.

        @param header_length: the number of header attributes for context 
        @param payload_length: the number of bytes to include for the payload
        @param output_shape: the number of layers in the output 
        """
        # run constructor for inherited class
        super(ContextFNNBatchNormDropout, self).__init__()
        
        self.header_length = header_length
        self.payload_length = payload_length
        self.output_shape = output_shape
        self.batch_size = 1024
        
        self.linear1 = nn.Sequential(
            nn.Linear(self.header_length + self.payload_length, 1024),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 1024),
            nn.Dropout(p = 0.2),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 512),
            nn.Dropout(p = 0.2),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 256),
            nn.Dropout(p = 0.2),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 128),
            nn.Dropout(p = 0.2),
        )
        self.linear5 = nn.Sequential(
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
        self.cnn = False
        self.vocab = False
        self.learning_rate = 1e-4
        self.extractable_layers = {
            'linear3': self.linear3,
            'linear4': self.linear4,
            'linear5': self.linear5,
        }

    def forward(self, x, extract = None):
        """
        Define the forward operation for an input sample x.

        @param x: the tensor to conduct the forward operation on
        @param extract: layer to extract if not None (default = None) 
        """
        # pass the input through each defined layer 
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        if extract == 'linear3':
            return x
        x = self.linear4(x)
        if extract == 'linear4': 
            return x
        x = self.linear5(x)
        if extract == 'linear5':
            return x
        x = self.output(x)

        # if no layer has been returned, assert none were given 
        assert (extract is None)

        return x 