import torch
import torch.nn as nn
from torch.nn import functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        """
        Constructor for a single head of self-attention.
        """
        super(SingleHeadAttention, self).__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Define the forward operation for an input sample x. 

        @param x: the tensor to conduct the forward operation on
        """
        _, _, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores 
        weights = q @ k.transpose(-2, -1) * (C ** -0.5)
        weights = F.softmax(weights, dim = -1)
        weights = self.dropout(weights)
        # perform the weighted aggregation of values 
        v = self.value(x)

        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        """
        Constructor for multiple heads of self-attention.
        """
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Define the forward operation for an input sample x. 

        @param x: the tensor to conduct the forward operation on
        """
        out = torch.cat([h(x) for h in self.heads], dim = -1) # (B, T, C)
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        """
        Constructor for simple feed forward network.
        """
        super(FeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Define the forward operation for an input sample x. 

        @param x: the tensor to conduct the forward operation on
        """
        return self.network(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        """
        Constructor for a single transformer block.
        """
        super(Block, self).__init__()
        head_size = n_embd // n_head 
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.self_attention = MultiHeadAttention(n_head, head_size, n_embd, dropout)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.feedforward = FeedForward(n_embd, dropout)

    def forward(self, x):
        """
        Define the forward operation for an input sample x. 

        @param x: the tensor to conduct the forward operation on
        """
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feedforward(self.layer_norm2(x))

        return x

class ContextTransformer(nn.Module):
    def __init__(self, header_length, payload_length, output_shape):
        """
        Constructor for the Transformer class with header context.

        @param header_length: the number of header attributes for context
        @param payload_length: the number of bytes to include for the payload
        @param output_shape: the number of layers in the output
        """
        # run constructor for inherited class
        super(ContextTransformer, self).__init__()

        self.header_length = header_length
        self.payload_length = payload_length
        self.output_shape = output_shape
        self.batch_size = 96

        vocab_size = 256
        block_size = 1500
        n_embd = 384
        n_head = 6 
        n_layer = 2
        dropout = 0.2

        # each token directly reads off the logits for the next token from a lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.linear_norm = nn.LayerNorm(n_embd)
        
        # feed forward layers for classification 
        self.linear1 = nn.Sequential(
            nn.Linear(n_embd + self.header_length, 256),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 256),
            nn.Dropout(p = 0.2),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.BatchNorm1d(num_features = 128),
            nn.Dropout(p = 0.2),
        )
        self.linear3 = nn.Sequential(
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
        self.vocab = True
        self.learning_rate = 3e-4
        self.extractable_layers = {
            'linear1': self.linear1,
            'linear2': self.linear2,
            'linear3': self.linear3,
        }

    def forward(self, x, extract = None):
        """
        Define the forward operation for an input sample x.

        @param x: the tensor to conduct the forward operation on
        @param extract: layer to extract if not None (default = None) 
        """
        x_header = x[:,:self.header_length]
        x_payload = x[:,self.header_length:]

        # get number of batches and tokens
        _, T = x_payload.shape 
        # force x_payload to have type int
        # use this function to keep on same device
        x_payload = x_payload.int()

        token_embedding = self.token_embedding_table(x_payload)
        position_embedding = self.position_embedding_table(torch.arange(T, device = x_payload.device))
        x = token_embedding + position_embedding

        x = self.blocks(x)
        x = self.linear_norm(x)
        
        # use mean pooling to create a sentence embedding 
        x = x.mean(dim = 1)

        x = torch.cat([x_header, x], axis = 1)
        x = self.linear1(x)
        if extract == 'linear1':
            return x
        x = self.linear2(x)
        if extract == 'linear2':
            return x
        x = self.linear3(x)
        if extract == 'linear3':
            return x 
        x = self.output(x)

        # if no layer has been returned, assert none were given 
        assert (extract is None)

        return x