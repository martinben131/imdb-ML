#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""


import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
import re
import numpy as np
from random import randrange

device = torch.device('cuda:0')
###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    rev = " ".join(sample)
    rev = rev.replace('\'', '')
    rev = re.sub(r"</?\w+[^>]*>", '', rev)
    rev = re.sub(r"[^a-zA-Z']", ' ', rev)
    final = rev.split()
    final = [i for i in final if len(i) > 1]
    return final

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    # Convert the datasetLabel to longs() as is required by the tnn.CrossEntropyLoss() function 
    datasetLabel = datasetLabel - 1
    return datasetLabel.long()

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    # Take the maximum of our 5 output values from our network
    pred = netOutput.argmax(dim=1, keepdim=True)
    # Add a value of one since ratings go from 1-5 not 0-4
    pred = torch.add(pred, 1)
    return pred.float()

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()
        self.dropout_prob = 0.5
        self.input_dim = 256
        self.hidden_dim = 128
        self.hidden_dim_linear = 200
        self.lstm_dim = 128
        self.output_size = 5

        self.lstm = tnn.LSTM(
            input_size=2*self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bias=True,
            dropout=self.dropout_prob,
            num_layers=4,
            bidirectional=True)

        self.fc = tnn.Sequential(
            tnn.Linear(self.hidden_dim*2, self.hidden_dim_linear),
            tnn.BatchNorm1d(self.hidden_dim_linear),
            tnn.ReLU(inplace=True),
            tnn.Linear(self.hidden_dim_linear, self.output_size)
        )

        self.gru = tnn.GRU(
            input_size=50,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
            dropout=0.5
        )

        self.dropout = tnn.Dropout(p=self.dropout_prob)
    def forward(self, input, length):
        # Pass input directly in gru 
        gru_out, hid = self.gru(input)
        # Apply a transpose to allow for maxpooling 
        # maxpool = torch.transpose(gru_out,1,2).contiguous()
        # Apply adaptive max pooling to 
        # maxpool = F.adaptive_max_pool1d(maxpool, gru_out.size(2)).squeeze(2)
        # Pass this maxpooled tensor into our LSTM 
        packed_output, (hidden, cell) = self.lstm(gru_out)
        output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        # Apply the dropout and fully connected layer 
        output = self.fc(self.dropout(output))
        return output


# class loss(tnn.Module):
#     """
#     Class for creating a custom loss function, if desired.
#     You may remove/comment out this class if you are not using it.
#     """
#
#     def __init__(self):
#         super(loss, self).__init__()
#
#     def forward(self, output, target):



lossFunc = tnn.CrossEntropyLoss()
net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.001)
