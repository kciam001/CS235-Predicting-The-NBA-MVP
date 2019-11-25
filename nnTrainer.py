import csv
import numpy as np
import pickle as pikl
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from DataSetCreator import NBADataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Net(nn.Module):
    def __init__(self, numFeatures):
        super().__init__()
        #Linear(input output)
        self.fc1 = nn.Linear(numFeatures, 15)
        self.fc2 = nn.Linear(15, 15)
        self.output = nn.Linear(15, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        input = self.activation(self.fc1(input))
        input = self.activation(self.fc2(input))
        output = self.output(input)
        return output

def main():

    EPOCH = 10000
    BATCH_SIZE = 20
    dataInfo = {
        'features':['g', 'mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws', 'votes_first', 'points_won', 'points_max']
        , 'label':'award_share'}

    train = NBADataset('data/train_data(feature reduced).csv', dataInfo)
    trainDataset = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    numFeatures = len(dataInfo['features'])
    net = Net(numFeatures=numFeatures)

    #optimzer function
    optimizer = optim.SGD(net.parameters(), lr=.01)
    #loss function
    lossFunction = nn.MSELoss(reduction='sum')
    
    for epoch in range(EPOCH):
        for batch, labels in trainDataset:
            labels = labels.view(-1,1)
            net.zero_grad()
            output = net(batch)
            loss = lossFunction(output, labels.float())
            loss.backward()
            optimizer.step()
        print(loss)


if __name__ == '__main__':
    main()