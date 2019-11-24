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

    EPOCH = 100
    BATCH_SIZE = 2
    dataInfo = {'features':['fga','fg3a'], 'label':'award_share'}

    train = NBADataset('dummyData.csv', dataInfo)
    trainDataset = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    test = NBADataset('dummyData.csv', dataInfo)
    testDataset = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    numFeatures = len(dataInfo['features'])
    net = Net(numFeatures=numFeatures)

    #optimzer function
    optimizer = optim.SGD(net.parameters(), lr=.001)
    #loss function
    lossFunction = nn.MSELoss(reduction='sum')
    
    for epoch in range(EPOCH):
        for batch, labels in trainDataset:
            labels = labels.view(BATCH_SIZE,1)
            net.zero_grad()
            output = net(batch)
            loss = lossFunction(output, labels.float())
            loss.backward()
            optimizer.step()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, labels in testDataset:
                labels = labels.view(BATCH_SIZE,1)
                output = net(batch) 
                for idx, i in enumerate(output):
                    if abs(i - labels[idx]) < .1 :
                        correct +=1
                    total += 1
        print("Accuracy: {}".format(round(correct/total,3)))      


if __name__ == '__main__':
    main()