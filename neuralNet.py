import csv
import numpy as np
import pickle as pikl
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from DataSetCreator import NBADataset
from matplotlib import pyplot as plt
import argparse

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

def train(model, epoch, batchSize, dataInfo, dataSet, validDataSet, path, show):
    #optimzer function
    optimizer = optim.SGD(model.parameters(), lr=.001)
    #loss function
    lossFunction = nn.MSELoss()
    losses = []
    #training loop
    for e in range(epoch):
        running_loss = 0.0
        for i, data in enumerate(dataSet):
            #get the batch + labels
            batch,labels = data
            labels = labels.view(-1,1)
            #zero out the gadients
            optimizer.zero_grad()
            #forward + optimization
            output = model(batch)
            loss = lossFunction(output, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        #calculate the values
        epoch_loss = running_loss / len(dataSet)
        losses.append(epoch_loss)
        #eval model and print statistics
        if e % 1000 ==  999:
            #evaluate
            correct = 0
            total = 0
            #validate the model on validationSet
            with torch.no_grad():
                for i, data in enumerate(validDataSet):
                    batch,labels = data
                    labels = labels.view(-1,1)
                    output = model(batch)
                    for index, val in enumerate(output):
                        if (abs(val - labels[index])) < .2:
                            correct +=1
                        total +=1 
            print("epoch {0}: loss: {1}, accuracy: {2} ".format(e,round(epoch_loss,5),round(correct/total, 5)))

    #torch.save(model.state_dict(),path)
    if show:
        plt.plot(np.array(losses), 'r')
        plt.show()

def loadModel(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def SplitDataSet(dataset, split):
    testSize = int(split * len(dataset))
    trainSize = len(dataset) - testSize
    return random_split(dataset, [trainSize, testSize])

def main():
    #Training variables
    EPOCH = 10000
    BATCH_SIZE = 40
    #Validation split variable
    VALIDATION_SPLIT = .2

    #parser
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    #features and label information
    dataInfo = {
        'features':['g', 'mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws']
        , 'label':'award_share'}
    numFeatures = len(dataInfo['features']) 

    #load the training dataset
    trainDataset = NBADataset('data/train_data(feature reduced).csv', dataInfo)
    #split the data
    trainSet, validSet = SplitDataSet(trainDataset,VALIDATION_SPLIT)
    #create training Dataloader
    trainDataLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)
    #create the validation Dataloader
    validDataLoader = DataLoader(validSet, batch_size=BATCH_SIZE, shuffle=True)

    #create the model
    net = Net(numFeatures=numFeatures)

    #PATH to save or load weights 
    PATH = './NBA_net.pth'

    #train the model
    if args.train:
        train(net, EPOCH, BATCH_SIZE, dataInfo, trainDataLoader, validDataLoader, PATH, args.show)

    #eval the model
    elif args.eval:
        net = loadModel(net,PATH)


if __name__ == '__main__':
    main()