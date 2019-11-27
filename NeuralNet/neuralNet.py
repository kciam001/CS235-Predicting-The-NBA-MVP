import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from DataSetCreator import NBADataset, SplitDataSet
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

def evaluate(model, validDataSet):
    correct = 0
    total = 0
    #validate the model on validationSet
    with torch.no_grad():
        for i, data in enumerate(validDataSet):
            batch,labels = data
            labels = labels.view(-1,1)
            output = model(batch)
            for index, val in enumerate(output):
                if (abs(val - labels[index])) < .15:
                    correct +=1
                total +=1 
    return float(correct)/total

def train(numFeatures, numEpochs, dataSet, validDataSet, path, showPlot, saveModel):
    model = Net(numFeatures)
    checkpoint = 50
    #learning rate
    learningRate = 0.001
    #optimzer + loss function
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    lossFunction = nn.MSELoss(reduction='sum')
    losses = []

    #training loop
    for epoch in range(numEpochs):
        runningLoss = 0.0
        for i, data in enumerate(dataSet):
            #get the batch + labels
            batch,labels = data
            labels = labels.view(-1,1)
            #zero out the gadients
            optimizer.zero_grad()
            #forward + optimization
            output = model(batch)
            loss = torch.sqrt(lossFunction(output, labels.float()))
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()
        #calculate the values
        epochLoss = runningLoss / len(dataSet)
        losses.append(epochLoss)
        if epoch % checkpoint == checkpoint - 1:
            accuracy = evaluate(model, validDataSet)
            print("Epoch {0}, Loss: {1:.5f}, {2:.5f}".format(epoch+1,epochLoss,accuracy) )
    
    if saveModel:
        torch.save(model.state_dict(),path)
    if showPlot:
        plt.plot(np.array(losses), 'r')
        plt.show()

def loadModel(numFeatures,path):
    model = Net(numFeatures)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def main():
    #Training variables
    EPOCH = 1000
    BATCH_SIZE = 50
    #Validation split variable
    VALIDATION_SPLIT = .2

    #parser
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    parser.add_argument('--train', action='store_true', help="training mode")
    parser.add_argument('--eval', action='store_true', help="evaluation mode")
    parser.add_argument('--plot', action='store_true', help="plot metrics after training model")
    parser.add_argument('--save', action='store_true', help="save the model weights after training")
    parser.add_argument('-w','--weightsPath', help="path that will save/load weights")
    parser.add_argument('-d', '--dataPath',required=True,help="name of csv file(minus .csv) that will be loaded. Must be located in ./Data folder")
    args = parser.parse_args()

    #features and label information
    dataInfo = {
        'features':['mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws']
        , 'label':'award_share'}
    numFeatures = len(dataInfo['features']) 

    WEIGHTS_PATH = None
    #PATH to save or load weights 
    if args.weightsPath:
        WEIGHTS_PATH = args.weightsPath
    else:
        WEIGHTS_PATH = './NBA_net.pth'

    #dataset to be loaded    
    dataPath = "../Data/" + args.dataPath + '.csv'

    #train the model
    if args.train:
        #load the training dataset
        trainDataset = NBADataset(dataPath, dataInfo)
        #split the data
        trainSet, validSet = SplitDataSet(trainDataset,VALIDATION_SPLIT)
        #create training Dataloader
        trainDataLoader = DataLoader(trainSet, batch_size=BATCH_SIZE)
        #create the validation Dataloader
        validDataLoader = DataLoader(validSet, batch_size=BATCH_SIZE)
        #create a model
        train(numFeatures, EPOCH, trainDataLoader, validDataLoader, WEIGHTS_PATH, args.plot, args.save)

    #eval the model
    elif args.eval:
        net = loadModel(numFeatures,WEIGHTS_PATH)
        #load in the examples
        evalData = NBADataset(dataPath, dataInfo)
        #predict output for each example
        for i, data in enumerate(evalData):
            batch, _ = data
            output = net(batch)
            print("Player: {0} - {1:4f}".format(evalData.getPlayerName(i), output.item() ))

if __name__ == '__main__':
    main()