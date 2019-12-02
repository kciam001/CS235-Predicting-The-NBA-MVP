import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from NBADataset import NBADataset, SplitDataSet, PrintDataSet, KFoldCross
from matplotlib import pyplot as plt
import argparse

class Net(nn.Module):
    def __init__(self, numFeatures):
        super().__init__()
        self.fc1 = nn.Linear(numFeatures, 10)
        self.fc2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        input = self.activation(self.fc1(input))
        input = self.activation(self.fc2(input))
        output = self.output(input)
        return output

def loadModel(numFeatures,path):
    model = Net(numFeatures)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def train(model, dataLoader, optimizer, lossFunction):
    model.train()
    runningLoss = 0.0
    #training loop
    for i, data in enumerate(dataLoader):
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
    epochLoss = runningLoss / len(dataLoader)
    return epochLoss

def test(model, dataLoader):
    model.eval()
    correct = 0
    total = 0
    #validate the model on validationSet
    with torch.no_grad():
        for i, data in enumerate(dataLoader):
            batch,labels = data
            labels = labels.view(-1,1)
            output = model(batch)
            for index, val in enumerate(output):
                if (abs(val - labels[index])) < .1:
                    correct +=1
                total +=1 
    try:
        accuracy = float(correct)/total
        return accuracy
    except ZeroDivisionError:
        return 0

def givePredictions(model, evalData):
    model.eval()
    #predict output for each example
    for i, data in enumerate(evalData):
        batch, _ = data
        output = model(batch)
        print("Player: {0} - {1:4f}".format(evalData.getPlayerName(i), output.item()))
    

def main():
    #Training variables
    EPOCH = 1500
    BATCH_SIZE = 50
    VALIDATION_SPLIT = .2
    LEARNING_RATE = .001

    #parser
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    parser.add_argument('--train', action='store_true', help="training mode")
    parser.add_argument('--eval', action='store_true', help="evaluation mode")
    parser.add_argument('-p','--plot', action='store_true', help="plot metrics after training model")
    parser.add_argument('-s','--save', action='store_true', help="save the model weights after training")
    parser.add_argument('-k','--kfold', help="kfold validation")
    parser.add_argument('-w','--weights', help="weights file to save/load")
    parser.add_argument('-d', '--dataPath',required=True,help="name of csv file(minus .csv) that will be loaded. Must be located in ./Data folder")
    args = parser.parse_args()

    #features and label information
    dataInfo = {
        'features':['mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws']
        , 'label':'award_share'}
    numFeatures = len(dataInfo['features']) 

    #weightsPath
    weightsPath = "../Weights/{0}.pth".format(args.weights) if args.weights is not None else '../Weights/NBA_net.pth'

    #dataset to be loaded    
    dataPath = "../Data/" + args.dataPath + '.csv'

    if args.kfold:
         #create model, optimizer, lossFunction
        model = Net(numFeatures)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        lossFunction = nn.MSELoss(reduction='sum')
        #training dataset
        trainDataset = NBADataset(dataPath, dataInfo) 
        for k in range(int(args.kfold)):
            highest = 0.0
            #split the dataset based on k
            trainSample, testSample = KFoldCross(trainDataset,int(args.kfold),k)     
            trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE,sampler=trainSample)
            testDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE,sampler=testSample)
            #epoch loop
            for epcoh in range(1, EPOCH + 1):
                epochLoss = train(model, trainDataLoader, optimizer, lossFunction)
                accuracy = test(model,testDataLoader) 
                if accuracy > highest:
                    highest = accuracy
            print("for K = {0}, Accuracy: {1:5f}".format(k,highest))
    
    #train the model
    elif args.train:
        #create model, optimizer, lossFunction
        model = Net(numFeatures)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
        lossFunction = nn.MSELoss(reduction='sum')

        #load the datasets
        trainDataset = NBADataset(dataPath, dataInfo)
        trainSet, validSet = SplitDataSet(trainDataset,VALIDATION_SPLIT)
        trainDataLoader = DataLoader(trainSet, batch_size=BATCH_SIZE,shuffle=True)
        validDataLoader = DataLoader(validSet, batch_size=BATCH_SIZE,shuffle=False)

        #epoch loop
        losses = []
        accuracies = []
        for e in range(1, EPOCH + 1):
            epochLoss = train(model, trainDataLoader, optimizer, lossFunction)
            accuracy = test(model,validDataLoader)
            losses.append(epochLoss)
            accuracies.append(accuracy)
            if e%50 == 49:
                print("Epoch {0}: Loss {1:4f}, Accuracy: {2:4f}".format(e+1,epochLoss,accuracy))

        if args.plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(np.array(losses), 'r')
            ax1.set(xlabel='Epoch')
            ax1.set_title('Loss')
            ax2.plot(np.array(accuracies), 'g')   
            ax2.set_title('Accuracy') 
            ax2.set(xlabel='Checkpoint')   
            plt.show()

        if args.save:
            torch.save(model.state_dict(),weightsPath) 

    #eval the model
    elif args.eval:
        model = Net(numFeatures)
        model.load_state_dict(torch.load(weightsPath))
        model.eval()
        #load in the examples
        evalData = NBADataset(dataPath, dataInfo)
        givePredictions(model,evalData)

if __name__ == '__main__':
    main()