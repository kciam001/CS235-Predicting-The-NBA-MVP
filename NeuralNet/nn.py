import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from NBADataset import NBADataset, SplitDataSet, PrintDataSet, KFoldCross
from matplotlib import pyplot as plt
import argparse
import random

class Net(nn.Module):
    def __init__(self, numFeatures):
        super().__init__()
        self.fc1 = nn.Linear(numFeatures, 10)
        self.fc2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        #first hidden layer
        input = self.activation(self.fc1(input))
        #input = self.activation(self.fc2(input))
        #output layer
        output = self.output(input)
        return output

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

def evalMetrics(model, dataLoader):
    predictions = []
    groundTruth = []
    runningLoss = 0.0
    model.eval()
    #validate the model on validationSet
    with torch.no_grad():
        for i, data in enumerate(dataLoader):
            batch,labels = data
            labels = labels.view(-1,1)
            output = model(batch)
            predictions.extend(output.numpy())
            groundTruth.extend(labels.numpy())

    predictions = np.reshape(predictions,(-1))
    groundTruth = np.reshape(groundTruth,(-1))
    error = predictions - groundTruth
    accuracy = (np.absolute(error) < .1).sum() / predictions.size
    #Mean absolute error
    MAE = np.sum( np.absolute(error) ) /predictions.size
    #Mean square error
    MSE = np.sum( np.square(error) ) / predictions.size
    return {'Accuracy': accuracy, 'MAE': MAE, 'RMSE':np.sqrt(MSE)}

def givePredictions(model, evalData):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(evalData):
            player, _ = data
            output = model(player)
            predictions.append(output.item())
    idxs = sorted(range(len(predictions)), key=lambda k: predictions[k], reverse=True)
    #print sorted top players
    for i, pos in enumerate(idxs):
        print("{0}. {1}: {2:.5f}".format(i+1,evalData.getPlayerName(pos),predictions[pos]))

def main():
    #Training variables
    EPOCH = 1000
    BATCH_SIZE = 50
    VALIDATION_SPLIT = .2
    LEARNING_RATE = .001
    REDUCED_FEATURES = ['mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws']

    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    parser.add_argument('--train', action='store_true', help="training mode")
    parser.add_argument('--test', action='store_true', help="evaluation mode")
    parser.add_argument('-r','--random', action='store_true', help="use random subset of features for training")
    parser.add_argument('-p','--plot', action='store_true', help="plot metrics after training model")
    parser.add_argument('-s','--save', action='store_true', help="save the model weights after training")
    parser.add_argument('-k','--kfold', help="kfold validation")
    parser.add_argument('-w','--weights', help="weights file to save/load")
    parser.add_argument('-d', '--dataset',required=True,help="name of csv file(minus .csv) that will be loaded. Must be located in ./Data folder")
    args = parser.parse_args()

    #features and label information
    dataInfo = {
        'features': REDUCED_FEATURES
        , 'label':'award_share'}
    numFeatures = len(dataInfo['features']) 

    #weightsPath
    weightsPath = "../Weights/{0}.pth".format(args.weights) if args.weights is not None else '../Weights/NBA_net.pth'

    #dataset to be loaded    
    dataPath = "../Data/" + args.dataset + '.csv'

    #build the model
    model = Net(numFeatures)
    #create model, optimizer, lossFunction
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    lossFunction = nn.MSELoss(reduction='sum')
    #training dataset
    dataset = NBADataset(dataPath, dataInfo)

    #eval the model
    if args.test:
        model.load_state_dict(torch.load(weightsPath))
        model.eval()
        #load in the examples
        givePredictions(model,dataset)
        return

    elif args.kfold:
        minErrors = []
        for k in range(int(args.kfold)):
            lowestRMSE = float('inf')
            #split the dataset based on k
            trainSample, testSample = KFoldCross(dataset,int(args.kfold),k)     
            trainDataLoader = DataLoader(dataset, batch_size=BATCH_SIZE,sampler=trainSample)
            testDataLoader = DataLoader(dataset, batch_size=BATCH_SIZE,sampler=testSample)
            #epoch loop
            for epcoh in range(1, EPOCH + 1):
                epochLoss = train(model, trainDataLoader, optimizer, lossFunction)
                metrics = evalMetrics(model, testDataLoader)
                if metrics['RMSE'] < lowestRMSE:
                    lowestRMSE = metrics['RMSE']
            minErrors.append(lowestRMSE)
            print("for K = {0}, RMSE:{1:.5f}".format(k,lowestRMSE))
        print("average RMSE:{0:.5f}".format(np.mean(np.array(minErrors))  ))
    
    #train the model
    elif args.train:
        trainSet, validSet = SplitDataSet(dataset,VALIDATION_SPLIT)
        trainDataLoader = DataLoader(trainSet, batch_size=BATCH_SIZE,shuffle=True)
        validDataLoader = DataLoader(validSet, batch_size=BATCH_SIZE,shuffle=False)

        #epoch loop
        losses = []
        for e in range(1, EPOCH + 1):
            epochLoss = train(model, trainDataLoader, optimizer, lossFunction)
            metrics = evalMetrics(model, validDataLoader)
            losses.append(epochLoss)
            if e%50 == 49:
                print("Epoch {0}: Loss {1:4f}".format(e+1,epochLoss))

        if args.plot:
            fig, (ax1) = plt.subplots(1, 1)
            ax1.plot(np.array(losses), 'r')
            ax1.set(xlabel='Epoch')
            ax1.set(ylabel='Loss')
            ax1.set_title('Training Loss')             
            plt.show()

        if args.save:
            torch.save(model.state_dict(),weightsPath) 
            print("saving weights")

if __name__ == '__main__':
    main()