import csv
import numpy as np
import torch
import math 
from torch.utils.data import Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

def ProcessCSV(file):
    #print("Loaded data from {0}".format(file))
    results = []
    #open csv file
    datafile = open(file, 'r')
    myreader = csv.reader(datafile)
    for row in myreader:
        results.append(row)
    #the features are the first row of the csv file
    features = [s.lower() for s in results[0]]
    data = np.array(results[1:])
    return features, data  

def index(arr, value):
    try:
        return arr.index(value)
    except ValueError:
        return []

class NBADataset(Dataset):
    def __init__(self, csvFile, featureNames): 
        features, rawData = ProcessCSV(csvFile)
        #get indexes
        labelIdx = index(features, featureNames['label']) 
        playerIdx = index(features,'player') 
        #get corresponding column
        self.awdShare = rawData[:,labelIdx].astype(float)
        #store playName information if possible
        self.playerNames = rawData[:,playerIdx]
        #get indices for our features
        indices = [features.index(ft) for ft in featureNames['features'] if ft in features]
        self.data = rawData[:,indices].astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sample = self.data[idx,:]
        label = self.awdShare[idx]
        return torch.Tensor(sample), label
    
    def getPlayerName(self,idx):
        if self.playerNames.size > 0:
            return self.playerNames[idx].item()
        else:
            return None

def SplitDataSet(dataset, split):
    testSize = int(split * len(dataset))
    trainSize = len(dataset) - testSize
    return random_split(dataset, [trainSize, testSize])

def PrintDataSet(data):
    for batch,label in data:
        print("example: ", batch, ", Label: ",label)

def KFoldCross(dataset,k, i):
    size = len(dataset)
    w = math.ceil(size/k)
    testIdx = [i for i in range(i*w,(i+1)*w) if i < len(dataset)]
    trainIdx = [i for i in range(len(dataset)) if i not in testIdx and i < len(dataset)]
    return SubsetRandomSampler(trainIdx), SubsetRandomSampler(testIdx)
