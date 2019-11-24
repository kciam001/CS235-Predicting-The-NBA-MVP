import csv
import numpy as np
import torch 
from torch.utils.data import Dataset

def processFloats(data):
    for i, row in enumerate(data):
        temp = np.array(row)
        data[i] = temp.astype(float)
    return np.array(data)

def ProcessCSV(file):
    results = []
    #open csv file
    datafile = open(file, 'r')
    myreader = csv.reader(datafile)
    for row in myreader:
        results.append(row)
    #the features are the first row of the csv file
    features = np.array(results[0])
    #convert the string floats to actual floats
    data = processFloats(results[1:])
    return features, data  

class NBADataset(Dataset):
    def __init__(self, csvFile, featureNames): 
        features, data = ProcessCSV(csvFile)
        #get index for our label
        idx = np.where(features == featureNames['label'])
        #get corresponding column
        self.awdShare = data[:,idx]

        #create empty array to hold features we care about
        self.data = np.zeros(shape=(self.awdShare.shape[0],1))
        #extract feature columns
        for name in featureNames['features']:
            index = np.where(features == name)
            self.data = np.append(self.data, data[:,index].reshape(-1,1), axis=1)
        #delete 0 column
        self.data = np.delete(self.data,0,axis=1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sample = self.data[idx,:]
        label = self.awdShare[idx]
        return torch.Tensor(sample), label