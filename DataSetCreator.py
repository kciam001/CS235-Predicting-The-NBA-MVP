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
    def __init__(self, csvFile): 
        features, data = ProcessCSV(csvFile)
        #get the award share column. This is our label
        idx = np.where(features=='award_share')
        self.awdShare = data[:,idx]
        #delete this from the dataset
        self.data = np.delete(data,idx,axis=1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        sample = self.data[idx,:]
        label = self.awdShare[idx]
        return torch.Tensor(sample), label