import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

def ProcessCSV(file):
    results = []
    datafile = open(file, 'r')
    myreader = csv.reader(datafile)
    for row in myreader:
        results.append(row)
    data = np.array(results)
    features = data[0,:]  
    data = np.delete(data, (0), axis=0)
    return features, data  

class NBADataset(Dataset):
    def __init__(self, csvFile): 


class Net(nn.Module):
    def __init__(self, numFeatures, numOut):
        super().__init__()
        #Linear(input output)
        self.fc1 = nn.Linear(numFeatures, 15)
        self.fc2 = nn.Linear(15, 15)
        self.output = nn.Linear(15, numOut)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        input = self.activation(self.fc1(input))
        input = self.activation(self.fc2(input))
        output = self.output(input)
        return output

def main():
    numFeatures = 5
    numOut = 1
    net = Net(numFeatures=numFeatures,numOut=numOut)
    #optimzer function
    optimizer = optim.SGD(net.parameters(), lr=.001)
    #loss function
    lossFunction = nn.MSELoss(reduction='sum')


    trainSet = []
    testSet = []
    EPOCH = 5

    for epoch in range(EPOCH):
        for data in trainSet:
            #unpack the data X and Y
            X, Y = data
            net.zero_grad()
            output = net(X.view(-1,numFeatures))
            loss = lossFunction(output, Y)
            loss.backward()
            optimizer.step()
    
    correct = 0
    total = 0

if __name__ == '__main__':
    main()