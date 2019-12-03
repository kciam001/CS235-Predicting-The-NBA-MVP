"""
For testing new data
> py knnRegression.py <number of neighbors> <train file name> <test file name>
Test file must have award_share column of arbitrary values

For cross validation
> py knnRegression.py <number of neighbors> <train file name> <number of folds>

For leave one out validation
> py knnRegression.py <number of neighbors> <train file name>
"""

import csv
import math
import sys

def knnRegression(trainList, testList):
    averageAwardShares = []

    for testPlayer in testList:
        nearestNeighbors = []

        for trainPlayer in trainList:
            distance = 0

            for testFeature, trainFeature in zip(testPlayer[1:-1], trainPlayer[1:-1]):
                distance += (float(testFeature) - float(trainFeature))**2
            distance **= 0.5
            nearestNeighbors.append([trainPlayer, distance])
            if len(nearestNeighbors) > int(kNeighbors):
                nearestNeighbors.remove(max(nearestNeighbors, key = lambda x: x[1]))

        averageAwardShare = 0

        for neighbors in nearestNeighbors:
            averageAwardShare += float(neighbors[0][-1])
        averageAwardShare /= len(nearestNeighbors)

        averageAwardShares.append(averageAwardShare)
    
    return averageAwardShares

if len(sys.argv) < 3:
    print('Insufficient number of parameters')

    sys.exit(0)
elif len(sys.argv) > 4:
    print('Dropping excess parameters')
kNeighbors = sys.argv[1]
trainFileName = sys.argv[2]
with open(trainFileName, 'r') as trainFile:
    trainReader = csv.reader(trainFile)
    next(trainReader)

    trainList = list(trainReader)
try:
    kFold = math.ceil(len(trainList) / int(sys.argv[3]))
    accuracies = 0

    for i, j in zip(range(0, len(trainList), kFold), range(1, len(trainList) + 1)):
        averageAwardShares = knnRegression([trainList[k] for k in range(len(trainList)) if k not in range(i, i + kFold)], trainList[i:i + kFold])
        score = 0

        for k, averageAwardShare in zip(range(i, i + kFold), averageAwardShares):
            if abs(float(trainList[k][-1]) - averageAwardShare) < 0.1:
                score += 1

        accuracy = score / len(averageAwardShares)
        accuracies += accuracy

        print('Score for cross validation iteration ' + str(j) + ': ' + str(accuracy))
    averageAccuracy = accuracies / j
    print('Average accuracy of cross validation: ' + str(averageAccuracy))
except IndexError:
    score = 0

    for i in range(len(trainList)):
        averageAwardShare = knnRegression([trainList[j] for j in range(len(trainList)) if j != i], [trainList[i]])
        
        if abs(float(trainList[i][-1]) - averageAwardShare[0]) < 0.1:
            score += 1

    accuracy = score / len(trainList)

    print('Average accuracy of leave one out validation: ' + str(accuracy))
except ValueError:
    testFileName = sys.argv[3]
    with open(testFileName, 'r') as testFile:
        testReader = csv.reader(testFile)
        next(testReader)

        testList = list(testReader)
    averageAwardShares = knnRegression(trainList, testList)

    for player, averageAwardShare in zip(testList, averageAwardShares):
        print(player[0] + ' [award_share]: ' + str(averageAwardShare))
