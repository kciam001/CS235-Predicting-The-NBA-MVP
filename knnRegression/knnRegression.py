import csv
import sys

trainFileName = sys.argv[1]
testFileName = sys.argv[2]
k = sys.argv[3]

with open(trainFileName, 'r') as trainFile, open(testFileName, 'r') as testFile:
    trainReader = csv.reader(trainFile)
    testReader = csv.reader(testFile)
    next(trainReader)
    next(testReader)
    
    trainList = list(trainReader)
    testList = list(testReader)

    for testPlayer in testList:
        nearestNeighbors = []

        for trainPlayer in trainList:
            distance = 0

            for testFeature, trainFeature in zip(testPlayer[1:-1], trainPlayer[1:-1]):
                distance += (float(testFeature) - float(trainFeature))**2
            distance **= 0.5
            nearestNeighbors.append([trainPlayer, distance])
            if len(nearestNeighbors) > int(k):
                nearestNeighbors.remove(max(nearestNeighbors, key = lambda x: x[1]))

        average = 0

        for neighbors in nearestNeighbors:
            average += float(neighbors[0][-1])
        average /= len(nearestNeighbors)

        print(testPlayer[0] + ' award_share: ' + str(average))
