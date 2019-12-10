#!/bin/bash

#Run KNN Script
printf "\n"
cd knnRegression
printf "KNN predictions:\n"
python knnRegression.py 6 ../Data/TrainReduced.csv ../Data/FinalTestData.csv
cd ..
#Run NN script
printf "\n"
cd NeuralNet
printf "Neural Network predictions:\n"
python nn.py --test --dataset=FinalTestData
cd ..

#Run linear Regression script
printf "\n"
cd linearRegression
printf "Linear  Regression predictions:\n"
python -W ignore linearRegression.py
cd ..