import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def cost(stats, dependent, theta):
    m = len(dependent)
    c = np.sum((stats.dot(theta) - dependent) ** 2)/(2 * m)
    return c

def rmse(actual, pred):
    rmse = np.sqrt(sum((actual - pred)**2)/len(actual))
    return rmse

def r2(dependent, results):
    mean_y = np.mean(dependent)
    ss_tot = sum((dependent - mean_y) ** 2)
    ss_res = sum((dependent - results) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2
    
def lr(stats, dependent, theta, alpha, iterations):
	m = len(dependent)
	for iteration in range(iterations):
		gradient = stats.T.dot(stats.dot(theta) - dependent)/m
		theta = theta - alpha * gradient
		c = cost(stats, dependent, theta)
	return theta

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
data = pd.read_csv('train_data.csv')
testdata = pd.read_csv('final_test_data.csv')
stats = data.ix[:,1:-1]
award_share = data.ix[:,-1]

x0 = np.ones(stats.shape[0])[...,None]
statsdata = np.append(x0, stats, 1)
theta = np.zeros(statsdata.shape[1])
alpha = 0.0001
iter_ = 10000
newt = lr(statsdata, award_share, theta, alpha, iter_)

t = testdata.ix[:,1:]
test0 = np.ones(t.shape[0])[...,None]
tdata = np.append(test0, t, 1)
ty = testdata.ix[:,-1]
players = testdata.ix[0:,0]

pred = tdata.dot(newt)

players = pd.concat([players, pd.DataFrame(pred)], axis=1)
players = players.sort_values(by = 0, ascending=False)
players.columns = ['Player', 'Award Share']
print(players.head(5))

results = statsdata.dot(newt)
error = rmse(award_share, results)
r2_error = r2(award_share, results)
#print("RMSE: " + str(round(error, 4)))
