#!/usr/bin/python3

import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
import random
import math

def katiesplit(x,y):
    sz = len(x)
    inds = np.arange(len(x))
    np.random.shuffle(inds)
    x_train = [x[i] for i in inds[:sz//4]]
    x_test = [x[i] for i in inds[sz//4:2*sz//4]]
    x_validate = [x[i] for i in inds[3*sz//4:]]
    y_train = [y[i] for i in inds[:sz//4]]
    y_test = [y[i] for i in inds[sz//4:2*sz//4]]
    y_validate = [y[i] for i in inds[3*sz//4:]]
    return (x_train,x_test,x_validate,y_train,y_test,y_validate)

def refeaturize(x,quadmod):
    features = []
    for v in x:
        arg,yr = math.modf(v) 
        features.append((int(v*8) , np.power(int(v*8),int(2))//100 , int(256 * (1.+np.cos(np.pi*2*arg))) , int(256*(1.+np.sin(np.pi*2*arg))), int(256 * (1.+np.cos(np.pi*4*arg))) , int(256*(1.+np.sin(np.pi*4*arg))) , int(256 * (1.+np.cos(np.pi*2*v/10))) , int(256*(1.+np.sin(np.pi*2*v/10))) ) )
    return features


def featurize(x):
    features = []
    for v in x:
        features.append((int(v*8) , np.power(int(v*8),int(2))//100 )) 
    return features

def main():
    if len(sys.argv)<2:
        print('syntax:\t%s <csv file>'%sys.argv[0])
        return
    co2data = pd.read_csv(sys.argv[1])
    print(co2data.head())
    print(co2data.isnull().any())
    co2data_full = co2data.copy(deep=True)
    co2data = co2data.dropna()
    print('referencing dates from 1950\ndays as % of the year')
    x = np.array(co2data['Decimal Date'] - 1950)
    print(x[:10])
    y = np.array(co2data['Carbon Dioxide (ppm)'])
    x_learn = featurize(x)
    y_learn = [val for val in y]
    x_train,x_test,x_validate,y_train,y_test,y_validate = katiesplit(x_learn,y_learn)
    quadmod = linear_model.LinearRegression().fit(x_train, y_train)
    print ("Accuracy (test): ", quadmod.score(x_test, y_test))
    print ("Accuracy (validate): ", quadmod.score(x_validate, y_validate))

    x_learn = refeaturize(x,quadmod)
    x_train,x_test,x_validate,y_train,y_test,y_validate = katiesplit(x_learn,y_learn)
    fullmod = linear_model.LinearRegression().fit(x_train, y_train)
    np.savetxt('%s.original'%sys.argv[1],np.column_stack((x,x_learn,y_learn)),fmt='%.2f')
    samp = np.zeros((1,2),dtype=int)
    samp = featurize([70.5])
    print(linmod.predict(samp))
    x = np.linspace(0,80,2561)
    x_pred = featurize( x )
    y_pred = linmod.predict(x_pred)
    np.savetxt('%s.predictions'%sys.argv[1],np.column_stack((x,x_pred,y_pred)),fmt='%.2f')
    return

if __name__ == '__main__':
    main()

