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

def refeaturize(x,center=0):
    features = []
    for v in x:
        v -= center
        arg = math.modf(v)[0]
        features.append((np.power(int(v*8),int(2))//100 , np.power(int(v*8),int(3))//10000 , int(256 * (1.+np.cos(np.pi*2*arg))) , int(256*(1.+np.sin(np.pi*2*arg))), int(256 * (1.+np.cos(np.pi*4*arg))) , int(256*(1.+np.sin(np.pi*4*arg))) , int(256 * (1.+np.cos(np.pi*2*v/10))) , int(256*(1.+np.sin(np.pi*2*v/10))) , int(256 * (1.+np.cos(np.pi*2*v/20))) , int(256*(1.+np.sin(np.pi*2*v/20)))) )
    return features


def featurize(x,center=0):
    features = []
    for v in x:
        features.append((int((v-center)*8) , np.power(int((v-center)*8),int(2))//100 )) 
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
    m = 40 #np.mean(x)
    y = np.array(co2data['Carbon Dioxide (ppm)'])

    x_learn = featurize(x,m)
    y_learn = [val for val in y]
    x_train,x_test,x_validate,y_train,y_test,y_validate = katiesplit(x_learn,y_learn)
    quadmod = linear_model.LinearRegression().fit(x_train, y_train)
    print ("Accuracy (test): ", quadmod.score(x_test, y_test))
    print ("Accuracy (validate): ", quadmod.score(x_validate, y_validate))

    np.savetxt('%s.original'%sys.argv[1],np.column_stack((x,x_learn,y_learn)),fmt='%.2f')

    x_learn = refeaturize(x,m)
    y_learn -= quadmod.predict(featurize(x,m))
    x_train,x_test,x_validate,y_train,y_test,y_validate = katiesplit(x_learn,y_learn)
    pertmod = linear_model.LinearRegression().fit(x_train, y_train)
    print ("Perturbation Accuracy (test): ", pertmod.score(x_test, y_test))
    print ("Perturbation Accuracy (validate): ", pertmod.score(x_validate, y_validate))

    x = np.linspace(0,80,2561)
    x_pred = refeaturize( x , m)
    y_pred = pertmod.predict(x_pred) + quadmod.predict( featurize(x,m) )
    np.savetxt('%s.predictions'%sys.argv[1],np.column_stack((x,x_pred,y_pred)),fmt='%.2f')
    return

if __name__ == '__main__':
    main()

