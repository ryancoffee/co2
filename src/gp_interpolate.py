#!/usr/bin/python3

import pandas as pd
import numpy as np
import sys
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel
import random
import math

def katiesplit(x,y):
    sz = len(x)
    inds = np.arange(len(x))
    np.random.shuffle(inds)
    x_train = [x[i] for i in inds[:sz//4]]
    x_test = [x[i] for i in inds[sz//4:2*sz//4]]
    x_validate = [x[i] for i in inds[2*sz//4:3*sz//4]]
    x_oob = [x[i] for i in inds[3*sz//4:]]
    y_train = [y[i] for i in inds[:sz//4]]
    y_test = [y[i] for i in inds[sz//4:2*sz//4]]
    y_validate = [y[i] for i in inds[2*sz//4:3*sz//4]]
    y_oob = [y[i] for i in inds[3*sz//4:]]
    return (x_train,x_test,x_validate,y_train,y_test,y_validate,x_oob,y_oob)

def refeaturize(x,center=0):
    features = []
    for v in x:
        v -= center
        arg = math.modf(v)[0]
        features.append((np.power(int(v*8),int(2))//100 , np.power(int(v*8),int(3))//10000 , int(256 * (np.cos(np.pi*2*arg))) , int(256*(np.sin(np.pi*2*arg))), int(256 * (np.cos(np.pi*4*arg))) , int(256*(np.sin(np.pi*4*arg))) , int(256 * (np.cos(np.pi*2*v/10))) , int(256*(np.sin(np.pi*2*v/10))) , int(256 * (np.cos(np.pi*2*v/20))) , int(256*(np.sin(np.pi*2*v/20)))) )
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
    x_train,x_test,x_validate,y_train,y_test,y_validate,x_oob,y_oob = katiesplit(x_learn,y_learn)
    quadmod = linear_model.LinearRegression().fit(x_train, y_train)
    print ("Accuracy (test): ", quadmod.score(x_test, y_test))
    print ("Accuracy (validate): ", quadmod.score(x_validate, y_validate))

    np.savetxt('%s.original'%sys.argv[1],np.column_stack((x,x_learn,y_learn)),fmt='%.2f')

    x_learn = refeaturize(x,m)
    y_learn -= quadmod.predict(featurize(x,m))
    x_train,x_test,x_validate,y_train,y_test,y_validate,x_oob,y_oob = katiesplit(x_learn,y_learn)
    pertmod = linear_model.LinearRegression().fit(x_train, y_train)
    print ("Perturbation Accuracy (test): ", pertmod.score(x_test, y_test))
    print ("Perturbation Accuracy (validate): ", pertmod.score(x_validate, y_validate))

    pred_arg = np.linspace(0,80,2561)
    x_pred = refeaturize( pred_arg , m)
    y_pred = pertmod.predict(x_pred) + quadmod.predict( featurize(pred_arg,m) )
    np.savetxt('%s.predictions'%sys.argv[1],np.column_stack((pred_arg,x_pred,y_pred)),fmt='%.2f')

    ''' =============== Gaussian process section =================== '''

    x_learn = x.copy()
    y_learn = [val for val in y]

    x_train,x_test,x_validate,y_train,y_test,y_validate,x_oob,y_oob = katiesplit(x_learn,y_learn)

    k1 = 66.0**2 * RBF(length_scale=67.0) # long term smooth rising trend
    k2 = 2.4**2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
    k3 = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78) # medium term irregularity
    k4 = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)  # noise terms
    kernel_gpml = k1 + k2 + k3 + k4
    gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, optimizer=None, normalize_y=True)
    gp.fit(np.asarray(x_train).reshape(-1,1), np.asarray(y_train).reshape(-1,1))

    print("GPML kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))
    print('GP score (test): ',  gp.score(np.asarray(x_test).reshape(-1,1), np.asarray(y_test).reshape(-1,1)))
    print('GP score (validate): ',  gp.score(np.asarray(x_validate).reshape(-1,1), np.asarray(y_validate).reshape(-1,1)))

    # Kernel with reduced parameters
    #k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
    k1 = 50.0**2 * RationalQuadratic(length_scale=50.0,alpha=1.0)  # long term smooth rising trend
    #k2 = 2.0**2 * RBF(length_scale=100.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")  # seasonal component
    k2 = 2.0**2 * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds=(.9,1.1))  # seasonal component
    #k3 = 10**2 * RationalQuadratic(length_scale=10.0, alpha=1.0) # medium term irregularities
    k4 = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, np.inf))  # noise terms
    k5 = ConstantKernel(constant_value = 10000,constant_value_bounds="fixed") # baseline shift
    kernel = k1 + k2 + k4 + k5


    gp_me = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)
    gp_me.fit(np.asarray(x_train).reshape(-1,1), np.asarray(y_train).reshape(-1,1))

    print("\nLearned kernel: %s" % gp_me.kernel_)
    print("Log-marginal-likelihood: %.3f" % gp_me.log_marginal_likelihood(gp_me.kernel_.theta))

    x_pred = pred_arg[:,np.newaxis]
    y_pred, y_std = gp.predict(np.asarray(x_pred).reshape(-1,1), return_std=True)
    y_pred_me, y_std_me = gp_me.predict(np.asarray(x_pred).reshape(-1,1), return_std=True)
    print('GP score (test): ',  gp_me.score(np.asarray(x_test).reshape(-1,1), np.asarray(y_test).reshape(-1,1)))
    print('GP score (validate): ',  gp_me.score(np.asarray(x_validate).reshape(-1,1), np.asarray(y_validate).reshape(-1,1)))

    np.savetxt('%s.gp_predictions'%sys.argv[1],np.column_stack((x_pred,y_pred,y_std,y_pred_me,y_std_me)),fmt='%.2f')

    print("Now doing compound quadratic linregression, then reduced GP\n\n")
    print("... actually going to skip this for now")
    return

    x = np.array(co2data['Decimal Date'] - 1950)
    m = 40 #np.mean(x)
    y = np.array(co2data['Carbon Dioxide (ppm)'])

    x_learn = x.copy()
    y_learn = [val for val in y]
    x_train,x_test,x_validate,y_train,y_test,y_validate,x_oob,y_oob = katiesplit(x_learn,y_learn)
    quadmod = linear_model.LinearRegression().fit( featurize(x_oob,m) , y_oob)

    k1 = 2.0**2 * RBF(length_scale=100.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")  # seasonal component
    k2 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0) # medium term irregularities
    k3 = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, np.inf))  # noise terms
    kernel_compound = k1 + k2 + k3 
    gp = GaussianProcessRegressor(kernel=kernel_compound, alpha=0, normalize_y=True)
    gp.fit(np.asarray(x_train).reshape(-1,1), np.asarray(y_train - quadmod.predict( featurize(x_train,m) ).reshape(-1,1)))
    print("\nLearned kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

    y_pred = gp.predict(np.asarray(x_pred).reshape(-1,1), return_std=False)
    y_qm = quadmod.predict(featurize(x_pred,m))
    y_pred = np.asarray(y_pred).reshape(-1,1)+ y_qm 
    #print('GP score (test): ',  gp.score(np.asarray(x_test).reshape(-1,1), np.asarray(y_test).reshape(-1,1) - quadmod.predict(featurize(x_test,m)) ))
    #print('GP score (validate): ',  gp.score(np.asarray(x_validate).reshape(-1,1), np.asarray(y_validate - quadmod.predict(featurize(x_validate,m))).reshape(-1,1) ))

    np.savetxt('%s.quad_gp_predictions'%sys.argv[1],np.column_stack((x_pred,y_pred)),fmt='%.2f')






    return

if __name__ == '__main__':
    main()

