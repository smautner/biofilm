
import matplotlib.pyplot as plt
import structout as so
import numpy as np
from lmz import *



def lasso(model,X,y):
    from sklearn.linear_model import Lasso
    m =[ Lasso(alpha=a).fit(X,y).coef_ for a in model.alphas_]
    coe = [sum(abs(mm)>.00001) for mm in m]
    e = model.mse_path_.mean(axis=1)
    so.bins(e,minmax=False)
    so.lprint(coe) # i should make this work
