

import ubergauss.tools as t
import sklearn.datasets as ds
import numpy as np
from biofilm import util
X,y = ds.make_classification()
t.ndumpfile([X,y],"mydata.sav")

from biofilm.util.data import loadfolds


import biofilm.optimize6 as o6

import dirtyopts



optidoc = '''
--methods str+ any  'extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp'
--out str jsongoeshere
--n_jobs int 1
--time int 100
--memoryMBthread int 8000
--randinit int 1337  # should be the same as the one in data.py
--preprocess bool False
--tmp_folder str
'''


optidoc22='''
# theese are the options for reading data
--infile str mydata.sav.npz
--loader str  # optional path to a python file that introduces a load function

--randinit int 1337
--folds int 5
--foldselect int 0

--subsample int -1
--Z bool False

--featurefile str
--featurecount int -1
'''

data,f,i = loadfolds(**dirtyopts.parse(optidoc22).__dict__) # should return the first /5 folds..
args = dirtyopts.parse( optidoc) # should just give us the defaults
model=o6.optimize(*data, args)


# scorehistory =  np.nan_to_num(\
#         model.performance_over_time_['single_best_optimization_score'].to_numpy(),nan=0.0)
# util.report(model, args.out, additionaloutput=\
#     {'scorehistory': scorehistory , 'performancelog': model.performance_over_time_})



