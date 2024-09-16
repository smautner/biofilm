

## install

```
pip install biofilm
conda install -c conda-forge biofilm
```



## Optimization options:

```
--methods str+ any  'extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp'
--out str jsongoeshere
--n_jobs int 1
--time int 3600
--memoryMBthread int 8000
--randinit int 1337  # should be the same as the one in data.py
--preprocess bool False
--tmp_folder str
--refit bool True
--instancegroups str   # a jsonfile containing a dictionary instance_name -> group name
--autosk_debug bool False   # autosklearn logging
--autosk_debugfile str autosklearn.log
--autosk_debugout str+ file_handler  # console   is another option, chooses where to output debug
--ensemble int 1  # ensemble size, autosklearn will combine the best models
```

## Feature selection options:

go to biofilm and run python biofilm-features.py -h

```
# feature selection options
--method str lasso  or svm or all or corr or variance
--out str numpycompressdumpgoeshere
--plot bool False
--svmparamrange float+ 0.01 0.15 0.001

# data reading options
--infile str myNumpyDump
--randinit int -1
--folds int 5
--subsample int -1
--Z bool False
```


## data loading

```
a) tools.ndumpfile([X,y, featurenames, instancenames],fname) where feature and instancenames are optional or
b) provide --loader whose read function will be called (examples/npzloader)

defaultformat: X,y in a npz dump, features and instances get enumerated
a custom dataloader: X,y, features, instances
loadfoldsreturns: (X,Y,x,y) features namesOfTestInstances
```

## outputs

```
optimize:
	out.model: {score:score, modelparams:modelparams}
	out.csv: instanceId, reallabel, predicted label, probability
feature selection:
	out: featuremask, featureproba, featureId
```


