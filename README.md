

# install

```
conda install -c conda-forge biofilm
```

# Feature selection is already nice:

go to biofilm and run python biofilm-features.py -h

```
# options for feature selection:
--method str lasso  or svm or all or corr or variance
--out str numpycompressdumpgoeshere
--plot bool False
--svmparamrange float+ 0.01 0.15 0.001

# theese are the options for reading data
--infile str myNumpyDump
--randinit int -1
--folds int 5
--subsample int -1
--Z bool False
```



## data loading

a) tools.ndumpfile([X,y, featurenames, instancenames],fname) where feature and instancenames are optional or
b) provide --loader whose read function will be called (examples/npzloader)

defaultformat: X,y in a npz dump, features and instances get enumerated
a custom dataloader: X,y, features, instances
loadfoldsreturns: (X,Y,x,y) features namesOfTestInstances


## outputs
optimize:
	out.model: {score:score, modelparams:modelparams}
	out.csv: instanceId, reallabel, predicted label, probability
feature selection:
	out: featuremask, featureproba, featureId




