

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

