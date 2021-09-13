
'''
1. load data
we can do it in a fancy way, by providing a python file that has a read(path) function. as demonstrated in examples/cherriload.py
'''

loaddata = '--infile examples/cherry --loader examples/cherriload.py '

'''
2. extract features
'''
if False:
    parallel -j 32 --joblog feat.log python biofilm/biofilm-features.py @(loaddata)\
        --foldselect '{1}' --method forest --out '{1}' ::: $(seq 0 4)

'''
3. run optimization
'''
loaddata += '--featurefile {1}'
parallel -j 32 --joblog feat.log python biofilm/biofilm-optimize6.py  @(loaddata)\
    --foldselect '{1}' --out '{1}.out' --n_jobs 5 --time 120 ::: $(seq 0 4)
