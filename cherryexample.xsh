'''
1. load data
we can do it in a fancy way, by providing a python file that has a
read(path) function. as demonstrated in examples/cherriload.py
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
-> outputs crossvall results in .csv files and model params in .model
'''
loaddata += '--featurefile {1} --foldselect {1}'
if False:
    parallel -j 32 --joblog feat.log python biofilm/biofilm-optimize6.py  @(loaddata)\
        --out '{1}.optimized' --n_jobs 5 --time 120 ::: $(seq 0 4)


'''
4. plot performance (so far)
'''
if False:
    python biofilm/biofilm-out.py --infiles *.csv


'''
5. do crossval for all models
      use all 5 models to crossvalidate over all instances to compare them...
'''
if False:

    # rum models
    parallel -j 32 --joblog delme.log python biofilm/biofilm-cv.py  @(loaddata) --model '{2}'\
        --out '{2}_{1}.last' ::: $(seq 0 4) ::: $(ls *.model)

if True:
    import dill
    import pprint
    loadfile = lambda filename: dill.load(open(filename, 'rb'))

    for i in range(0,4):
        $i = i
        filez = `$i.*.last.csv`
        python biofilm/biofilm-out.py --infiles @(filez)
        d = loadfile(filez[0].split("_")[0])
        print("self-score: ",d['score'])
        print("PARAMETERS")
        pprint.pprint(d['params'])


