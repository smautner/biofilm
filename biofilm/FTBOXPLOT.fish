set -x 'MKL_NUM_THREADS' 1
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1
set -x 'NUMEXPR_MAX_THREADS' 1


#############
# special data loading options
############


set samples -1

set load  --infile pigdataframe2.pkl --loader ../examples/pigdataframe.py  \
    --fabikernel True  --pigblacklist kernel

##############
## data loading
#############

set folds 10
set randseeds (seq 1 1) 
set foldselect (seq 0 (math $folds -1))
set load2 --randinit {1} --foldselect {2} --folds $folds



#########################
# feature selectors 
#########################

set ftmethods svm #svm all corr variance agglocorr agglocore agglosvm

set para -j 32 --joblog log.txt  python biofilm-features.py  --Z False 
set task --method {3} --out res/{1}_{2}_{3}.ft --n_jobs 2 --runsvm False --penalty l2 --svmparamrange -3 2 10


#parallel $para $load $load2 $task ::: $randseeds ::: $foldselect ::: $ftmethods  


##########################3
# optimization 
##########################


set para -j 32 --joblog log2.txt  python biofilm_fabimlp.py  --Z False
set task --method any --out res/{1}_{2}_{3}.json --n_jobs 1 --debug True

#parallel $para $load $load2 --featurefile res/{1}_{2}_{3}_{4}.ft.npz  $task ::: $randseeds ::: $folds ::: $ftmethods ::: $blacklist
parallel $para $load $load2  $task ::: $randseeds ::: $foldselect ::: $ftmethods 



#awk  '{arr[$2]+=$4}END{for (a in arr) print a, arr[a]}' | sort



# -<< ZOMG  >>-  

function aaa
    python -c "
import numpy as np
import os
import basics as ba


def load(folder,loader):
    r= []
    for f in os.listdir(folder): 
            dim = f[:f.find('.')].split('_')
            myres = { f'd{i}':int(z) for i,z in enumerate(dim) } 
            myres['data'] = loader(f'{folder}/{f}')
            r.append(myres)
    return r 


r = load('./res', ba.jloadfile)
import pprint
pprint.pprint(r)

#import matplotlib
#matplotlib.use('module://matplotlib-sixel')
#import matplotlib.pyplot as plt
#plt.plot(r)
#plt.show()
"
end 

