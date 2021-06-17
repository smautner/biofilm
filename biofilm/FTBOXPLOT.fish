set -x 'MKL_NUM_THREADS' 1
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1




parallel --record-env  

#########################
# feature selectors 
#########################

set ftmethods svm all corr variance agglocorr agglocore
set blacklist kernel genome no
set randseeds (seq 1 5) 
set folds (seq 0 4)
set samples 10000


set randseeds (seq 1 1) 
set samples 1000



set para -j 32 --joblog log.txt --env _ python biofilm-features.py 

set load  --infile pigdataframe2.pkl --randinit {1} --loader ../examples/pigdataframe.py 
set load2 --foldselect {2} --pigdatasubsample $samples --pigblacklist {4} --Z True

set task --method {3} --out res/{1}_{2}_{3}_{4}.ft --n_jobs 1 --penalty l1 

parallel $para $load $load2 $task ::: $randseeds ::: $folds ::: $ftmethods ::: $blacklist


##########################3
# optimization 
##########################


set para -j 32 --joblog log2.txt --env _ python biofilm-optimize6.py 
set load3  --featurefile res/{1}_{2}_{3}_{4}.ft.npz
set task --method any_classifier --out res3/{1}_{2}_{3}_{4}.json

parallel $para $load $load2 $load3  $task ::: $randseeds ::: $folds ::: $ftmethods ::: $blacklist



#awk  '{arr[$2]+=$4}END{for (a in arr) print a, arr[a]}' | sort


# -<< ZOMG  >>-  

function aaa
    python -c "
import numpy as np
import os

def load(folder,loader):
    r= []
    for f in os.listdir(folder): 
            dim = f[:f.find('.')].split('_')
            myres = { f'd{i}':int(z) for i,z in enumerate(dim) } 
            myres['data'] = loader(f'{folder}/{f}')
            r.append(myres)
    return r 

r = load('./res', lambda x: np.load(x)['arr_0'])
r = [di['data'] for di in r]
print(np.mean(r))

#import matplotlib
#matplotlib.use('module://matplotlib-sixel')
#import matplotlib.pyplot as plt
#plt.plot(r)
#plt.show()
"
end 


