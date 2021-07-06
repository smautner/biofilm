
import matplotlib
matplotlib.use('module://matplotlib-sixel')
import dirtyopts
import numpy as np
from lmz import *   
import matplotlib.pyplot as plt
from sklearn.metrics import  f1_score
##################
# the purpose oof this is to draw a precision recall curve 
# should also use the average rank of an instance  as a threshold if multiple seeds were used
#################

doc = '''
--infiles str+ ''
--rawproba bool False
--showproblems int 0
--drawAll bool False
'''

args = dirtyopts.parse(doc)



##########
# READ ALL THE DATA
#########
from collections import defaultdict
seeds = defaultdict(list)
for e in args.infiles: 
    for line in open(e,'r').read().split('\n'):
        if len(line) < 4: 
            continue
        instance, truth, pred, score, seed  = line.split(',')
        seeds[int(seed)].append( [float(score), int(truth), instance, int(pred) ])



############3
# get truth and scores  for each instance
############
y = []
scores = []
insta = instance
predictions = [] 
for sti_list  in seeds.values(): 
    sti_list.sort(key=lambda x: x[2] )
    score, truth, instance, prediction = Transpose(sti_list)
    y= truth
    insta = instance
    predictions.append(prediction)
    scores.append(score)

####
# rank transform
####
if not args.rawproba:
    scores = [ np.argsort(np.argsort(s))   for s in scores ]


#########
# get avg score
#########
#avg_score= [np.mean(x) for x in zip(scores)] if len(scores)> 1 else scores[0]
avg_score= [np.mean(x) for x in zip(*scores)] 

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve( truth, avg_score)
plt.plot(recall, precision, label=f"mean {'score' if args.rawproba else 'rank'} ({len(scores)} repeats)")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title(f'score: {np.mean([f1_score(truth,x) for x in predictions])} instances: {len(truth)}')

if args.drawAll: 
    for s in scores:
        precision, recall, thresholds = precision_recall_curve( truth, s)
        plt.plot(recall, precision)

plt.legend()
plt.show()

###########
# get probalmatic instances
###########
if args.showproblems > 0:
    truth = np.array(truth) 
    scores2 = np.array(avg_score) 
    scores = np.array(scores).T
    print('SCORESHAPE',scores.shape)
    instances = np.array(insta) 
    pmask = truth ==1 
    nmask = truth ==0

    sorted_pos = np.argsort( scores2[pmask] )
    print(f"problematic pos instances:")
    for r in [(instances[pmask][x], scores2[pmask][x])  for x in sorted_pos[:args.showproblems]]:
        print(r)
    sorted_neg = np.argsort( -scores2[nmask] )
    print(f"problematic neg instances:")
    for r in [(instances[nmask][x], scores2[nmask][x])  for x in sorted_neg[:args.showproblems]]:
        print(r)

    plt.close()
    plt.hist(scores2[pmask], bins = 100)
    plt.title('positive')
    plt.show()
    plt.close()
    plt.title('negative')
    plt.hist(scores2[nmask], bins = 100)
    plt.show()

