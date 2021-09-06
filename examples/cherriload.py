import RNA_RNA_binding_evaluation.get_features as cri
import eden.graph as eg
import networkx as nx
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz
partner =  {a:b for a,b in zip("({[<",")}]>")  }


def mkgraph(sequence, structure):
    graph = nx.Graph()
    lifo = defaultdict(list)
    cut = structure.index("&")
    structure = structure.replace("&","")
    sequence = sequence.replace("&","")
    for i,(s,n) in enumerate(zip(structure, sequence)):
        graph.add_node(i, label=n)
        if i > 0 and  i != cut:
            graph.add_edge(i, i-1, label='-')

        # ADD PAIRED BASES
        if s in ['(','[','<']:
            lifo[partner[s]].append(i)
        if s in [')',']','>']:
            j = lifo[s].pop()
            graph.add_edge(i, j, label='=')
    return graph
    #return eg.vectorize([graph], discrete = False) # keep here in case i want nested edges ...


def convert(negname, posname, outname):
    d1 = cri.loadDF(negname)
    d2 = cri.loadDF(posname)

    # making y is easy
    y = np.array([0]*len(d1)+[1]*len(d2))

    # convert the dataframes, deleting the 'object' dtypes as they are strings ...
    X = np.vstack(( d1.to_numpy(), d2.to_numpy()))
    X = X[:,d1.dtypes != 'object']
    X=csr_matrix(X.astype(np.float64))

    # vectorize the binding...
    X2 = [mkgraph(seq, stru) for d in [d1,d2] for seq,stru in  zip(d['subseqDP'],d['hybridDP'])  ]
    X2 = eg.vectorize(X2)

    #print(f" {X2.shape=} {X.shape=}")
    #print(d1['subseqDP'])
    #print(d1['hybridDP'])

    X= csr_matrix(hstack((X,X2)))
    save_npz(outname+'.X.csr',X)
    np.savez(outname+'.y.npz',y)
    return X,y


def read(name):
    X = load_npz(name+".X.csr.npz")
    y = np.load(name + '.y.npz',allow_pickle=True)['arr_0']
    #y = np.load(name + '.y.npz',allow_pickle=True).reshape((-1,1))
    #print(f" {X.shape} {y.shape}")
    return X,y,np.array(range(X.shape[0])), np.array(range(X.shape[1]))

def makedata():
    p = "/home/ubuntu/repos/RNA_RNA_binding_evaluation/test_data/training/test_context_feat/"
    d1 = p+'test_con300_neg.csv'
    d2 = p+"test_con300_pos.csv"
    convert(d1,d2,'cherry')


#X,y,_,sd = read('cherry')

