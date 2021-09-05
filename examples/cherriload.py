import RNA_RNA_binding_evaluation.get_features as cri
import eden.graph as eg
import networkx as nx
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, hstack
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


def read(filename):
    d1 = cri.loadDF(filename+'test_con300_neg.csv')
    d2 = cri.loadDF(filename+"test_con300_pos.csv")

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

    X= hstack((X,X2))
    return X,y

###


if __name__ == "__main__":
    p = "/home/ubuntu/repos/RNA_RNA_binding_evaluation/test_data/training/test_context_feat/"
    x,p =  read(p)
    np.savez('eeeh.npz',(x,p))
