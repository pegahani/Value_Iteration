import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import cProfile

import sys
import time
import pickle
import Problem
import propagation_V

#-------------------------------------------------
import V_bar_search
import V_bar_search_mine
#-------------------------------------------------

sys.path.insert(0, "/Users/fl/flhome/1recherche/LN/corpus/etudiants/Pegah/code/Value_Iterationghub/Value_Iteration")
import m_mdp

ftype = np.float32

def generate_inequalities(_d):
    """
    gets d dimension and returns back a unit cube of d_dimension inequalities
    :param _d: space dimension
    :return: set of inequalities in which members are lists
    """
    inequalities = []

    for x in itertools.combinations( range(0, _d), 1 ) :
        inequalities.append([0] + [ 1 if i in x else 0 for i in xrange(_d) ])
        inequalities.append([1] + [ -1 if i in x else 0 for i in xrange(_d) ])

    return inequalities

def interior_easy_points(dim):
    #dim = len(self.points[0])
    l = []
    for i in range(dim):
        l.append(random.uniform(0.0, 1.0))
    return np.array(l, dtype=ftype)

def load(s, a, d, _id = ""):
    """
    Creates a new mdp, initialize related global variables and saves what is needed for reuse
    :type _id: string e.g. 80-1 to save in param80-1.dmp
    """
    global _lambda_rand
    global _state
    global _action
    global _d
    global m
    global Uvec
    global exact

    _d = d
    _state, _action = (s, a)
    _lambda_rand = interior_easy_points(_d)
    #m = m_mdp.make_grid_VVMDP(_lambda_rand, n=2)
    m = m_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand, None)
    Uvec = m.value_iteration(epsilon=0.00001)
    exact = m.initial_states_distribution().dot(Uvec)

    print 'exact', exact

    name = "param" + _id + ".dmp"
    pp = pickle.Pickler(open(name, 'w'))
    pp.dump(_lambda_rand)
    pp.dump((_state,_action, _d))
    pp.dump(m)
    pp.dump(Uvec)
    pp.dump(exact)

def reload(_id):
    """
    Reloads a saved mdp and initialize related global variables
    :type _id: string e.g. 80-1 to reload param80-1.dmp
    """
    name = "param" + _id + ".dmp"
    pup = pickle.Unpickler(open(name, 'r'))
    global _lambda_rand
    global _state
    global _action
    global _d
    global m
    global Uvec
    global exact

    _lambda_rand = pup.load()
    _state, _action, _d = pup.load()
    m = pup.load()
    Uvec = pup.load()
    exact = pup.load()

def load_V_Vectors(cluster_v_bar_epsilon,_name):

    global V_vectors
    global p
    #global v_prog

    res = open("show_out" + ".txt", "w")

    _p = Problem.Problem(initial=[{s:[random.randint(0,na-1)] for s in range(n)}, np.zeros(d, dtype=ftype),
                     np.zeros((n,d),dtype=ftype)], _mdp=m, _cluster_error=cluster_v_bar_epsilon, _epsilon= epsilon_error)

    """the main problem is defined as a mdp with two errors: epsilon_error: the used error in hierarchical clustering
    using cosine similarity metric. cluster_v_bar_epsilon: the used error for stopping criteria in generating
    \mathcal{V} set."""
    _v_prog = propagation_V.propagation_V(m=m, inequalities= generate_inequalities(m.d) ,cluster_v_bar_epsilon = cluster_v_bar_epsilon, epsilon_error = epsilon_error)
    _v_prog.initialize_LP()

    start1 = time.clock()
    tem = _v_prog.convex_hull_search(_p)
    stop1 = time.clock()

    #cProfile.run('v_prog.convex_hull_search(p)')

    print >> res, 'time v propagation',stop1- start1

    print >> res, 'iteration', tem[1]
    res.flush()

    _V_vectors = tem[0]
    V_vectors = _V_vectors

    print 'V_vectors', _V_vectors
    print 'len(V_vectors)', len(_V_vectors)

    print >> res, 'V_vectors', _V_vectors
    res.flush()
    print >> res, 'len(V_vectors)', len(_V_vectors)
    res.flush()

    p = _p
    #v_prog = _v_prog

    name = "param" + _name + "v-propagate.dmp"
    pp = pickle.Pickler(open(name, 'w'))
    pp.dump(p)
    #pp.dump(v_prog)
    pp.dump(V_vectors)

def relod_V_Vectors(_name):

    global V_vectors
    global p
    #global v_prog

    name = "param" + _name + "v-propagate.dmp"
    pup = pickle.Unpickler(open(name, 'r'))

    p = pup.load()
    #v_prog = pup.load()
    V_vectors = pup.load()

    if d==2:
        a = [ftype(item[0]) for item in V_vectors]
        b = [ftype(item[1]) for item in V_vectors]

        plt.scatter(a,b)
        plt.show()

def plot_(vectors, d):

    a = [ftype(item[0]) for item in vectors]
    b = [ftype(item[1]) for item in vectors]

    if d==2:
        plt.scatter(a,b)
        plt.show()

    elif d==3:
        c = [ftype(item[2]) for item in vectors]

        fig = pylab.figure()
        ax = Axes3D(fig)
        ax.scatter(a, b, c, color = 'r')
        pyplot.show()

if __name__ == '__main__':

    res = open("search-V" + ".txt", "w")
    start_all = time.clock()

    n = 10
    na = 5
    d = 5

    _cluster_v_bar_epsilon = 0.1 #we fix it at 0.1
    epsilon_error = 0.2


    #load(n, na, d, "test")
    reload("test")

    #load_V_Vectors(_cluster_v_bar_epsilon, "test")
    relod_V_Vectors("test")

    plot_(V_vectors, d)

    #-------------------------------------------------
    #V = V_bar_search_mine.V_bar_search(_mdp= m, _V_bar=V_vectors, lam_random= m.get_lambda())
    V = V_bar_search.V_bar_search(_mdp= m, _V_bar=V_vectors, lam_random= m.get_lambda())
    #-------------------------------------------------


    #_random_lambda_number is number of lambda random selected inside Lambda polytope
    start = time.clock()

    #-------------------------------------------------
    temp = V.v_optimal(_random_lambda_number = 1000)
    #temp = V.v_optimal()
    #-------------------------------------------------

    stop = time.clock()

    stop_all = time.clock()

    #cProfile.run('V.v_optimal(_random_lambda_number = 1000)')
    v_opt = temp[0]
    query = temp[1]


    print >> res, 'V_exact', exact
    res.flush()

    print >> res, 'v_opt', v_opt
    res.flush()

    print >> res, 'query counter', query
    res.flush()

    print >> res, 'error', np.dot(_lambda_rand, v_opt) - np.dot(_lambda_rand, exact)
    res.flush()

    print >> res, 'total time', stop_all-start_all

