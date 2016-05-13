import random
import  numpy as np
import cProfile

import sys
import time
import pickle
import Problem
import propagation_V

#-------------------------------------------------
#import V_bar_search
import V_bar_search_mine
#-------------------------------------------------

sys.path.insert(0, '/Users/pegah/Pycharm/Value_Iteration')
import m_mdp

ftype = np.float32

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
    global v_prog

    res = open("show_out" + ".txt", "w")

    _p = Problem.Problem(initial=[{s:[random.randint(0,na-1)] for s in range(n)}, np.zeros(d, dtype=ftype),
                     np.zeros((n,d),dtype=ftype)], _mdp=m, _cluster_error=cluster_v_bar_epsilon, _epsilon= epsilon_error)

    """the main problem is defined as a mdp with two errors: epsilon_error: the used error in hierarchical clustering
    using cosine similarity metric. cluster_v_bar_epsilon: the used error for stopping criteria in generating
    \mathcal{V} set."""
    _v_prog = propagation_V.propagation_V(m=m, cluster_v_bar_epsilon = cluster_v_bar_epsilon, epsilon_error = epsilon_error)

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
    v_prog = _v_prog

    name = "param" + _name + "v-propagate.dmp"
    pp = pickle.Pickler(open(name, 'w'))
    pp.dump(p)
    pp.dump(v_prog)
    pp.dump(V_vectors)

def relod_V_Vectors(_name):

    global V_vectors
    global p
    global v_prog

    name = "param" + _name + "v-propagate.dmp"
    pup = pickle.Unpickler(open(name, 'r'))

    p = pup.load()
    v_prog = pup.load()
    V_vectors = pup.load()

if __name__ == '__main__':

    res = open("search-V" + ".txt", "w")
    start_all = time.clock()

    n = 5
    na = 5
    d = 2

    _cluster_v_bar_epsilon = 0.1 #we fix it at 0.1
    epsilon_error = 0.2


    #load(n, na, d, "test")
    reload("test")

    # print "reload parameters*****"
    # print "state", _state
    # print "action", _action
    # print "d", _d
    # print "lambda_random", _lambda_rand
    # print "m",m
    # print "Uvec", Uvec


    #load_V_Vectors(_cluster_v_bar_epsilon, "test")
    relod_V_Vectors("test")

    #-------------------------------------------------
    V = V_bar_search_mine.V_bar_search(_mdp= m, _V_bar=V_vectors, lam_random= m.get_lambda())
    #V = V_bar_search.V_bar_search(_mdp= m, _V_bar=V_vectors, lam_random= m.get_lambda())
    #-------------------------------------------------


    #_random_lambda_number is number of lambda random selected inside Lambda polytope
    start = time.clock()

    #-------------------------------------------------
    #temp = V.v_optimal(_random_lambda_number = 1000)
    temp = V.v_optimal()
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

