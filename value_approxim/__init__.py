import random
import  numpy as np
import cProfile

import sys
import time
import Problem
import propagation_V
import V_bar_search

sys.path.insert(0, '/Users/pegah/Pycharm/Value_Iteration')
import m_mdp

ftype = np.float32

def interior_easy_points(dim):
    #dim = len(self.points[0])
    l = []
    for i in range(dim):
        l.append(random.uniform(0.0, 1.0))
    return np.array(l, dtype=ftype)

if __name__ == '__main__':

    res = open("show_out" + ".txt", "w")
    start_all = time.clock()

    n = 128
    na = 5
    d=3

    _cluster_v_bar_epsilon = 0.1 #we fix it at 0.1
    epsilon_error = 0.3

    _lambda_rand = interior_easy_points(d)

    #m = m_mdp.make_grid_VVMDP(_lambda_rand, n=2)
    m = m_mdp.make_simulate_mdp_Yann(n, na, _lambda_rand, None)

    #m.set_Lambda(_lambda_rand)

    Uvec = m.policy_iteration()
    exact = m.initial_states_distribution().dot(Uvec)

    p = Problem.Problem(initial=[{s:[random.randint(0,na-1)] for s in range(n)}, np.zeros(d, dtype=ftype),
                     np.zeros((n,d),dtype=ftype)], _mdp=m, _cluster_error=_cluster_v_bar_epsilon, _epsilon= epsilon_error)

    """the main problem is defined as a mdp with two errors: epsilon_error: the used error in hierarchical clustering
    using cosine similarity metric. cluster_v_bar_epsilon: the used error for stopping criteria in generating
    \mathcal{V} set."""
    v_prog = propagation_V.propagation_V(m=m, cluster_v_bar_epsilon = _cluster_v_bar_epsilon, epsilon_error = epsilon_error)

    start1 = time.clock()
    tem = v_prog.convex_hull_search(p)
    stop1 = time.clock()

    cProfile.run('v_prog.convex_hull_search(p)')

    print >> res, 'time v propagation',stop1- start1

    print >> res, 'iteration', tem[1]
    res.flush()

    V_vectors = tem[0]

    print >> res, 'V_vectors', V_vectors
    res.flush()
    print >> res, 'len(V_vectors)', len(V_vectors)
    res.flush()

    V = V_bar_search.V_bar_search(_mdp= m, _V_bar=V_vectors, lam_random= m.get_lambda())
    #_random_lambda_number is number of lambda random selected inside Lambda polytope
    start = time.clock()
    temp = V.v_optimal(_random_lambda_number = 1000)
    stop = time.clock()

    stop_all = time.clock()

    print >> res, "optimal v_bar time", stop-start

    cProfile.run('V.v_optimal(_random_lambda_number = 1000)')
    v_opt = temp[0]

    print >> res, 'V_vectors', V_vectors
    res.flush()
    print >> res, 'len(V_vectors)', len(V_vectors)
    res.flush()
    print >> res, 'V_exact', exact
    res.flush()
    print >> res, "lambda random", _lambda_rand
    res.flush()

    print >> res, 'V_exact', exact
    res.flush()

    print >> res, 'v_opt', v_opt
    res.flush()

    print >> res, 'query counter', temp[1]
    res.flush()

    print >> res, 'error', np.dot(_lambda_rand, v_opt) - np.dot(_lambda_rand, exact)
    res.flush()

    print >> res, 'total time', stop_all-start_all

