import random
import  numpy as np

import sys
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
    return l

if __name__ == '__main__':

    n = 4
    na = 5
    d=2

    _cluster_v_bar_epsilon = 0.1
    epsilon_error = 0.1

    _lambda_rand = interior_easy_points(d)
    #_lambda_rand = [0.8410933688420644, 0.6431895253761406]

    m = m_mdp.make_grid_VVMDP(_lambda_rand, n=2)
    #m = m_mdp.make_simulate_mdp_Yann(n, na, _lambda_rand, None)

    #m.set_Lambda(_lambda_rand)

    Uvec = m.policy_iteration()
    exact = m.initial_states_distribution().dot(Uvec)

    p = Problem.Problem(initial=[{s:[random.randint(0,na-1)] for s in range(n)}, np.zeros(d, dtype=ftype),
                    np.zeros((n,d),dtype=ftype)], _mdp=m, _cluster_error=epsilon_error, _epsilon= _cluster_v_bar_epsilon)

    """the main problem is defined as a mdp with two errors: epsilon_error: the used error in hierarchical clustering
     using cosine similarity metric. cluster_v_bar_epsilon: the used error for stopping criteria in generating
     \mathcal{V} set."""
    v_prog = propagation_V.propagation_V(m=m, cluster_v_bar_epsilon = _cluster_v_bar_epsilon, epsilon_error = epsilon_error)
    V_vectors = v_prog.convex_hull_search(p)

    print 'V_vectors', V_vectors
    print 'length of V_vectors', len(V_vectors)
    print 'V_exact', exact
    #print "lambda random", _lambda_rand

    #V_vectors = [np.array([ 7.93087196,  1.17448664]), np.array([ 8.90109158,  9.95837021]),
    #             np.array([ 8.79565525,  9.99999332]), np.array([  4.76846984e-03,   9.99999332e+00]),
    #             np.array([  4.29162290e-03,   9.10988331e+00])]


    #V = V_bar_search.V_bar_search(_mdp= m, _V_bar=V_vectors, lam_random= m.get_lambda())
    #v_opt = V.v_optimal()

