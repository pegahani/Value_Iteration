from Tkinter import *
import collections
import copy
from operator import add
import random
import cplex
import numpy as np
import itertools
import operator
import scipy.cluster.hierarchy as hac
import scipy.spatial.qhull
from scipy.spatial import ConvexHull


try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
    from scipy.spatial.distance import cdist as linfDistance
except:
    from sparse_mat import dok_matrix,csr_matrix,l1distance

ftype = np.float32

class weng:

    def __init__(self, _mdp, _lambda, _lambda_inequalities):

        self.mdp = _mdp
        self.Lambda = np.zeros(len(_lambda), dtype= ftype)
        self.Lambda[:] = _lambda

        self.Lambda_inequalities = _lambda_inequalities

        self.query_counter_ = 0

        """initialize linear programming as a minimization problem"""
        self.prob = cplex.Cplex()
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

        constr, rhs = [], []
        _d = self.mdp.d

        self.prob.variables.add(lb=[0.0] * _d, ub=[1.0] * _d)

        self.prob.set_results_stream(None)
        self.prob.set_log_stream(None)

        """add sum(lambda)_i = 1 on problem constraints"""
        c = [[j, 1.0] for j in range(0,_d)]
        constr.append(zip(*c))
        rhs.append(1)

        """inside this function E means the added constraint is an equality equation
        there are three options for sense:
        G: constraint is greater than rhs,
        L: constraint is lesser than rhs,
        E: constraint is equal to rhs"""

        self.prob.linear_constraints.add(lin_expr=constr, senses="E" * len(constr), rhs=rhs)
        self.prob.write("show-Ldominance.lp")

    def get_initial_distribution(self):
        return self.mdp.initial_states_distribution()

    def get_Lambda(self):
        return self.mdp.get_lambda()

    #****************************  comparison part ****************************************

    def pareto_comparison(self, a, b):
        a = np.array(a, dtype= ftype)
        b = np.array(b, dtype= ftype)

        assert len(a)==len(b), \
                "two vectors don't have the same size"

        return all(a>b)

    def cplex_K_dominance_check(self, _V_best, Q):

        _d = len(_V_best)

        ob = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
        self.prob.objective.set_linear(ob)
        self.prob.write("show-Ldominance.lp")
        self.prob.solve()

        result = self.prob.solution.get_objective_value()
        if result < 0.0:
            return False

        return True

    def is_already_exist(self, inequality_list, new_constraint):
        """

        :param inequality_list: list of inequalities. list of lists of dimension d+1
        :param new_constraint: new added constraint to list of inequalities of dimension d+1
        :return: True if new added constraint is already exist in list of constraints for Lambda polytope
        """

        if new_constraint in inequality_list:
                return True
        else:
                for i in range(2*self.mdp.d , len(inequality_list)):

                        devision_list = [np.float32(x/y) for x, y in zip(inequality_list[i], new_constraint)[1:]]
                        if all( x == devision_list[0] for x in devision_list):
                                return True

        return False

    def generate_noise(self, _d,  _noise_deviation):
        vector_noise = np.zeros(_d, dtype=ftype)
        for i in range(_d):
            vector_noise[i]= np.random.normal(0.0, _noise_deviation)

        return vector_noise

    def Query(self, _V_best, Q, noise):

        bound = [0.0]
        _d = len(_V_best)

        constr = []
        rhs = []

        if not noise:
            if self.Lambda.dot(_V_best) > self.Lambda.dot(Q):
                new_constraints = bound+map(operator.sub, _V_best, Q)
                if not self.is_already_exist(self.Lambda_inequalities, new_constraints):
                    c = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
                    constr.append(zip(*c))
                    rhs.append(0.0)
                    self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

                    self.Lambda_inequalities.append(new_constraints)

                return _V_best

            else:
                new_constraints = bound+map(operator.sub, Q, _V_best)
                if not self.is_already_exist(self.Lambda_inequalities, new_constraints):
                    c = [(j, float(Q[j] - _V_best[j])) for j in range(0, _d)]
                    constr.append(zip(*c))
                    rhs.append(0.0)
                    self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

                    self.Lambda_inequalities.append(new_constraints)

                return Q
        else:

            noise_value = random.gauss(0, noise)
            #noise_vect = self.generate_noise(len(self.Lambda), noise)
            #V_best_noisy = noise_vect + _V_best

            #if self.Lambda.dot(V_best_noisy)>self.Lambda.dot(Q):
            if self.Lambda.dot(_V_best)-self.Lambda.dot(Q) + noise_value > 0:

                c = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
                constr.append(zip(*c))
                rhs.append(0.0)
                self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

                self.Lambda_inequalities.append(bound+map(operator.sub, _V_best, Q))
                return _V_best
            else:

                c = [(j, float(Q[j] - _V_best[j])) for j in range(0, _d)]
                constr.append(zip(*c))
                rhs.append(0.0)
                self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

                self.Lambda_inequalities.append( bound+map(operator.sub, Q, _V_best))
                return Q

        #return None

    def get_best(self, _V_best, Q, _noise):

        #if ( _V_best == Q).all():
        #    return Q

        if self.pareto_comparison(_V_best, Q):
            return _V_best

        if self.pareto_comparison(Q, _V_best):
            return Q


        if self.cplex_K_dominance_check(Q, _V_best):
            return Q

        elif self.cplex_K_dominance_check(_V_best, Q):
            return _V_best


        query = self.Query(_V_best, Q, _noise)
        self.query_counter_ = self.query_counter_ + 1

        return query

    #****************************  comparison part ****************************************

    def value_iteration_weng(self, k, noise, threshold, exact):
        """
        this function find the optimal v_bar of dimension d using Interactive value iteration method
        :param k: max number of iteration
        :param noise: user noise variance
        :param threshold: the stopping criteria value
        :return: it list f d-dimensional vectors after any posing any query to the user. the last vector in list is the
        optimal value solution of algorithm.
        """

        gather_query = []
        gather_diff = []

        n, na, d =self.mdp.nstates , self.mdp.nactions, self.mdp.d
        Uvec_old_nd = np.zeros( (n,d) , dtype=ftype)

        delta = 0.0

        for t in range(k):
            Uvec_nd = np.zeros((n,d), dtype=ftype)

            for s in range(n):
                _V_best_d = np.zeros(d, dtype=ftype)
                for a in range(na):
                    #compute Q function
                    Q_d       = self.mdp.get_vec_Q(s, a, Uvec_old_nd)
                    _V_best_d = self.get_best(_V_best_d, Q_d, _noise= noise)

                Uvec_nd[s] = _V_best_d

            Uvec_final_d = self.get_initial_distribution().dot(Uvec_nd)
            Uvec_old_d = self.get_initial_distribution().dot(Uvec_old_nd)
            delta = linfDistance([np.array(Uvec_final_d)], [np.array(Uvec_old_d)], 'chebyshev')[0,0]

            gather_query.append(self.query_counter_)
            gather_diff.append(abs( np.dot(self.get_Lambda(),Uvec_final_d) - np.dot(self.get_Lambda(), exact)))

            if delta <threshold:
                return(Uvec_final_d, gather_query, gather_diff)
            else:
                Uvec_old_nd = Uvec_nd

        return(Uvec_final_d, gather_query, gather_diff)

#********************************************
def generate_inequalities(_d):
    inequalities = []

    for x in itertools.combinations( xrange(_d), 1 ) :
        inequalities.append([0] + [ 1 if i in x else 0 for i in xrange(_d) ])
        inequalities.append([1] + [ -1 if i in x else 0 for i in xrange(_d) ])

    return inequalities

def interior_easy_points(dim):
    l = []
    for i in range(dim):
        l.append(random.uniform(0.0, 1.0))
    return l

