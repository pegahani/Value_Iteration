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
        self.wen = open("output_weng" + ".txt", "w")

        """initialize linear programming as a minimization problem"""
        self.prob = cplex.Cplex()
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

        constr, rhs = [], []
        _d = self.mdp.d

        self.prob.variables.add(lb=[0.0] * _d, ub=[1.0] * _d)

        self.prob.set_results_stream(None)
        self.prob.set_log_stream(None)
        self.prob.set_warning_stream(None)

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
        # self.prob.write("show-Ldominance.lp")
        self.pareto = 0
        self.kd = 0
        self.queries = 0

    def get_initial_distribution(self):
        return self.mdp.initial_states_distribution()

    def get_Lambda(self):
        return self.mdp.get_lambda()

    # ****************************  comparison part ****************************************

    @staticmethod
    def pareto_comparison(a, b):
        a = np.array(a, dtype=ftype)
        b = np.array(b, dtype=ftype)

        assert len(a) == len(b), \
            "two vectors don't have the same size"

        return all(a >= b)

    def cplex_K_dominance_check(self, _V_best, Q):

        _d = len(_V_best)

        ob = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
        self.prob.objective.set_linear(ob)
        # self.prob.write("show-Ldominance.lp")
        self.prob.solve()

        result = self.prob.solution.get_objective_value()
        if result < 0.0:
            return False
        # print >> self.wen, _V_best - Q, ">> 0"
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
            for i in range(len(inequality_list)):
                first = True
                for x, y in zip(inequality_list[i], new_constraint)[1:]:
                    if x == 0 and y == 0:
                        continue
                    if first:
                        u, v = x , y
                        first = False
                    elif (u * y != x * v):
                        break
                else :
                    return True
        #print "new_constraint", new_constraint
        return False

    @staticmethod
    def generate_noise(_d, _noise_deviation):
        vector_noise = np.zeros(_d, dtype=ftype)
        for i in range(_d):
            vector_noise[i] = np.random.normal(0.0, _noise_deviation)

        return vector_noise

    def Query(self, _V_best, Q, noise):

        bound = [0.0]
        _d = len(_V_best)

        constr = []
        rhs = []

        Vscal = self.Lambda.dot(_V_best)
        Qscal = self.Lambda.dot(Q)
        # choice of the best policy
        if not noise:
            keep = (Vscal > Qscal)
        else:
            noise_value = random.gauss(0, Vscal * noise)
            keep = (Vscal + noise_value > Qscal )
        # Generating the new constraint in accordance
        if keep:
            new_constraints = bound + map(operator.sub, _V_best, Q)
        else:
            new_constraints = bound + map(operator.sub, Q, _V_best)

        # memorizing it
        if not self.is_already_exist(self.Lambda_inequalities, new_constraints):
            if keep:
                c = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
                print >> self.wen,  "Constrainte", self.query_counter_, _V_best - Q, "|> 0"
            else:
                c = [(j, float(Q[j] - _V_best[j])) for j in range(0, _d)]
                print >> self.wen, "Constrainte", self.query_counter_, Q - _V_best, "|> 0"
            self.query_counter_ += 1
            constr.append(zip(*c))
            rhs.append(0.0)
            self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
            self.Lambda_inequalities.append(new_constraints)

        # return the result
        if keep:
            return _V_best
        else:
            return Q



        # if not noise:
        #     if self.Lambda.dot(_V_best) > self.Lambda.dot(Q):
        #         new_constraints = bound+map(operator.sub, _V_best, Q)
        #         if not self.is_already_exist(self.Lambda_inequalities, new_constraints):
        #             c = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
        #             constr.append(zip(*c))
        #             rhs.append(0.0)
        #             self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
        #
        #             self.Lambda_inequalities.append(new_constraints)
        #             print >> self.wen,  "Constrainte", self.query_counter_, _V_best - Q, "|> 0"
        #             self.query_counter_ += 1
        #         return _V_best
        #
        #     else:
        #         new_constraints = bound+map(operator.sub, Q, _V_best)
        #         if not self.is_already_exist(self.Lambda_inequalities, new_constraints):
        #             c = [(j, float(Q[j] - _V_best[j])) for j in range(0, _d)]
        #             constr.append(zip(*c))
        #             rhs.append(0.0)
        #             self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
        #
        #             self.Lambda_inequalities.append(new_constraints)
        #             print >> self.wen, "Constrainte", self.query_counter_, Q - _V_best, "|> 0"
        #             self.query_counter_ += 1
        #         return Q
        # else:
        #
        #     noise_value = random.gauss(0, noise)
        #     #noise_vect = self.generate_noise(len(self.Lambda), noise)
        #     #V_best_noisy = noise_vect + _V_best
        #
        #     #if self.Lambda.dot(V_best_noisy)>self.Lambda.dot(Q):
        #     if self.Lambda.dot(_V_best)-self.Lambda.dot(Q) + noise_value > 0:
        #
        #         c = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
        #         constr.append(zip(*c))
        #         rhs.append(0.0)
        #         self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
        #
        #         self.Lambda_inequalities.append(bound+map(operator.sub, _V_best, Q))
        #         return _V_best
        #     else:
        #
        #         c = [(j, float(Q[j] - _V_best[j])) for j in range(0, _d)]
        #         constr.append(zip(*c))
        #         rhs.append(0.0)
        #         self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
        #
        #         self.Lambda_inequalities.append( bound+map(operator.sub, Q, _V_best))
        #         return Q
        #
        #         # return None

    def get_best(self, _V_best, Q, _noise):

        if ( _V_best == Q).all():
             return Q

        if self.pareto_comparison(_V_best, Q):
            self.pareto += 1
            return _V_best

        if self.pareto_comparison(Q, _V_best):
            self.pareto += 1
            return Q

        if self.cplex_K_dominance_check(Q, _V_best):
            self.kd += 1
            return Q

        if self.cplex_K_dominance_check(_V_best, Q):
            self.kd += 1
            return _V_best

        query = self.Query(_V_best, Q, _noise)
        self.queries += 1
        # self.query_counter_ += 1 ## seulement si existe --> dans Query

        return query

    # ****************************  comparison part ****************************************

    def value_iteration_weng(self, k, noise, threshold, exact, _error_exat_approx = None):
        """
        this function find the optimal v_bar of dimension d using Interactive value iteration method
        :param k: max number of iteration
        :param noise: user noise variance
        :param threshold: the stopping criteria value
        :param exact: the weight vector used to simulate user answers to queries.
        :return: it list f d-dimensional vectors after any posing any query to the user. the last vector in list is the
        optimal value solution of algorithm.
        """

        gather_query = []
        gather_diff = []
        self.query_counter_ = 0

        n, na, d = self.mdp.nstates, self.mdp.nactions, self.mdp.d
        Uvec_old_nd = np.zeros((n, d), dtype=ftype)
        Uvec_nd = np.zeros((n, d), dtype=ftype)

        delta = 0.0  # seems useless and harmless

        for t in range(k):
            #print t,
            if t % 50 == 0:
                print ""
            #Uvec_nd = np.zeros((n, d), dtype=ftype)

            for s in range(n):
                _V_best_d = np.zeros(d, dtype=ftype)
                for a in range(na):
                    # compute Q function
                    # Q_d = self.mdp.get_vec_Q(s, a, Uvec_old_nd)
                    Q_d = self.mdp.get_vec_Q(s, a, Uvec_nd)
                    _V_best_d = self.get_best(_V_best_d, Q_d, _noise=noise)

                Uvec_nd[s] = _V_best_d

                Uvec_temp = self.get_initial_distribution().dot(Uvec_nd)
                gather_query.append(self.query_counter_)
                gather_diff.append(abs( np.dot(self.get_Lambda(),Uvec_temp) - np.dot(self.get_Lambda(), exact)))


            Uvec_final_d = self.get_initial_distribution().dot(Uvec_nd)
            Uvec_old_d = self.get_initial_distribution().dot(Uvec_old_nd)
            delta = linfDistance([np.array(Uvec_final_d)], [np.array(Uvec_old_d)], 'chebyshev')[0, 0]

            #gather_query.append(self.query_counter_)
            #gather_diff.append(abs( np.dot(self.get_Lambda(),Uvec_final_d) - np.dot(self.get_Lambda(), exact)))

            #temporary: just for approximation project to harmonize a stopping criteria regarding aproximate error
            if _error_exat_approx:
                new_delta = gather_diff[-1]
                if new_delta < _error_exat_approx:
                    return Uvec_final_d, gather_query, gather_diff, t
                else:
                    Uvec_old_nd = Uvec_nd.copy()

            #gather_diff.append(linfDistance( [np.array(Uvec_final_d)] , [np.array(exact)], 'chebyshev')[0,0])
            #gather_diff.append(delta) # problem de side effect

            print >> self.wen, "iteration = ", t, "query =", gather_query[-1] , " error= ", gather_diff[-1],\
                " +" if (len(gather_diff) > 2 and gather_diff[-2] < gather_diff[-1]) else " "

            if not _error_exat_approx:
                if delta < threshold:
                    self.prob.write("show-LdominanceWeng.lp")
                    return Uvec_final_d, gather_query, gather_diff, t
                else:
                    Uvec_old_nd = Uvec_nd.copy()

        print >> self.wen,  "iteration = ", t, "query =", gather_query[-1] , " error= ", gather_diff[-1],\
        "+ " if (len(gather_diff) > 2 and gather_diff[-2] < gather_diff[-1]) else " "

        return Uvec_final_d, gather_query, gather_diff, t

# ********************************************
def generate_inequalities(_d):
    inequalities = []

    for x in itertools.combinations(xrange(_d), 1):
        inequalities.append([0] + [1 if i in x else 0 for i in xrange(_d)])
        inequalities.append([1] + [-1 if i in x else 0 for i in xrange(_d)])

    return inequalities

def interior_easy_points(dim):
    l = []
    for i in range(dim):
        l.append(random.uniform(0.0, 1.0))
    return l

