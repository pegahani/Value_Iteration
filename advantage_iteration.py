import collections
import copy
import random
import cplex
import numpy as np
import itertools
import operator
import scipy.cluster.hierarchy as hac
import scipy.spatial.qhull as ssq
from scipy.spatial import ConvexHull
import advantage


try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
    from scipy.spatial.distance import cdist as linfDistance
except:
    from sparse_mat import dok_matrix, csr_matrix, l1distance

ftype = np.float32

hullsuccess = 0
hullexcept = 0


class avi:

    def __init__(self, _mdp, _lambda, _lambda_inequalities):

        self.mdp = _mdp
        self.nstates = self.mdp.nstates
        self.nactions = self.mdp.nactions
        self.Lambda = np.zeros(len(_lambda), dtype=ftype)
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
        self.prob.set_warning_stream(None)

        """add sum(lambda)_i = 1 on problem constraints"""
        c = [[j, 1.0] for j in range(0,_d)]
        constr.append(zip(*c))
        rhs.append(1)

        """inside this function E means the added constraint is an equality equation
        there are three options for sense:
        G: constraint is greater than rhs,
        L: constraint is lesser than rhs,
        E: constraint is equal than rhs"""

        self.prob.linear_constraints.add(lin_expr=constr, senses="E" * len(constr), rhs=rhs)
        # self.prob.write("show-CPconfig.lp")
        self.wen = open("output_avi" + ".txt", "w")
        self.pareto = 0
        self.kd = 0
        self.queries = 0
        self.nbclusters = 0

    def reset(self, _mdp, _lambda, _lambda_inequalities):
        self.mdp = _mdp
        self.Lambda = np.zeros(len(_lambda), dtype=ftype)
        self.Lambda[:] = _lambda

        self.Lambda_inequalities = _lambda_inequalities

        self.query_counter_ = 0
        self.query_counter_with_advantages = 0

        # reset prob for the new example

    # def setStateAction(self):
    #     self.n = self.mdp.nstates
    #     self.na = self.mdp.nactions

    def get_Lambda(self):
        return self.mdp.get_lambda()

    # def justify_cluster(self, _vectors_list_dict, _clustered_pairs_vectors):
    #     # TODO now unused
    #     """
    #     this function get list of vectors of d dimension and a dictionary of pairs and vectors.
    #     it reassigns pairs to vectors.
    #
    #     :param _vectors_list_dict: dictionary of cluster_id, list of vectors (the convex hull)
    #     :param _clustered_pairs_vectors: dictionary of (index, (pairs lists,vectors lists))
    #     :return: find related pair from _clustered_pairs_vectors to any vector from _convex_hull_vectors_list
    #
    #     example:
    #        _clustered_pairs_vectors = {1:[[ 0.        ,  0.10899174],
    #        [ 0.        ,  0.10899174],
    #        [ 0.        ,  0.32242826]]}
    #
    #        _convex_hull_vectors_list =  {1: {(0, 1): array([ 0.        ,  0.10899174], dtype=float32),
    #          (0, 0): array([ 0.        ,  0.10899174], dtype=float32), (2, 1): array([ 0.        ,  0.32242826], dtype=float32),
    #          (2, 0): array([ 0.        ,  0.32242826], dtype=float32), (1, 0): array([ 0.        ,  0.01936237], dtype=float32),
    #          (1, 1): array([ 0.        ,  0.01936237], dtype=float32)}}
    #
    #         it returns: {1: ([(0, 1), (0, 0), (2, 1)], array([[ 0.        ,  0.10899174],
    #                    [ 0.        ,  0.10899174],
    #                    [ 0.        ,  0.32242826]], dtype=float32))}
    #     """
    #     # TODO Note fl: the reassignment is not deterministic: in the example, the answer could also be (0,1)(0,0)(2,0) ??
    #
    #     _dic_pairs_vectors = {}
    #
    #     for key in _vectors_list_dict.iterkeys():
    #         if len(_vectors_list_dict[key]) == 0:
    #             _dic_pairs_vectors[key] = ([k for k in _clustered_pairs_vectors[key].iterkeys()],
    #                                        self.get_advantages(_clustered_pairs_vectors[key]))
    #         else:
    #             policy = []
    #             for i in _vectors_list_dict[key]:
    #                 policy.append(self.keys_of_value(_clustered_pairs_vectors[key], i))
    #             _dic_pairs_vectors[key] = (policy, _vectors_list_dict[key])
    #
    #     return _dic_pairs_vectors

    # *********************** comparison part **************************

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

    def already_exists(self, inequality_list, new_constraint):
        """
        :param inequality_list: list of inequalities. list of lists of dimension d+1
        :param new_constraint: new added constraint to list of inequalities of dimension d+1
        :return: True if new added constraint already exists in list of constraints for Lambda polytope
        """
        if new_constraint in inequality_list:
            return True
        else:
            # TODO correct the range now that the 0 <x<1 are in bounds
            # for i in range(2 * self.mdp.d, len(inequality_list)): # at start, skips the cube defining constraints
            for i in range(len(inequality_list)):

                division_list = [np.float32(x / y) for x, y in zip(inequality_list[i], new_constraint)[1:]if not y == 0]
                if all(x == division_list[0] for x in division_list):
                    return True
                # thisone = True
                # for j0 in range (self.mdp.d):
                #     if inequality_list[i][j0] != 0 or new_constraint[j0+1] != 0 :
                #         break
                # for j in range (j0 + 1, self.mdp.d):
                #     if not inequality_list[i][j0] * new_constraint[j + 1] - inequality_list[i][j] * new_constraint[
                #                 j0 + 1] == 0:
                #         thisone = False
                #         break
                # if thisone == True:
                #     return thisone

        return False

    def generate_noise(self, _d, _noise_deviation):
        vector_noise = np.zeros(_d, dtype=ftype)
        for i in range(_d):
            vector_noise[i] = np.random.normal(0.0, _noise_deviation)

        return vector_noise

    def Query_policies(self, _V_best, Q, noise):
        """ simulates a user query, which must answer if _V_best is prefered to Q or the rverse
        :param _V_best: one dimensional vector of size d
        :param Q: same format as _V_best
        :param noise: True or False
        :return: the prefered vector.  As a side effect, adds a constraint representing the preference in prob.
        """

        bound = [0.0]
        _d = len(_V_best[1])

        constr = []
        rhs = []

        if not noise:
            keep = (self.Lambda.dot(_V_best[1]) > self.Lambda.dot(Q[1]))
            if keep:
                new_constraints = bound + map(operator.sub, _V_best[1], Q[1])
            else:
                new_constraints = bound + map(operator.sub, Q[1], _V_best[1])
            if not self.already_exists(self.Lambda_inequalities, new_constraints):
                if keep:
                    c = [(j, float(_V_best[1][j] - Q[1][j])) for j in range(0, _d)]
                    print >> self.wen,  "Constrainte", self.query_counter_, _V_best[1] - Q[1], "|> 0"
                else:
                    c = [(j, float(Q[1][j] - _V_best[1][j])) for j in range(0, _d)]
                    print >> self.wen, "Constrainte", self.query_counter_, Q[1] - _V_best[1], "|> 0"
                self.query_counter_ += 1
                constr.append(zip(*c))
                rhs.append(0.0)
                self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
                self.Lambda_inequalities.append(new_constraints)
            if keep:
                return _V_best
            else:
                return Q

        #noise_vect = self.generate_noise(len(self.Lambda), noise)
        #V_best_noisy = noise_vect + _V_best[1]

        noise_value = random.gauss(0, noise)

        #if Lambda_noisy.dot(_V_best[1]) > Lambda_noisy.dot(Q[1]):
        if self.Lambda.dot(_V_best[1])-self.Lambda.dot(Q[1]) + noise_value > 0:
            #print >>log, "correct response",self.Lambda.dot(_V_best[1]) > self.Lambda.dot(Q[1]), "wrong response",True

            c = [(j, float(_V_best[1][j] - Q[1][j])) for j in range(0, _d)]
            constr.append(zip(*c))
            rhs.append(0.0)
            self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

            self.Lambda_inequalities.append(bound+map(operator.sub, _V_best[1], Q[1]))
            return _V_best
        else:
            #print >>log, "correct response",self.Lambda.dot(_V_best[1]) > self.Lambda.dot(Q[1]), "wrong response",False

            c = [(j, float(Q[1][j] - _V_best[1][j])) for j in range(0, _d)]
            constr.append(zip(*c))
            rhs.append(0.0)
            self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

            self.Lambda_inequalities.append( bound+map(operator.sub, Q[1], _V_best[1]))
            return Q

    def get_best_policies(self, _V_best, Q, _noise):

        if self.pareto_comparison(_V_best[1], Q[1]):
            self.pareto += 1
            return _V_best

        if self.pareto_comparison(Q[1], _V_best[1]):
            self.pareto += 1
            return Q

        if self.cplex_K_dominance_check(Q[1], _V_best[1]):
            self.kd += 1
            return Q

        elif self.cplex_K_dominance_check(_V_best[1], Q[1]):
            self.kd += 1
            return _V_best

        query = self.Query_policies(_V_best, Q, _noise)
        self.queries += 1


        return query

    # *********************** comparison part **************************

    def value_iteration_with_advantages(self, limit, noise, cluster_threshold, min_change, exact):
        """
        best_policyvaluepair is a pair made of a dictionary of state:action items and a value vector of size d.
        :param limit: max number of iterations
        :param noise: a vector of size d, none if no noise
        :param cluster_threshold: the threshold to build clusters (max distance between two of its vectors)
        :param min_change: iteration stops when the value changes less than this min
        :param exact: the weights (lambda vector) used to simulate users answers to queries.
        :return:
        """

        gather_query = []
        gather_diff = []
        gather_clusters = []
        self.adv = advantage.Advantage(self.mdp, cluster_threshold)

        d = self.mdp.d
        currentUvecs_nd = np.zeros((self.nstates, d), dtype=ftype) # initial value vector per state
        previousvalue_d = np.zeros(d, dtype=ftype) # a value vector

        # initial policy-value node:
        best_policyvaluepair = [{s: [random.randint(0, self.nactions - 1)] for s in range(self.nstates)},
                                np.zeros(d, dtype=ftype)]
        currenvalue_d = best_policyvaluepair[1]

        # limit = 1
        for t in range(limit):
            # computes all the advantages in a dictionary {(state, action):vector ...}
            advantages_dic = self.mdp.calculate_advantages_dic(currentUvecs_nd, True)
            # removes advantages equal to vector 0
            advantages_dic = self.adv.clean_Points(advantages_dic)
            if advantages_dic == {}:
                print "dictionaire vide"
                return currenvalue_d, gather_query, gather_diff
            # feeds into internal class format
            advantages_dic = self.adv.AdvantagesDict(advantages_dic)
            # computes a dictionary of clusters, where each cluster is a pair ([(s,a)...], V) (the list of (s,a) in the
            # cluster, and the sum of the (vectorial) advantages and the previous \beta(s) \dot \bar V(s)
            clusters_dic = self.adv.accumulate_advantage_clusters(currentUvecs_nd, advantages_dic, cluster_threshold)
            # policies = self.declare_policies(clusters_dic, best_policyvaluepair[0], currentUvecs_nd)
            # only replaces actions in the best policy by actions in the cluster when their state is the same
            policies = self.adv.declare_policies(clusters_dic, best_policyvaluepair[0])
            
            # after merge Pegah***
            #advantages_pair_vector_dic = self.mdp.calculate_advantages_labels(matrix_nd, True)
            #cluster_advantages = self.adv.accumulate_advantage_clusters(matrix_nd, advantages_pair_vector_dic,
            #                                                        cluster_error)
            #policies = self.adv.declare_policies(cluster_advantages, best_p_and_v_d[0])
            # after merge Pegah***

            # Updates the best (policy, value) pair. The value inherited from the previous iteration is fist cleaned
            # to protects against keeping the (policy, value) pair from previous iteration
            best_policyvaluepair = [best_policyvaluepair[0], np.zeros(d, dtype=ftype)]
            for val in policies.itervalues():
                best_policyvaluepair = self.get_best_policies(best_policyvaluepair, val, noise)

            print t, ":", len(best_policyvaluepair[0]),
            if t%25 == 0:
                print
            currentUvecs_nd = self.mdp.update_matrix(policy_p=best_policyvaluepair[0], _Uvec_nd=currentUvecs_nd)
            currenvalue_d = best_policyvaluepair[1]

            delta = linfDistance([np.array(currenvalue_d)], [np.array(previousvalue_d)], 'chebyshev')[0, 0]

            gather_query.append(self.query_counter_)
            gather_diff.append(self.Lambda.dot(exact) - self.Lambda.dot(currenvalue_d))
            gather_clusters.append(self.nbclusters)

            print >> self.wen,  "iteration = ", t, "query =", gather_query[len(gather_query)-1] , \
                "clusters =", self.nbclusters, "error= ", gather_diff[len(gather_diff)-1], \
                " +" if (len(gather_diff) > 2 and gather_diff[-2] < gather_diff[-1]) else " "

            if delta < min_change:
                print "\n", exact
                print currenvalue_d
                print self.adv.get_initial_distribution().dot(currentUvecs_nd)
                return currenvalue_d, gather_query, gather_diff, hullsuccess, hullexcept, t
            else:
                previousvalue_d = currenvalue_d.copy()

        print >> self.wen,  "iteration = ", t, "query =", gather_query[-1] ,  \
                "clusters =", self.nbclusters," error= ", gather_diff[-1],\
            " +" if (len(gather_diff) > 2 and gather_diff[-2] < gather_diff[-1]) else " "

        # noinspection PyUnboundLocalVariable
        return currenvalue_d, gather_query, gather_diff, hullsuccess, hullexcept, t

#************ noise ************************
    @staticmethod
    def generate_noise(_d, _noise_deviation):
        vector_noise = np.zeros(_d, dtype=ftype)
        for i in range(_d):
            vector_noise[i] = np.random.normal(0.0, _noise_deviation)

        return vector_noise

# ********************************************
    @staticmethod
    def generate_inequalities(_d):
        inequalities = []
        for x in itertools.combinations(xrange(_d), 1):
            inequalities.append([0] + [1 if i in x else 0 for i in xrange(_d)])
            inequalities.append([1] + [-1 if i in x else 0 for i in xrange(_d)])
        return inequalities

    @staticmethod
    def interior_easy_points(dim):
        # dim = len(self.points[0])
        l = []
        s = 0
        for i in range(dim):
            l.append(random.uniform(0.0, 1.0))
            s += l[i]
        for i in range(dim):
            l[i] /= s
        return l



