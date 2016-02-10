import collections
import copy
from operator import add
import random
import cplex
import numpy as np
import itertools
import operator
import scipy.cluster.hierarchy as hac
import scipy.spatial.qhull as ssq
from scipy.spatial import ConvexHull

try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
    # noinspection PyPep8Naming
    from scipy.spatial.distance import cdist as linfDistance
except:
    from sparse_mat import dok_matrix, csr_matrix, l1distance

ftype = np.float32



class avi:
    """ A class for the advantage based value iteration algorithm.
        Embeds a VVMDP. Instance variables:
        mdp, Lambda, Lambda_generate_inequalities, query_counter_, query_counter_with_advantages"""

    def __init__(self, _mdp, _lambda, _lambda_inequalities):

        self.mdp = _mdp
        self.nstates = self.mdp.nstates
        self.nactions = self.mdp.nactions
        self.Lambda = np.zeros(len(_lambda), dtype=ftype)
        self.Lambda[:] = _lambda

        self.Lambda_inequalities = _lambda_inequalities
        self.query_counter_with_advantages = 0

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
        self.prob.write("show-Ldominance.lp")
        self.wen = open("output_AIweng" + ".txt", "w")

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

    def get_initial_distribution(self):
        return self.mdp.initial_states_distribution()

    def cluster_cosine_similarity(self, _advantages_dict, _cluster_threshold):

        """
        this function receives advantages and clusters them using a threshold inside each cluster
        :param _advantages_dict: dictionary of pairs (s,a):advantage vector of dimension d
        :param _cluster_threshold: max distance between two point in any cluster(cosine similarity distance)
        :return: dictionary o clusters such as {1: {(2, 0): [ 0.18869102,  0.], (2, 1):[ 0.18869102,  0.]},
                                                2: {(0, 1):[ 0.,  0.19183344], (1, 0): array([ 0.,  0.06188244]}
        """

        d = self.mdp.d
        cluster_advantages_dic = {}

        # checked outside cluster_cosine to cope with the empty result
        #  Points_dic = avi.clean_Points(_advantages_dict)
        points_array = np.zeros((len(_advantages_dict), d),
                                dtype=ftype)  # array of vectors of size d (the advantages to cluster)
        dic_labels = {}  # array of (s,a) pairs. The index in points_array is also the index of the (s,a) pair
                        # providing this advantage.

        counter = 0
        for key, val in _advantages_dict.iteritems():
            points_array[counter] = val
            dic_labels[counter] = key
            counter += 1

        # calls scipy.cluster.hierarchy
        z = hac.linkage(points_array, method='complete', metric='cosine')
        tol = -1e-16
        z.real[z.real < tol] = 0.0
        labels = hac.fcluster(z, _cluster_threshold, criterion='distance')

        # pyplot.scatter(points_array[:,0], points_array[:,1], c=labels)
        # pyplot.show()
        # rewrites the result of scipy.cluster.hierarchy in proper format: a dictionary of dictionaries
        for la in range(1, max(labels) + 1):
            cluster_advantages_dic.setdefault(la, {})

        for index, label in enumerate(labels):
            avi.update(cluster_advantages_dic, {label: {dic_labels[index]: points_array[index, :]}})

        return cluster_advantages_dic

    def make_convex_hull(self, _dic, _label):
        # change dictionary types to array and extract lists without their (s,a) pairs
        _points = []
        _pairs = []

        diclist = []
        for key, val in _dic.iteritems():
            diclist.append((key, val))

        if _label == 'V':
            for val in _dic.itervalues():
                _points.append(np.float32(val[1]))
        else:
            for key, val in _dic.iteritems():
                _points.append(np.float32(val))
                _pairs.append(key)

        _points = np.array(_points)

        try:
            #TODO change return value type
            hull = ConvexHull(_points)
            hull_vertices = hull.vertices
            hull_points = _points[hull_vertices, :]
        except scipy.spatial.qhull.QhullError:
            print 'convex hull is not available for label:', _label
            hull_points = _points

        return hull_points

    def keys_of_value(self, dct, _vector):

        """
        :param dct: dictionary of (s,a) key and d-dimensional value vectors
        :param _vector: a vector of dimension d
        :return: the key of given dictionary
        """

        for k, v in dct.iteritems():
            if (ftype(v) == ftype(_vector)).all():
                del dct[k]
                return k

    def get_advantages(self, _clustered_results_val):

        l = []
        for val in _clustered_results_val.itervalues():
            l.append(val)
        return np.array(l)

    def justify_cluster(self, _vectors_list, _clustered_pairs_vectors):
        """

        this function get list of vectors of d dimension and a dictionary of pairs and vectors.
        it reassigns pairs to vectors.

        :param _vectors_list: list of d-dimensional vectors
        :param _clustered_pairs_vectors: dictionary of (index, (pairs lists,vectors lists))
        :return: find related pair from _clustered_pairs_vectors to any vector from _convex_hull_vectors_list

        example:
           _clustered_pairs_vectors = {1:[[ 0.        ,  0.10899174],
           [ 0.        ,  0.10899174],
           [ 0.        ,  0.32242826]]}

           _convex_hull_vectors_list =  {1: {(0, 1): array([ 0.        ,  0.10899174], dtype=float32),
             (0, 0): array([ 0.        ,  0.10899174], dtype=float32), (2, 1): array([ 0.        ,  0.32242826], dtype=float32),
             (2, 0): array([ 0.        ,  0.32242826], dtype=float32), (1, 0): array([ 0.        ,  0.01936237], dtype=float32),
             (1, 1): array([ 0.        ,  0.01936237], dtype=float32)}}

            it returns: {1: ([(0, 1), (0, 0), (2, 1)], array([[ 0.        ,  0.10899174],
                       [ 0.        ,  0.10899174],
                       [ 0.        ,  0.32242826]], dtype=float32))}
        """
        # TODO Note fl: the reassignment is not deterministic: in the example, the answer could also be (0,1)(0,0)(2,0) ??

        _dic_pairs_vectors = {}

        for key in _vectors_list.iterkeys():
            if len(_vectors_list[key]) == 0:
                _dic_pairs_vectors[key] = ([k for k in _clustered_pairs_vectors[key].iterkeys()],
                                           self.get_advantages(_clustered_pairs_vectors[key]))
            else:
                policy = []
                for i in _vectors_list[key]:
                    policy.append(self.keys_of_value(_clustered_pairs_vectors[key], i))
                _dic_pairs_vectors[key] = (policy, _vectors_list[key])

        return _dic_pairs_vectors

    def sum_cluster_and_matrix(self, pair_vector_cluster, _matrix_nd):
        """
        this function receives dictionary of clusters including assigned pairs and advantages
        and nxd matrix
        :param pair_vector_cluster:  dictionary of clusters including assigned pairs and advantages in which
        advantages are vectors of dimension d
        :param _matrix_nd: a related matrix of dimension nxd
        :return: for each cluster if there is (s,a-i) and (s,a_j) choose one of them randomly and make sum on
        all related vectors in the same cluster after add beta.matrix_nd
        """

        n = self.nstates
        d = len(self.Lambda)

        final_dic = {}
        dic_clusters_sum_v_old = {}

        for key, val in pair_vector_cluster.iteritems():
            sum_d = np.zeros(d, dtype=ftype)
            pairs_list = []

            for i in range(n):
                selected_pairs = [val[0].index(pair) for pair in val[0] if pair[0] == i]

                if selected_pairs:
                    pair_index = random.choice(selected_pairs)
                    # pair_index = min(selected_pairs)
                    sum_d = map(add, sum_d, val[1][pair_index])
                    pairs_list.append(val[0][pair_index])

            final_dic[key] = (pairs_list, sum_d)

        for k, v in final_dic.iteritems():
            dic_clusters_sum_v_old[k] = (v[0], map(add, self.get_initial_distribution().dot(_matrix_nd), v[1]))

        return dic_clusters_sum_v_old

    def accumulate_advantage_clusters(self, _old_value_vector, _advantages, _cluster_threshold):

        """

        this function cluster advantages, make a convex hull on each cluster and returns back a dictionary
        of sum of vectors in each cluster and related pair of (state, action) in the same cluster

        :param _old_value_vector: a matrix of dimension nxd that will be added to improvements concluded from advantages
        :param _advantages: set of all generated advantages, each advantage is a vector of dimension d
        :param _cluster_threshold: max possible distance(cosine similarity distance) between two points in each cluster
        :return: returns back a dictionary of clusters including: key : value. Key is a counter of dictionary
                value is a pair like: ([(0, 1), (2, 0), (0, 0), (2, 1)], [0.0, 0.73071181774139404]) which first element
                are pairs and second element is sum on all related vectors + beta._matrix_nd
        """

        # _advantages = {(0, 1): np.array([ 0.,  0.], dtype=np.float32), (1, 2): np.array([ 1.,  2.], dtype=np.float32),
        #                (3, 2): np.array([ 0.,  0.], dtype=np.float32), (0, 0): np.array([ 0.,  0.], dtype=np.float32),
        #                (3, 1): np.array([ 0.,  0.], dtype=np.float32), (3, 3): np.array([ 0.,  0.], dtype=np.float32),
        #                (3, 0): np.array([ 0.,  0.], dtype=np.float32), (2, 2): np.array([ 2.,  2.], dtype=np.float32),
        #                (1, 1): np.array([ 1.,  1.], dtype=np.float32), (1, 4): np.array([ 1.,  4.], dtype=np.float32),
        #                (0, 2): np.array([ 0.,  0.], dtype=np.float32), (2, 0): np.array([ 2.,  0.], dtype=np.float32),
        #                (1, 3): np.array([ 1.,  3.], dtype=np.float32), (2, 3): np.array([ 2.,  3.], dtype=np.float32),
        #                (2, 1): np.array([ 2.,  1.], dtype=np.float32), (0, 4): np.array([ 0.,  0.], dtype=np.float32),
        #                (2, 4): np.array([ 0.,  0.], dtype=np.float32), (0, 3): np.array([ 0.,  0.], dtype=np.float32),
        #                (3, 4): np.array([ 0.,  0.], dtype=np.float32), (1, 0): np.array([ 1.,  0.], dtype=np.float32)}

        clustered_advantages = self.cluster_cosine_similarity(_advantages, _cluster_threshold)
        convex_hull_clusters = {}

        for key, val in clustered_advantages.iteritems():
            tempo = self.make_convex_hull(val, key)
            convex_hull_clusters[key] = tempo

        if bool(clustered_advantages):
            cluster_pairs_vectors = self.justify_cluster(convex_hull_clusters, clustered_advantages)
            sum_on_convex_hull_temp = self.sum_cluster_and_matrix(cluster_pairs_vectors, _old_value_vector)
            sum_on_convex_hull = {key: val for key, val in sum_on_convex_hull_temp.iteritems() if val[1]}

            return sum_on_convex_hull

        return {}

    def declare_policies(self, _policies, pi_p):
        """
        this function receives dictionary of state action pairs an related vector value improvements
        and returns back dictionary of policies related to given pairs and the same vector value improvement
        :param matrix_nd: UNUSED (only in commented part)
        :param _policies: dictionary of this form : {0: ((1, 0), (0, 1)), [ 1.20030463,  0.        ])
        :param pi_p: the given policy without counting improvement in accounts
        :return: dictionary of new policies and related improved vector values
        """

        _pi_p = pi_p.copy()
        V_append_d = np.zeros(self.mdp.d, dtype=ftype)

        new_policies = {}
        _pi_old = copy.deepcopy(_pi_p)

        for k, policy in _policies.iteritems():
            for key, val in _pi_p.iteritems():
                tempo = [item[1] for item in policy[0] if item[0] == key]
                if tempo:
                    _pi_p[key] = tempo
                    # else:
                    #    adv_d = self.get_initial_distribution()[key]*(self.mdp.get_vec_Q(key, _pi_old[key][0],  matrix_nd)-matrix_nd[key])
                    #    V_append_d = operator.add(V_append_d, adv_d)
                    #    print 'dakhele else V_append_d = ', V_append_d

            # V_append_d = np.zeros(self.mdp.d, dtype=ftype)

            #TODO check it again

            new_policies[k] = (_pi_p,np.float32(operator.add(policy[1], V_append_d)) ) #np.float32(policy[1]))

            #global log
            #print >> log, 'added vector ', policy[1], ' added advantage', np.dot(self.get_Lambda(), policy[1])

            _pi_p = copy.deepcopy(_pi_old)

        return new_policies

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
        self.prob.write("show-Ldominance.lp")
        self.prob.solve()

        result = self.prob.solution.get_objective_value()
        if result < 0.0:
            return False
        print >> self.wen, _V_best - Q, ">> 0"
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
                else:
                    c = [(j, float(Q[1][j] - _V_best[1][j])) for j in range(0, _d)]
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
            return _V_best

        if self.pareto_comparison(Q[1], _V_best[1]):
            return Q

        if self.cplex_K_dominance_check(Q[1], _V_best[1]):
            return Q

        elif self.cplex_K_dominance_check(_V_best[1], Q[1]):
            return _V_best

        query = self.Query_policies(_V_best, Q, _noise)

        # if this query is asked for value iteration with advantages
        self.query_counter_with_advantages += 1

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

        log = open("output_avi" + ".txt", "w")

        gather_query = []
        gather_diff = []

        d = self.mdp.d
        currentUvecs_nd = np.zeros((self.nstates, d), dtype=ftype) # initial value vector per state
        previousvalue_d = np.zeros(d, dtype=ftype) # a value vector

        # initial policy-value node:
        best_policyvaluepair = (
            {s: [random.randint(0, self.nactions - 1)] for s in range(self.nstates)}, np.zeros(d, dtype=ftype))
        currenvalue_d = best_policyvaluepair[1]

        # limit = 1
        for t in range(limit):

            advantages_dic = self.mdp.calculate_advantages_dic(currentUvecs_nd, False, best_policyvaluepair[0])
            advantages_dic = avi.clean_Points(advantages_dic)
            if advantages_dic == {}:
                return currenvalue_d, gather_query, gather_diff
            clusters_dic = self.accumulate_advantage_clusters(currentUvecs_nd, advantages_dic, cluster_threshold)
            policies = self.declare_policies(clusters_dic, best_policyvaluepair[0], currentUvecs_nd)

            for val in policies.itervalues():
                best_policyvaluepair = self.get_best_policies(best_policyvaluepair, val, noise)

            currentUvecs_nd = self.mdp.update_matrix(policy_p=best_policyvaluepair[0], _Uvec_nd=currentUvecs_nd)
            currenvalue_d = best_policyvaluepair[1]

            delta = linfDistance([np.array(currenvalue_d)], [np.array(previousvalue_d)], 'chebyshev')[0, 0]

            gather_query.append(self.query_counter_with_advantages)
            gather_diff.append(abs(self.Lambda.dot(currenvalue_d) - self.Lambda.dot(exact)))

            print >> self.wen,  "iteration = ", t, "query =", gather_query[len(gather_query)-1] , " error= ", gather_diff[len(gather_diff)-1], \
                " +" if (len(gather_diff) > 2 and gather_diff[len(gather_diff)-2] < gather_diff[len(gather_diff)-1]) else " "

            if delta < min_change:
                return currenvalue_d, gather_query, gather_diff
            else:
                previousvalue_d = currenvalue_d.copy()

        print >> self.wen,  "iteration = ", t, "query =", gather_query[len(gather_query)-1] , " error= ", gather_diff[len(gather_diff)-1],\
            " +" if (len(gather_diff) > 2 and gather_diff[len(gather_diff)-2] < gather_diff[len(gather_diff)-1]) else " "

        # noinspection PyUnboundLocalVariable
        return currenvalue_d, gather_query, gather_diff

# ********************************************
    @staticmethod
    def clean_Points(_points):
        """ returns a copy of _points where all pairs having as value the vector \bar 0 are deleted
        :param _points: a dictionary where values are vectors
        :rtype: dictionary"""
        _dic = {}
        for key, value in _points.iteritems():
            # if not np.all(value == 0):
            if np.any(value): #  avoids a syntax warning
                _dic[key] = value
        return _dic

    @staticmethod
    def make_convex_hull(_dic, _label):
        """
        :param _dic: avantages dictionary of a given cluster
        :param _label: the key of the cluster
        :return: [an array of advantages] avantages dictionnary of the convex hull
        """
        # change dictionary types to array and extract lists without their (s,a) pairs
        # TODO change the return type to avoid the work in justify_cluster: at the end of the try, _pairs[hull_vertices]
        # are the keys of hull_points
        _points = []
        _pairs = []

        diclist = []  # TODO diclist is unused
        for key, val in _dic.iteritems():
            diclist.append((key, val))

        if _label == 'V': # TODO I do not cope with this case, which does not seem to happen ??
            for val in _dic.itervalues():
                _points.append(np.float32(val[1]))
        else:
            for key, val in _dic.iteritems():
                _points.append(np.float32(val))
                _pairs.append(key)

        _points = np.array(_points)
        _pairs = np.array(_pairs)

        try:
            hull = ConvexHull(_points)
            hull_vertices = hull.vertices
            hull_points = _points[hull_vertices, :]
            hull_pairs = _pairs[hull_vertices]
        except ssq.QhullError:
            print 'convex hull is not available for label:', _label
            hull_points = _points
            hull_pairs = _pairs

        return hull_points
        # CH_advantages_dic = {k:v for (k,v) in zip(hull_pairs, hull_points)}
        # return CH_advantages_dic

    @staticmethod
    def keys_of_value(dct, _vector):

        """
        :param dct: dictionary of (s,a) key and d-dimensional value vectors
        :param _vector: a vector of dimension d
        :return: the key of _vector in the given dictionary, AFTER DELETING IT from dct
        """

        for k, v in dct.iteritems():
            # noinspection PyUnresolvedReferences
            if (ftype(v) == ftype(_vector)).all():
                del dct[k]
                return k

    @staticmethod
    def get_advantages(_clustered_results_val):

        l = []
        for val in _clustered_results_val.itervalues():
            l.append(val)
        return np.array(l)

    @staticmethod
    def generate_noise(_d, _noise_deviation):
        vector_noise = np.zeros(_d, dtype=ftype)
        for i in range(_d):
            vector_noise[i] = np.random.normal(0.0, _noise_deviation)

        return vector_noise

    @staticmethod
    def update( dic, _u):
        """ Update a dictionary with new values. If the value is itself a mapping, the old value of key is supposed
            to be itself a dictionary which is recursively updated
            :param _u: new (key,value) pairs to replace existing (if key exists) or add (else) in dic
            :param dic: the dictionary to update """
        for k, v in _u.iteritems():
            if isinstance(v, collections.Mapping):
                r = avi.update(dic.get(k, {}), v)
                dic[k] = r
            else:
                dic[k] = _u[k]
        return dic

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



# ********************************************

def generate_inequalities(_d):
    inequalities = []

    for x in itertools.combinations(xrange(_d), 1):
        inequalities.append([0] + [1 if i in x else 0 for i in xrange(_d)])
        inequalities.append([1] + [-1 if i in x else 0 for i in xrange(_d)])

    return inequalities


