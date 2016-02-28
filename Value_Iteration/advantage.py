#Francois functions

import collections
import copy
from operator import add
import random
import numpy as np
import operator
import scipy.cluster.hierarchy as hac
import scipy.spatial.qhull as ssq
from scipy.spatial import ConvexHull

try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
    from scipy.spatial.distance import cdist as linfDistance
except:
    from sparse_mat import dok_matrix, csr_matrix, l1distance

ftype = np.float32

class Advantage:

    """ A class for the advantage based value iteration algorithm.
    Embeds a VVMDP. Instance variables:
    mdp, Lambda, Lambda_generate_inequalities, query_counter_, query_counter_with_advantages"""
    class AdvantagesDict:
        """
        avantages dictionary: keys are pairs (state,action), their value is an avantage vector
        """
        def __init__(self, dico = None):
            if dico:
                self.elems = dico
            else:
                self.elems = {}
        def set(self, s_a_pair, v):
            self.elems[s_a_pair] = v
        def get(self, s_a_pair):
            return self.elems[s_a_pair]
        def getdefault(self, s_a_pair,v=None):
            return self.elems.get(s_a_pair,v) # v by default
        def size(self):
            return len(self.elems)
        def __iter__(self):
            return self.elems.__iter__()
        def iteritems(self):
            for i in self.elems.iteritems():
                yield i
        def itervalues(self):
            for i in self.elems.itervalues():
                yield i
        def iterkeys(self):
            for i in self.elems.iterkeys():
                yield i

    def __init__(self, _mdp, _cluster_threshold):
        self.mdp = _mdp
        self.cluster_error = _cluster_threshold

    def update(self, dic, _u):
        """ Update a dictionary with new values. If the value is itself a mapping, the old value of key is supposed
            to be itself a dictionary which is recursively updated
            :param _u: new (key,value) pairs to replace existing (if key exists) or add (else) in dic
            :param dic: the dictionary to update """
        for k, v in _u.iteritems():
            if isinstance(v, collections.Mapping):
                r = self.update(dic.get(k, {}), v)
                dic[k] = r
            else:
                dic[k] = _u[k]
        return dic

    def get_initial_distribution(self):
        return self.mdp.initial_states_distribution()

    def clean_Points(self, _points):
        """ returns a copy of _points where all pairs having as value the vector \bar 0 are deleted
        :param _points: a dictionary where values are vectors
        :rtype: dictionary"""
        _dic = {}
        for key, value in _points.iteritems():
            # if not np.all(value == 0):
            if np.any(value): #  avoids a syntax warning
                _dic[key] = value
        return _dic

    def cluster_cosine_similarity(self, _advantages_dict, _cluster_threshold):

        """
        this function receives advantages and clusters them using a threshold inside each cluster
        :param _advantages_dict: object of class avi.avantages_dict, with all the advantages to cluster
        :param _cluster_threshold: max distance between two point in any cluster(cosine similarity distance)
        :return: dictionary of advantages_dict: the key is the cluster id, and the value is an advantages_dict with
        all the advantages of this cluster
        """

        d = self.mdp.d
        cluster_advantages_dic = {}

        advantages_array = np.zeros((_advantages_dict.size(), d),
                                dtype=ftype)  # array of vectors of size d (the advantages to cluster)
        sa_pairs_array = []  # array of (s,a) pairs. The index in points_array is also the index of the (s,a) pair
                        # providing this advantage.

        counter = 0
        for state_act, advantage in _advantages_dict.iteritems():
            advantages_array[counter] = advantage
            sa_pairs_array.append(state_act)
            counter += 1

        # calls scipy.cluster.hierarchy
        z = hac.linkage(advantages_array, method='complete', metric='cosine')
        tol = -1e-16
        z.real[z.real < tol] = 0.0

        if z.size == 0:
            print "z", z
            return {}

        labels = hac.fcluster(z, _cluster_threshold, criterion='distance')

        # pyplot.scatter(points_array[:,0], points_array[:,1], c=labels)
        # pyplot.show()

        # rewrites the result of scipy.cluster.hierarchy in proper format: a dictionary of dictionaries
        for la in range(1, max(labels) + 1):
            cluster_advantages_dic.setdefault(la, Advantage.AdvantagesDict())

        for index, label in enumerate(labels):
            cluster_advantages_dic[label].set(sa_pairs_array[index], advantages_array[index, :])
        self.nbclusters = len(cluster_advantages_dic)
        # maxlen = max([d.size() for d in cluster_advantages_dic.itervalues()])
        # print self.nbclusters, maxlen
        return cluster_advantages_dic

    def keys_of_value(self, dct, _vector):

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

    def get_advantages(self, _clustered_results_val):

        l = []
        for val in _clustered_results_val.itervalues():
            l.append(val)
        return np.array(l)

    def make_convex_hull(self, _dic, _label):
        """
        :param _dic: AdvantagesDic of a given cluster
        :param _label: the key of the cluster
        :return: AdvantagesDic of the convex hull
        """
        # change dictionary types to array and extract lists without their (s,a) pairs
        # TODO change the return type to avoid the work in justify_cluster: at the end of the try, _pairs[hull_vertices]
        # are the keys of hull_points
        _points = []
        _pairs = []
        global hullsuccess
        global hullexcept

        if _label == 'V': # TODO I do not cope with this case, which does not seem to happen ??
            for val in _dic.itervalues():
                _points.append(np.float32(val[1]))
        else:
            for key, val in _dic.iteritems():
                _points.append(np.float32(val))
                _pairs.append(key)

        _points = np.array(_points)
        _pairs = np.array(_pairs)

        # hull.vertices collects the indexes of of selected _points, and also of their s_a_pair.
        try:
            hull = ConvexHull(_points)
            hull_vertices = hull.vertices
            hull_points = _points[hull_vertices, :]
            hull_pairs = _pairs[hull_vertices]
            hullsuccess += 1
        except ssq.QhullError:
            # print 'convex hull is not available for label:', _label
            hull_points = _points
            hull_pairs = _pairs
            hullexcept += 1

        # return hull_points
        hull_points = hull_points.tolist()
        hull_pairs = [ tuple(p) for p in hull_pairs] # converts back since np.arrays can't be dictionary keys
        CH_advantages_dic = Advantage.AdvantagesDict({k:v for (k,v) in zip(hull_pairs, hull_points)})
        return CH_advantages_dic

    def sum_cluster_and_matrix(self, policy_values_clusters, _matrix_nd):
        """
        this function receives a clusters policy-values dictionary and a nxd matrix (2nd arg). For each cluster if the policy
        has two actions in the same state, chooses one of them randomly.  Then sums all advantages in the same cluster,
        and adds the expected vectorial value of matrix_nd (the previous vectorial value per state)
        :param policy_values_clusters: a clusters-policy-values dictionary (dictionary of clusters as pairs (policy, value)
        :param _matrix_nd: a related matrix of dimension nxd
        :return: clusters-value dictionary, i.e. a dictionary with keys = ids of clusters, and values = pairs (l,v)
        where l is the pruned list of (state, action) and v is (old expected value + sum of advantages)
        """

        n = self.mdp.nstates
        d = self.mdp.d

        final_dic = {}
        dic_clusters_sum_v_old = {}

        for key, val in policy_values_clusters.iteritems():
            sum_d = np.zeros(d, dtype=ftype)
            pairs_list = []
            selected_pairs_dic = {}

            # builds a dictionary of list of indexes per state in the policy
            for ind, pair in enumerate(val[0]): # pair[0] is the state
                selected_pairs_dic.setdefault(pair[0],[]).append(ind)

            # rebuild a policy where double states are pruned and advantages are summed.
            for s in range(n):
                # selected_pairs = [val[0].index(pair) for pair in val[0] if pair[0] == i]
                if s in selected_pairs_dic:
                    pair_index = random.choice(selected_pairs_dic[s])
                    # pair_index = min(selected_pairs)
                    sum_d = map(operator.add, sum_d, val[1][pair_index])
                    pairs_list.append(val[0][pair_index])

            final_dic[key] = (pairs_list, sum_d)

        # adds the expected vectorial value of _matrix_nd to the advantage of the cluster
        e_val = self.get_initial_distribution().dot(_matrix_nd)
        for k, v in final_dic.iteritems():
            dic_clusters_sum_v_old[k] = (v[0], map(operator.add, e_val, v[1]))

        return dic_clusters_sum_v_old

    def make_clusters_pv_dict(self, _clusters_dict):
        """
        transforms a clusters advantages dictionary into a clusters policy-values dictionary
        :param _clusters_dict: a clusters advantages dictionary
        :return: a clusters policy-values dictionary
        """
        pv_dict = {}
        for key in _clusters_dict.iterkeys():
            policy = [k for k in _clusters_dict[key].iterkeys()]
            values = [v for v in _clusters_dict[key].itervalues()]
            # for k, v in _clusters_dict[key].iteritems():
            #     policy.append(k)
            #     values.append(v)
            pv_dict[key] = (policy, values)

        return pv_dict

    def accumulate_advantage_clusters(self, _old_value_vector, _advantages, _cluster_threshold):

        """

        this function cluster advantages, make a convex hull on each cluster, selects at most one action per state
         (randomly) and returns back a dictionary
        of (poliey, value) of each cluster

        :param _old_value_vector: a matrix of dimension nxd that will be added to improvements concluded from advantages
        :param _advantages: AdvantagesDict (all possible advantages)
        :param _cluster_threshold: max possible distance(cosine similarity distance) between two points in each cluster
        :return: returns back a dictionary of clusters including: key : value. Key is a numerical cluster id,
                value is a pair (policy,VV) where policy is a list of (state, action) pairs and VV is the sum on all
                related vectors + beta._matrix_nd
        """

        clustered_advantages = self.cluster_cosine_similarity(_advantages, _cluster_threshold)
        # clustered_advantages now contains a dictionary where keys are cluster ids and vals are AdvantagesDict
        # convex_hull_clusters = {}
        #
        # for key, val in clustered_advantages.iteritems():
        #     tempo = self.make_convex_hull(val, key) #  tempo is also an AdvantagesDic
        #     convex_hull_clusters[key] = tempo
        convex_hull_clusters = clustered_advantages

        if bool(clustered_advantages):
            # cluster_pairs_vectors = self.justify_cluster(convex_hull_clusters, clustered_advantages)
            # cluster_pairs_vectors is a dictionary with values = pairs (list, list)
            cluster_pairs_vectors = self.make_clusters_pv_dict(convex_hull_clusters)
            sum_on_convex_hull_temp = self.sum_cluster_and_matrix(cluster_pairs_vectors, _old_value_vector)
            sum_on_convex_hull = {key: val for key, val in sum_on_convex_hull_temp.iteritems() if val[1]}

            return sum_on_convex_hull

        return {}

    def declare_policies(self, _policies, pi_p):
        """
        this function receives dictionary of state action pairs an related vector value improvements
        and returns back dictionary of policies related to given pairs and the same vector value improvement
        :param _policies: dictionary with key = cluster id, value = (policy list, vectorial value)
        :param pi_p: a policy dict {state:action, ...}
        :return: dictionary of new policies and related improved vector values
        """

        _pi_p = pi_p.copy()
        V_append_d = np.zeros(self.mdp.d, dtype=ftype)

        new_policies = {}
        _pi_old = copy.deepcopy(_pi_p)

        for k, pv_pair in _policies.iteritems():
            for key, val in _pi_p.iteritems():
                tempo = [item[1] for item in pv_pair[0] if item[0] == key]
                if tempo:
                    _pi_p[key] = tempo
                    # else:
                    #    adv_d = self.get_initial_distribution()[key]*(self.mdp.get_vec_Q(key, _pi_old[key][0],  matrix_nd)-matrix_nd[key])
                    #    V_append_d = operator.add(V_append_d, adv_d)
                    #    print 'dakhele else V_append_d = ', V_append_d

            # V_append_d = np.zeros(self.mdp.d, dtype=ftype)

            #TODO V_append_d is alvais 0

            new_policies[k] = (_pi_p,np.float32(operator.add(pv_pair[1], V_append_d)) ) #np.float32(policy[1]))

            #global log
            #print >> log, 'added vector ', policy[1], ' added advantage', np.dot(self.get_Lambda(), policy[1])

            _pi_p = copy.deepcopy(_pi_old)

        return new_policies