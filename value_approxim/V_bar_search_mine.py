import copy
import random
import numpy as np
import itertools
import operator
import scipy
from scipy.spatial import ConvexHull
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value

try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
    from scipy.spatial.distance import cdist as linfDistance
except:
    from sparse_mat import dok_matrix,csr_matrix,l1distance

ftype = np.float32

class V_bar_search:

    def generate_inequalities(self, _d):
        """
        gets d dimension and returns back a unit cube of d_dimension inequalities
        :param _d: space dimension
        :return: set of inequalities in which members are lists
        """
        inequalities = []

        for x in itertools.combinations( xrange(_d), 1 ) :
            inequalities.append([0] + [ 1 if i in x else 0 for i in xrange(_d) ])
            inequalities.append([1] + [ -1 if i in x else 0 for i in xrange(_d) ])

        return inequalities

    def __init__(self, _mdp, _V_bar, lam_random):
        self.Lambda_ineqalities = self.generate_inequalities(_mdp.d)
        self.V_bar_list_d = _V_bar
        self.lam_random = lam_random
        self.query_number = 0

    def make_convex_hull(self):
        """
        returns back a convex hull on given set of optimal V_bars
        :return: convex hull of type list of arrays
        """
        hull_points_d = []
        try:
            print "self.V_bar_list_d******************", self.V_bar_list_d
            hull = ConvexHull(self.V_bar_list_d)
            hull_vertices = hull.vertices

            for i in hull_vertices:
                hull_points_d.append(self.V_bar_list_d[i])

        except scipy.spatial.qhull.QhullError:
                hull_points_d = self.V_bar_list_d

        return hull_points_d

    def generate_pairs(self, _list_d):
        """
        it receives list of d_dimensional arrays(set of optimal V_bars) and returns back all pairs combinations of their
        members
        :param _list_d: list of horizontal arrays of dimension d
        :return: set of two dimensional tuples
        """

        length = len(_list_d)
        result_list = {}

        for i in range(length):
            for j in xrange(i+1,length):
                l = len(result_list)
                result_list[l] = ((i, _list_d[i]),(j, _list_d[j]))

        return result_list

    def pareto_comparison(self, a, b):

        a = np.array(a, dtype= ftype)
        b = np.array(b, dtype= ftype)

        assert len(a)==len(b), \
                "two vectors don't have the same size"

        return all(a>b)

    def find_not_comparable_pairs(self, _pairs, _isparetocheck):
        """
        receives
        :param _pairs:set of tuples(each tuple is a pair of two d-dimensional v_bars)
        ex. of a tuple: (np.array([ 5.04264307,  0.17840226], dtype=np.float32), np.array([ 6.90675974,  0.19439724], dtype=np.float32))
        :param _isparetocheck: if it is true, it checks pairs comparability regarding pareto comparison
                                otherwise pairs comparability regarding Kdominance comparison
        :return: dictionary of not comparable pairs
        """

        not_comparable_pair = {}

        if(_isparetocheck):
            for val in _pairs.itervalues():
                if not self.pareto_comparison(val[0][1], val[1][1]):
                    if not self.pareto_comparison(val[1][1], val[0][1]):
                        l = len(not_comparable_pair)
                        not_comparable_pair[l] = val
        else:
            for val in _pairs.itervalues():
                if not self.K_dominance_check(val[0][1], val[1][1]):
                    if not self.K_dominance_check(val[1][1], val[0][1]):
                        l = len(not_comparable_pair)
                        not_comparable_pair[l] = val

        return not_comparable_pair

    def K_dominance_check(self, _V_best_d, Q_d):
        """
        :param _V_best_d: a list of d-dimension
        :param Q_d: a list of d-dimension
        :return: True if _V_best_d is prefered to Q_d regarding self.Lambda_inequalities and using Kdominance
         other wise it returns False
        """
        _d = len(_V_best_d)

        prob = LpProblem("Ldominance", LpMinimize)
        lambda_variables = LpVariable.dicts("l", range(_d), 0)

        for inequ in self.Lambda_ineqalities:
            prob += lpSum([inequ[j + 1] * lambda_variables[j] for j in range(0, _d)]) + inequ[0] >= 0

        prob += lpSum([lambda_variables[i] * (_V_best_d[i]-Q_d[i]) for i in range(_d)])

        #prob.writeLP("show-Ldominance.lp")

        status = prob.solve()
        LpStatus[status]
        result = value(prob.objective)

        if result < 0:
            return False

        return True

    def K_dominnace_check_2(self, u_d, v_d, _inequalities):
        """
        :param u_d: a d-dimensional vector(list) like [ 8.53149891  3.36436796]
        :param v_d: tha same list like u_d
        :param _inequalities: list of constraints on d-dimensional Lambda Polytope like
         [[0, 1, 0], [1, -1, 0], [0, 0, 1], [1, 0, -1], [0.0, 1.4770889, -3.1250839]]
        :return: True if u is Kdominance to v regarding given _inequalities otherwise False
        """
        _d = len(u_d)

        prob = LpProblem("Kdominance", LpMinimize)
        lambda_variables = LpVariable.dicts("l", range(_d), 0)

        for inequ in _inequalities:
            prob += lpSum([inequ[j + 1] * lambda_variables[j] for j in range(0, _d)]) + inequ[0] >= 0

        prob += lpSum([lambda_variables[i] * (u_d[i]-v_d[i]) for i in range(_d)])

        #prob.writeLP("show-Ldominance.lp")

        status = prob.solve()
        LpStatus[status]

        result = value(prob.objective)
        if result < 0:
            return False

        return True

    def get_not_comparable_pairs_indexes(self, _not_comparable_pairs):
        """
        :param _not_comparable_pairs: dictionary of not comparable pairs for example:
        {0: ((0, array([ 2.42584229,  9.07489014], dtype=float32)),(1, array([ 8.82884312,  0.32941794], dtype=float32))),
         1: ((0, array([ 2.42584229,  9.07489014], dtype=float32)), (2, array([ 8.53149891,  3.36436796], dtype=float32)))}
         this includes two pairs of vectors. each pairs is including two pairs also in which he first element is vector index from
         given possible list of non-dominated V_bar_list_d. For instance (0, array([ 2.42584229,  9.07489014], dtype=float32) means:
         array([ 2.42584229,  9.07489014]) vector is the first vector in given V_bar_list_d
        :return: it returns back LIST of indexes of all vectors which are not comparable at leas with another vector in V_bar_list_d
        """
        list_indexes = []

        for i in range(len(_not_comparable_pairs)):
            list_indexes.append(_not_comparable_pairs[i][0][0])
            list_indexes.append(_not_comparable_pairs[i][1][0])

        list_indexes_not_comparable = list(set(list_indexes))

        return list_indexes_not_comparable

    def dominated_to_how_many_rest(self,u, v, _inequalities, rest_list, _vertices_points):
        """
        :param u:a pair in which the first element is vector position in V_bar_list_d and the second is the vector
        (6, array([ 4.53753138,  9.45397568], dtype=float32))
        :param v: is a pair like u for instance :
        (0, array([ 2.42584229,  9.07489014], dtype=float32))
        :param _inequalities: set of constrains for Lambda polytope
        :param rest_list: list of rest of indexes which are inside any comparable pairs: for instance if {0,1,2,3,4,5,6}
        are indexes of vectors which are not comparable with any vector in V_bar_list_d we will have: rest_list = [1,2,3,4,5]
        :param _vertices_points: is equal V_bar_list_d
        :return: it returns back a pair like (6, [6, 5, 1, 2, 3, 4]) which says vector u[1] is Kdominated to 6 vectors
        with indexes 6, 5, 1, 2, 3, 4 in V_bar_list_d regarding _inequalities as set of constraints on Lambda polytope
        """

        lists_indexes = [u[0], v[0]]
        _ineq = copy.copy(_inequalities)
        count = 1
        _ineq.append([0.0]+map(operator.sub, u[1], v[1]))

        for j in rest_list:
            check = self.K_dominnace_check_2(u[1], _vertices_points[j], _ineq)
            if check:
                lists_indexes.append(j)
                count+=1

        return (count,lists_indexes)

    def askUser(self, u, v):
        """
        :param u: first vector as a given list of d dimension
        :param v: second vector as a list type of d dimension
        :return: list which will be u-v or v-u depends on user response
        """

        self.query_number+=1

        lambda_dot_u = np.dot(np.array(self.lam_random), np.array(u))
        lambda_dot_v = np.dot(np.array(self.lam_random), np.array(v))

        if lambda_dot_u> lambda_dot_v:
            return map(operator.sub, u, v)
        else:
            return map(operator.sub, v, u)

        return


    def Find_pair(self, _not_comparable_pairs, dominated_to_how_many_points):

        """
        :param _not_comparable_pairs: is a dictionary of not comparable pairs in V_bar_list_d for instance:
        {0: ((0, array([ 2.42584229,  9.07489014], dtype=float32)), (1, array([ 8.82884312,  0.32941794], dtype=float32))),
         1: ((0, array([ 2.42584229,  9.07489014], dtype=float32)), (2, array([ 8.53149891,  3.36436796], dtype=float32)))
         in which the first element of any pair is the vector index from V_bar_list_d list.
        :param dominated_to_how_many_points: a dictionary that shows for each pair in _not_comparable_pairs, the first vector
        and second vector are dominated to how many rest of vectors indexes in V_bar_list_d. For instance:
        for the first pair in _not_comparable_pairs we will have:
        dominated_to_how_many_points = {0: (1, [0, 1]), 1: (1, [1, 0])} which means [ 2.42584229,  9.07489014] is dominated
        to 1 rest of vectors in _not_comparable_pairs that is a vector of index 1
        :return: the output can be like  ([0.29734421, -3.03495], [2, 0, 3, 4, 5, 6]) in which the first element is
        difference between two vectors of pairs like (u,v) from _not_comparable_pairs in which if u is kdominance to v,
        then u will be dominated to [2, 0, 3, 4, 5, 6] vector indexes from V_bar_list_d.
        """

        numbers_of_dominated = [t[0] for t in dominated_to_how_many_points.itervalues()]
        #max_index =  numbers_of_dominated.index(max(numbers_of_dominated))
        max_value = max(numbers_of_dominated)
        max_index  = random.choice([i for i, j in enumerate(numbers_of_dominated) if j == max_value])

        pair = _not_comparable_pairs[max_index/2]
        is_second_better = max_index%2

        result = self.askUser(pair[0][1], pair[1][1] )
        return (result, dominated_to_how_many_points[max_index][1])

    def which_query(self, _not_comparable_pairs, list_indexes_not_comparable, _vertices_points, _lambda_inequalities):
        """
        :param _not_comparable_pairs: dictionary of not comparable pairs rearding pareto or kdominance comparison.
        pairs have the same structure like pais in function find_not_comparable_pairs(). i.e each pair is also a pair
        like (1, [0.112, 0.33]) in which 1 is vector index in list self.V_bar_list_d
        :param list_indexes_not_comparable: list of indexes of not comparable vectors in self.V_bar_list_d as an example:
        [0,1,5,7] is indexes of vector in self.V_bar_list_d which are not comparable at least with another vector in self.V_bar_list_d
        :param _vertices_points: list of vectors which is technicaly equal self.V_bar_list_d
        :param _lambda_inequalities: list of Lambda polytope constraints
        :return:output is a pair of two elements. the first element is the difference between two selected vectors to be
        introduced as a cut on Lambda polytope. For instance in ([0.027866364, -0.015419066], [2, 0, 3, 4, 5, 6, 7, 8])
        the cut will be 0.027866364.lambda_1-0.015419066.lambda2 = 0. and the second element is a list of
        """

        dominated_to_how_many_points = {}
        _inequalities = copy.copy(_lambda_inequalities)

        #list_indexes_not_comparable = self.get_not_comparable_pairs_indexes(_not_comparable_pairs)
        for i in range(len(_not_comparable_pairs)):

            u = _not_comparable_pairs[i][0]
            v = _not_comparable_pairs[i][1]
            new_list = [u[0], v[0]]

            "this compare each pair with not comparable rest of pairs"
            rest_list = [item for item in list_indexes_not_comparable if item not in new_list]
            #rest_list = [item for item in range(len(self.V_bar_list_d)) if item not in new_list]

            #"this compare each pair with all convex hull vertices"
            "first element in list is preferred"
            dominated_to_how_many_points[2*i] = self.dominated_to_how_many_rest(u, v, _inequalities,
                                                                                rest_list, _vertices_points)

            "second element in list is preferred"
            dominated_to_how_many_points[2*i+1] = self.dominated_to_how_many_rest(v, u, _inequalities,
                                                                                rest_list, _vertices_points)

        "choose the query which removes more points"
        which_pair = self.Find_pair(_not_comparable_pairs, dominated_to_how_many_points)

        return which_pair

    def update_not_comparable_pairs(self, _not_comparable_pairs, compared_pair):

        """
        :param _not_comparable_pairs: dictionary of pairs of not comparable pairs from self.V_bar_list_d for instance:
        {0: ((1, array([ 8.82884312,  0.32941794], dtype=float32)), (6, array([ 4.53753138,  9.45397568], dtype=float32)))}
        :param compared_pair: a list of indexes of comparable vectors like [1,6]. I means after adding a new cut
        to self.Lambda_inequalities, these vectors (indexes of vectors) become comparable.
        this function removes all compared pairs from _not_comparable_pairs and returns back dictionary of rest of not comparable pairs
        for instance for the given example [6,1], (6,1) are indexes o two vectors that are comparable now. then we
        should remove the only pair from compared_pair, and the final answer will be an empty dictionary {}
        :return: a dictionary of res of not comparable pairs. this response is subset of _not_comparable_pairs
        """

        indexes_to_be_deleted_from_dict = []
        dict_indexes_as_list = [[item[0][0],item[1][0]] for item in _not_comparable_pairs.itervalues()]

        for i in range(1,len(compared_pair)):

            if [compared_pair[0], compared_pair[i]] in dict_indexes_as_list:
                indexes_to_be_deleted_from_dict.append(dict_indexes_as_list.index([compared_pair[0], compared_pair[i]]))

            elif [compared_pair[i], compared_pair[0]] in dict_indexes_as_list:
                indexes_to_be_deleted_from_dict.append(dict_indexes_as_list.index([compared_pair[i], compared_pair[0]]))
            else:
                print "pair is not inside the _not_comparable_pairs set"

        new_dic = {}
        counter = 0
        for key, val in _not_comparable_pairs.iteritems():
            if key not in indexes_to_be_deleted_from_dict:
                new_dic[counter]= val
                counter+=1

        return new_dic

    def Prepare_Lambda(self):
        """
        proposes enough cuts on the given unite cube in a way that all (v_i,v_j) vectors can be compared
        :return: it returns list of new constraints on Lambda polytope.
        """

        prep = open("prepare" + ".txt", "w")

        vertices_points = self.make_convex_hull()
        self.V_bar_list_d = copy.copy(vertices_points)
        pairs = self.generate_pairs(vertices_points)
        not_comparable_pair_pareto = self.find_not_comparable_pairs(pairs, _isparetocheck= True)
        not_comparable_pair_kdominance = self.find_not_comparable_pairs(not_comparable_pair_pareto, _isparetocheck= False)

        list_indexes_not_comparable = self.get_not_comparable_pairs_indexes(not_comparable_pair_kdominance)

        print >> prep, "vertices_points", vertices_points
        prep.flush()

        i = 0
        while(list_indexes_not_comparable):
            print >> prep, "i", i
            i+=1
            prep.flush()

            print >> prep, "list_indexes_not_comparable", list_indexes_not_comparable
            prep.flush()

            print >> prep, 'self.Lambda_ineqalities', self.Lambda_ineqalities
            prep.flush()

            print >> prep, "not_comparable_pair_kdominance", not_comparable_pair_kdominance
            prep.flush()

            query = self.which_query(not_comparable_pair_kdominance, list_indexes_not_comparable, vertices_points,self.Lambda_ineqalities)# lambda_inequal)
            self.Lambda_ineqalities.append([0.0]+query[0])

            not_comparable_pair_rest = self.update_not_comparable_pairs(not_comparable_pair_kdominance, query[1])
            not_comparable_pair_kdominance = self.find_not_comparable_pairs(not_comparable_pair_rest, _isparetocheck= False)

            list_indexes_not_comparable = self.get_not_comparable_pairs_indexes(not_comparable_pair_kdominance)

        print >> prep, "list_indexes_not_comparable", list_indexes_not_comparable
        prep.flush()

        print >> prep, "not_comparable_pair_kdominance", not_comparable_pair_kdominance
        prep.flush()

        return self.Lambda_ineqalities

    "this function returns back set of optimal V_bars, if I receive more than one It should have a theoretical error"
    def v_optimal(self):
        """
        this function returns the optimal V_bar for given set of optimal V_ds: self.V_bar
        :return: optimal V_bar of dimension d
        """
        opt = open("observe-optimal" + ".txt", "w")

        self.Prepare_Lambda()

        print >> opt, "self.V_bar_list_d", self.V_bar_list_d
        opt.flush()

        V_bar_length = len(self.V_bar_list_d)
        is_dominated_to_rest = []

        for i in range(V_bar_length):
            list_tempo = [item for item in range(V_bar_length) if item not in [i]]

            check = True
            dominated_counter = 0
            for j in list_tempo:
                #if check:
                check = self.K_dominance_check(self.V_bar_list_d[i], self.V_bar_list_d[j])
                if check:
                    dominated_counter+=1
                #else:
                    #break
            #is_dominated_to_rest.append(check)
            is_dominated_to_rest.append(dominated_counter)

        print >> opt, 'is_dominated_to_rest', is_dominated_to_rest
        opt.flush()

        #which_V_bars = [i for i,x in enumerate(is_dominated_to_rest) if x == True]
        which_V_bars = [is_dominated_to_rest.index(max(is_dominated_to_rest))]

        print >> opt, "which_V_bars", which_V_bars
        opt.flush()

        print >> opt, "self.Lambda_ineqalities",self.Lambda_ineqalities
        opt.flush()

        v_optimal = [self.V_bar_list_d[i] for i in which_V_bars]

        return (random.choice(v_optimal), self.query_number)
