import copy
import cplex
import numpy as np
import utils
import Problem
import scipy
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

import V_bar_search

from scipy.spatial.distance import cdist as linfDistance

ftype = np.float32

class propagation_V:

    def __init__(self, m, inequalities, cluster_v_bar_epsilon, epsilon_error):
        self.m = m
        self.d = self.m.d
        self.cluster_v_bar_epsilon = cluster_v_bar_epsilon
        self.epsilon_error = epsilon_error

        self.inequalities = inequalities

    def initialize_LP(self):
        """initialize linear programming as a minimization problem"""
        self.prob = cplex.Cplex()
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

        constr, rhs = [], []
        _d = self.d

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
        E: constraint is equal than rhs"""

        self.prob.linear_constraints.add(lin_expr=constr, senses="E" * len(constr), rhs=rhs)
        self.prob.write("show-Ldominance.lp")

    def produce_policy(self, v_n):
        """
        it takes an array form of policy and transforms it to dictionary form of policy
        :param v_n: an array of dimension |states|
        :return: a dictionary form of policy
        """
        return {i:[v_n[i]] for i in range(len(v_n))}

    """P_initial includes [0,..,0] vector here."""
    def make_convex_hull(self, P_initial, hull_vertices):
        """
        makes convex hull of given Points
        :param P_initial: matrix of many d-dimensional rows
        :return: pair of updated P_initial and list of vertices row index in P_initial
        """

        try:
            hull = ConvexHull(P_initial)
            hull_vertices = list(hull.vertices)
            P_initial = P_initial[hull_vertices, :]
        except scipy.spatial.qhull.QhullError:
            print 'convex hull is not available'
            print "P_initial", P_initial
            P_initial = P_initial
            hull_vertices = range(P_initial.shape[0])

        """checks if zero vector is inside the generated vertices of convex hull or not.
        If there is no zero in the vertices list, the zero vector will be added to P_initial and hull_vertices as well"""
        if not [0] * self.d in P_initial:
            P_initial = np.insert(P_initial, 0, np.zeros(shape=(1, self.d)), 0)
            hull_vertices.append(0)

        return (P_initial, hull_vertices)

    def in_hull(self, p, hull):
        """
        Test if point `p` is in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        q = np.array([p])

        if not isinstance(hull,Delaunay):
            try:
                hull = Delaunay(hull)
            except scipy.spatial.qhull.QhullError:
                return False

        return hull.find_simplex(q)>=0

    def epsilon_close_convex_hull(self, V_d, P_inintial, epsilon):
        """
        the function gets vector and check if there is a vector inside P_initial which is epsilon close to V_d
        :param V_d: d dimensional vector
        :param P_inintial: array of d dimensional rows
        :return: True or False
        """
        for item in xrange(P_inintial.shape[0]):
            dist = linfDistance([np.array(P_inintial[item, :])], [np.array(V_d)], 'chebyshev')[0,0]
            if dist < epsilon:
                return True

        return False

    def find_kdominated(self, v, P_inintial):

        for item in xrange(P_inintial.shape[0]):
            ob = [(j,float(P_inintial[item, :][j]-v[j])) for j in range(0, self.d)]

            self.prob.objective.set_linear(ob)
            self.prob.solve()

            result = self.prob.solution.get_objective_value()
            if result > 0.0:
                return True

        return False

    def check_epsilon(self, V_d, P_intial, epsilon):
        """
        this function gets a new d dimensional V_d and checks if V_d is inside Conv(P_initial) or
        is there any p in P_initial such that ||p-V_d|| <= epsilon and returns TRUE. if non of these situations satisfy,it
        returns FALSE.
        :param V_d: a d dimensional vector
        :param P_intial: array of several d-dimensional vectors
        :param epsilon: the epsilon error for checking
        :return: True or False
        """

        #TODO I may change this function: checks if a point inside a given convex hull?
        #if vector is inside old convex hull
        rep = self.in_hull(V_d, P_intial)
        #sometimes ishull returns True or False instead of [True] or [False] arrays
        if isinstance(rep, np.ndarray):
            rep = rep[0]

        if rep:
            return True
        #if vector is epsilon close to a vector in P_initial
        if self.epsilon_close_convex_hull(V_d, P_intial, epsilon):
            return True
        #if vector is kdominated by a vector in P_initial
        if self.find_kdominated(V_d, P_intial):
            return True

        return False

    def update_convex_hull_epsilon(self, frontier, problem):
        """
        this function gets set of current polytope vertices, generates new vectors inside \mathcal{V} polytope using
        clustering advantages and making a new convex hull of them.
        :param P_initial: matrix of d-dimensional rows, each row is a vertice of the given polytop.
        :param frontier: queue of type my_data_struc includes all nodes for extension
        :param hull_vertices: indices of P_initial vectors; we keep this index to not consider [0,...,0] vector in vector extensions.
        :param problem: the introduced problem as a mdp with two types of errors.
        :return: pairs of (P_initial, frontier) such that P_initial includes [0,..,0] vector too.
        """

        frontier_addition = utils.my_data_struc()

        """P_new saves vertices of the given convex hull"""
        P_new = np.zeros(shape=(1, self.d))
        for i in range(frontier.__len__()):
            P_new = np.vstack([P_new, frontier.A[i].state[1]])

        P_initial = copy.copy(P_new)

        #TODO may be using frontier is not anymore useful in our new method.
        for node in frontier.A:
            for child in node.expand(problem= problem):
                if not (self.check_epsilon(child.state[1], P_initial, self.epsilon_error)):
                    frontier_addition.append(child)
                    P_new = np.vstack([P_new, child.state[1]])
        """at the end of this loop, it added all new generated vectors of each vertice too"""

        length_hull_vertices = frontier.__len__()
        hull_vertices = range(length_hull_vertices)
        counter = 0
        for node in frontier_addition.A:
             frontier.append(node)
             hull_vertices.append(length_hull_vertices+counter)
             counter += 1

        temp_convex = self.make_convex_hull(P_new, hull_vertices)
        P_initial = temp_convex[0]
        hull_vertices = temp_convex[1]
        frontier.update([item-1 for item in hull_vertices if item-1 >= 0])

        return (P_initial, frontier, hull_vertices)

    def IsEqual(self, P1, P2):
        """
        this functions checks if two matrices P1 and P2 are equal or not. equality here means any row of P1 exists in P2
        and inverse.
        :param P1 the first given array
        :param P2 second given array
        :return True if two matrices have the same rows regardless of their permutations and False Otherwise
        """
        if P1.shape[0] != P2.shape[0]:
            return False

        for i in range(P2.shape[0]):
            if (not P2[i, :] in P1):
                return False

        for i in range(P1.shape[0]):
            if (not P1[i, :] in P2):
                return False

        return True

    def convex_hull_search(self, prob):
        """
        this function gets a problem as tree of nodes each node is pair of policy, V_bar and Uvec matrix as
        ({0:2, 1:0, 2:1, 3:2, 4:1}, [0.1,0.25], [[1.00, 0.30]
                                                 [0.50, 0.60]
                                                 [0.78, 0.43]
                                                 [1.40, 3.11]])
        and tries to propagate all v_bar using extending node in each iteration and take their vertices of the optimal convex hull.
        :param problem: tree of nodes each node is pair of policy and V_bar as ({0:2, 1:0, 2:1, 3:2, 4:1}, [0.1,0.25])
        :return: returns set of approximated non-dominated v_bar vectors: vectors of dimension d
        """

        """
        an array initialized by [0, ..,0] vector of dimension d, P_initial keeps
        the vertices of optimal convex hull after any iteration
        """
        P_initial= np.zeros(shape=(1, self.d))
        m = self.m
        """we use frontier to keep each required vertice of convex hull inside it. this structure contains only the vectors."""
        frontier = utils.my_data_struc()

        '''
        make initial v_bars using d vectors in which each vector of the form [0,..,0,1,0,..,0] and saves three required
        information : [Policy, v_bar, Uvec_n_d] as a Node structure. This list is assumed as a state for the graph.
        '''
        for i in range(self.d):
            m.set_Lambda(np.array([1 if j==i else 0 for j in xrange(self.d)]))
            Uvec_n_d = m.value_iteration(epsilon=0.00001)
            v_d = m.initial_states_distribution().dot(Uvec_n_d)
            n = Problem.Node([self.produce_policy(m.best_policy(Uvec_n_d)), v_d, Uvec_n_d])
            frontier.append(n)

        """add v-bar vectors related to [0,..,0,1,0,..,0] identical lambda vetors for initializing the optimal \mathcal{V}
        polytope"""
        for item in range(self.d):
                P_initial = np.vstack([P_initial, frontier.A[item].state[1]])

        #hull_vertices = range(P_initial.shape[0])
        #************************************************************

        """removes unused nodes from our graph. it means if related vector of Node is not considered in the convex hull
        update function will remove it from frontier"""
        temp = self.update_convex_hull_epsilon(frontier, prob)
        P_new = temp[0]
        frontier = temp[1]

        iteration = 0
        #while not(self.IsEqual(P_initial, P_new)):
        for i in range(250):
            if not(self.IsEqual(P_initial, P_new)):
                P_initial = P_new
                temp = self.update_convex_hull_epsilon(frontier, prob)
                P_new = temp[0]
                frontier = temp[1]
                iteration += 1
            else:
                return ([val for val in P_new[1:] if not all(v == 0.0 for v in val)], iteration)
            if i % 10 == 0:
                print i,"=", P_new.shape[0]

        #print 'iteration', iteration
        return ([val for val in P_new[1:] if not all(v == 0.0 for v in val)], iteration)

    def convex_hull_search_experimental(self, prob, random_lambdas, exact, aver_lambda):
        """
        this function gets a problem as tree of nodes each node is pair of policy, V_bar and Uvec matrix as
        ({0:2, 1:0, 2:1, 3:2, 4:1}, [0.1,0.25], [[1.00, 0.30]
                                                 [0.50, 0.60]
                                                 [0.78, 0.43]
                                                 [1.40, 3.11]])
        and tries to propagate all v_bar using extending node in each iteration and take their vertices of the optimal convex hull.
        :param problem: tree of nodes each node is pair of policy and V_bar as ({0:2, 1:0, 2:1, 3:2, 4:1}, [0.1,0.25])
        :return: returns set of approximated non-dominated v_bar vectors: vectors of dimension d
        """

        """
        an array initialized by [0, ..,0] vector of dimension d, P_initial keeps
        the vertices of optimal convex hull after any iteration
        """
        P_initial= np.zeros(shape=(1, self.d))
        m = self.m
        """we use frontier to keep each required vertice of convex hull inside it. this structure contains only the vectors."""
        frontier = utils.my_data_struc()

        '''
        make initial v_bars using d vectors in which each vector of the form [0,..,0,1,0,..,0] and saves three required
        information : [Policy, v_bar, Uvec_n_d] as a Node structure. This list is assumed as a state for the graph.
        '''
        for i in range(self.d):
            m.set_Lambda(np.array([1 if j==i else 0 for j in xrange(self.d)]))
            Uvec_n_d = m.value_iteration(epsilon=0.00001)
            v_d = m.initial_states_distribution().dot(Uvec_n_d)
            n = Problem.Node([self.produce_policy(m.best_policy(Uvec_n_d)), v_d, Uvec_n_d])
            frontier.append(n)

        """add v-bar vectors related to [0,..,0,1,0,..,0] identical lambda vetors for initializing the optimal \mathcal{V}
        polytope"""
        for item in range(self.d):
                P_initial = np.vstack([P_initial, frontier.A[item].state[1]])

        #hull_vertices = range(P_initial.shape[0])
        #************************************************************

        """removes unused nodes from our graph. it means if related vector of Node is not considered in the convex hull
        update function will remove it from frontier"""
        temp = self.update_convex_hull_epsilon(frontier, prob)
        P_new = temp[0]
        frontier = temp[1]

        res = open("check" + ".txt", "w")

        #lists for saving error vs |V| length
        errors = []
        queries = []
        vector_length = []

        iteration = 0
        #while not(self.IsEqual(P_initial, P_new)):
        for i in range(20):#(250):
            if not(self.IsEqual(P_initial, P_new)):
                P_initial = P_new
                temp = self.update_convex_hull_epsilon(frontier, prob)
                P_new = temp[0]
                frontier = temp[1]
                iteration += 1

                #to see error changes vs size of generated Vs
                vectors = [val for val in P_new[1:] if not all(v == 0.0 for v in val)]
                print >> res, "********** iteration", i, "******************"
                res.flush()

                queries_ave = []
                errors_ave= []
                res.flush()

                for j in range(aver_lambda):
                    index = i*aver_lambda+j

                    V = V_bar_search.V_bar_search(_mdp= self.m, _V_bar=vectors, lam_random = random_lambdas[index])
                    temp = V.v_optimal(_random_lambda_number = 1000)

                    v_opt = temp[0]
                    queries_ave.append(temp[1])
                    errors_ave.append(np.dot(random_lambdas[index], v_opt) - np.dot(random_lambdas[index], exact[index]))


                errors.append(np.abs(np.average(errors_ave)))
                queries.append(np.average(queries_ave))
                vector_length.append(len(vectors))

                print >> res, "errors_ave", errors
                print >> res, "vector length", vector_length
                print >> res, "asked queries", queries
                res.flush()


            else:
                print '*******final results **********', (vector_length, errors, queries, iteration)
                return (vector_length, errors, queries, iteration)
            if i % 10 == 0:
                print i,"=", P_new.shape[0]

        #print 'iteration', iteration
        print '*******final results **********', (vector_length, errors, queries, iteration)
        return (vector_length, errors, queries, iteration)