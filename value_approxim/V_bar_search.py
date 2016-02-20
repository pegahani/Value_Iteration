import itertools
import random
import cplex
import numpy as np

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

        """initialize linear programming as a minimization problem"""
        self.prob = cplex.Cplex()
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

        constr, rhs = [], []
        _d = _mdp.d

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


    def pareto_comparison(self, a, b):
        a = np.array(a, dtype=ftype)
        b = np.array(b, dtype=ftype)

        assert len(a) == len(b), \
            "two vectors don't have the same size"

        return all(a > b)

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

    def IsComparable(self, V_d, U_d):
        """
        it takes two vectors as inputs
        :param V_d: d dimensional vector
        :param U_d: d dimensional vector
        :return: returns back True if two vectors are comparable using pareto or Kdominance methods. Otherwise it returns
        False. The label including 1 or -1 represent respectively V_d is suoerior to U_d or U_d is suoerior to V_d
        """

        if self.pareto_comparison(V_d, U_d):
            return (1, True)
        if self.pareto_comparison(U_d, V_d):
            return (-1, True)

        if self.cplex_K_dominance_check(V_d, U_d):
            return (1, True)
        if self.cplex_K_dominance_check(U_d, V_d):
            return (-1, True)

        return (0, False)

    "this function returns back set of optimal V_bars, if I receive more than one It should have a theoretical error"
    def v_optimal(self):
        """
        this function returns the optimal V_bar for given set of optimal V_ds: self.V_bar
        :return: optimal V_bar of dimension d
        """
        V_bar_list = self.V_bar_list_d
        V_bar_len = len(self.V_bar_list_d)

        ambig_list = []
        not_ambig_list = {}

        print "V_bar_list", V_bar_list

        """define all comparable and not comparable pairs"""
        for j in range(0, V_bar_len):
            for i in range(0, j):
                tempo = self.IsComparable(V_bar_list[i], V_bar_list[j])
                if tempo[1]:
                    not_ambig_list[(i,j)] = (tempo[0])
                else:
                    ambig_list.append((i,j))

        print "ambig_list", ambig_list
        print "not_ambig_list", not_ambig_list

        """until ambig_list is not empty"""
        #while ambig_list:
        """this list includes a number between 0 and 1 for each pair. It defines how does each cut
        devides Lambda polytope to two parts"""
        probability_dic = {}

        for item in ambig_list:#initialize values at 0.0
            probability_dic[item] = 0.0

        print 'probability_dic',probability_dic


    "this function returns back set of optimal V_bars, if I receive more than one It should have a theoretical error"
    def v_optimal_1(self):
        """
        this function returns the optimal V_bar for given set of optimal V_ds: self.V_bar
        :return: optimal V_bar of dimension d
        """

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

        #which_V_bars = [i for i,x in enumerate(is_dominated_to_rest) if x == True]
        which_V_bars = [is_dominated_to_rest.index(max(is_dominated_to_rest))]

        v_optimal = [self.V_bar_list_d[i] for i in which_V_bars]

        return random.choice(v_optimal)