import copy
import itertools
import random
import cplex
import numpy as np
import operator
import sys
from random_polytope import random_point_polytop

ftype = np.float32
np.random.seed(314159265)

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
        self.query_counter_ = 0

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

    #******************functions related to comparisons***********
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

        #TODO Emiliano
        #self.prob.parameters.simplex.limits.lowerobj = -0.001
        #TODO Emiliano

        self.prob.write("show-Ldominance.lp")
        self.prob.solve()

        #TODO Emiliano
        #self.prob.parameters.simplex.limits.lowerobj = -1e+75
        #TODO Emiliano

        result = self.prob.solution.get_objective_value()
        if result < 0.0:
            return False

        return True

    def IsComparable(self, V_d, U_d, pareto_check):
        """
        it takes two vectors as inputs
        :param V_d: d dimensional vector
        :param U_d: d dimensional vector
        :return: returns back True if two vectors are comparable using pareto or Kdominance methods. Otherwise it returns
        False. The label including 1 or -1 represent respectively V_d is superior to U_d or U_d is superior to V_d
        """

        #if pareto check not been already check, test it here
        if pareto_check:
            if self.pareto_comparison(V_d, U_d):
                return (1, True)
            if self.pareto_comparison(U_d, V_d):
                return (-1, True)

        if self.cplex_K_dominance_check(V_d, U_d):
            return (1, True)
        if self.cplex_K_dominance_check(U_d, V_d):
            return (-1, True)

        return (0, False)

    #TODO manage all is_alredy_exist functions in one separate class
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
        return False

    def Query(self, _V_best, Q, noise):

        bound = [0.0]
        _d = len(_V_best)

        constr = []
        rhs = []

        Vscal = self.lam_random.dot(_V_best)
        Qscal = self.lam_random.dot(Q)
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

        #TODO check is_already_exist function: it seems that it has a problem
        # memorizing it
        if not self.is_already_exist(self.Lambda_ineqalities, new_constraints):
            if keep:
                c = [(j, float(_V_best[j] - Q[j])) for j in range(0, _d)]
                #print >> self.wen,  "Constrainte", self.query_counter_, _V_best - Q, "|> 0"
            else:
                #TODO may be we can change vector type from float to float32. like that is_already_exist function will be more effective
                c = [(j, float(Q[j] - _V_best[j])) for j in range(0, _d)]
                #print >> self.wen, "Constrainte", self.query_counter_, Q - _V_best, "|> 0"
            self.query_counter_ += 1
            constr.append(zip(*c))
            rhs.append(0.0)
            self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
            self.Lambda_ineqalities.append(new_constraints)

        # return the result
        if keep:
            return 1
        else:
            return -1

    #******************functions related to comparisons***********
    def find_closer_pair(self, ambig_list, random_lambda_numbers):

        """
        this function finds a pair like (v_i, v_j) such that the generated cut from this pair devides the Lambda polytope
        to approximately two equal parts
        :param ambig_list: list of ambiguous pairs indexes
        :param random_lambda_numbers: number of selected random lambda vector from Lambda polytope
        :return: the best cut
        """

        #list of propogated V_bars
        v_bar_list = self.V_bar_list_d

        """this list includes a number between 0 and 1 for each pair. It defines how does each cut
        devides Lambda polytope to two parts"""
        probability_dic = {}

        for item in ambig_list:#initialize values at 0.0
            probability_dic[item] = 0.0

        #list of selected random lambda in Lambda
        selected_lambda_list = [] #list of selected random lambda
        for j in xrange(random_lambda_numbers):
            selected_lambda_list.append(random_point_polytop(self.Lambda_ineqalities))

        """count number of times that V_i is better than V_j for random_lambda_numbers times of selecting random lambda
        in Lambda polytope"""
        for i in ambig_list:
            for j in xrange(random_lambda_numbers):
                #choose from list of selected random lambdas
                selected_lambda = selected_lambda_list[j] #random_point_polytop(self.Lambda_ineqalities)
                if(np.dot(selected_lambda, v_bar_list[i[0]]) - np.dot(selected_lambda, v_bar_list[i[1]]) >=  0 ):
                    probability_dic[i] += 1

        """compute probability of eah pair. it means for each pair (v-i,v-j) what is probability of  preferring v_i to
        v_j after choosing random_lambda_numbers inside Lambda polytope. after we calculate how each probability close
        to 0.5"""
        for key,val in probability_dic.items():
            probability_dic[key] = np.abs(val/random_lambda_numbers - 0.5)
            #if val == 0:
            #    self.markdefined(key, ambig_list)

        """select a cut like \lambda.(v_i - v_j) generated from (v_i, v_j) in ambig_list that devides Lambda polytope
        approximately to two parts """
        closer_pair = min(probability_dic, key=probability_dic.get)
        """this method choose the first pair, if there are some pairs with the same probability"""
        #TODO may be add a method to select randomly among pairs with the same probability

        return (closer_pair, ambig_list)

    def clean_ambiguity(self, _ambig_list, _not_ambig_dic):

        """
        get two lists of new ambigous and not_ambigous members with set of new lambda inequalities and modifies these
         two lists regarding lambda inequalities as a new consequence of new added cut
        :param _ambig_list: list of ambiguous pairs
        :param _not_ambig_dic: list of not ambiguous pairs with their labels
        :return: (ambig_list, not_ambig_dic) modified values for two instances
        """

        V_bar_list = self.V_bar_list_d

        ambig_list = copy.copy(_ambig_list)
        not_ambig_dic = copy.copy(_not_ambig_dic)

        #TODO we can remove some pairs based on the associativity of comparison. It means before checkig Kdominance comparison we can remove some other pairs regarding the associativity
        """For instance  not_ambiguous = {(0, 1): -1, (1, 4): 1, (1,2):1, (2, 4): 1, (2, 3): 1, (3, 4): 1, (0, 2): -1}
        ambig list=[(0, 3), (1, 3), (0, 4)]. If closer pair is (1,2) we can say (1,3) is not ambiguous because (1,2) and
        (2,4) are not ambiguous. Refere to Jaimison paper : `Active Ranking using Pairwise Comparisons`
        """
        for pair in _ambig_list:
            tempo = self.IsComparable(V_bar_list[pair[0]], V_bar_list[pair[1]], pareto_check = False)
            if tempo[1]:
                print 'this pair is not amboguous too', pair
                not_ambig_dic[pair] = tempo[0]
                ambig_list = self.markdefined(pair,tempo[0], ambig_list)

        return (ambig_list, not_ambig_dic)

    def get_Best_vector_1(self, _not_ambig_dic):
        """
        it takes a dictionary of all not ambiguous pairs and find the most prefered one according to their labels
        :param _not_ambig_dic: dictionary of all pairs in V_bar_list_d with their labels
        :return: the best v_d^* in V_bar_list_d
        """

        print '_not_ambig_dic', _not_ambig_dic
        v_bar_list = self.V_bar_list_d

        #score of each v_bar inside the given v_bar_list which is 0 at the begining for each V_bar
        #we use thses scores to find our proposed ranking on v_bar_list members
        indexes = [0]*len(v_bar_list)

        for key, val in _not_ambig_dic.items():
            if val == 1:
                indexes[key[0]] +=1
            elif val == -1:
                indexes[key[1]] +=1
            else:
                sys.exit("pair labels are 1 or -1 not something else!!")

        print "indexes", indexes
        print 'v_bar_list', v_bar_list
        #index of maximum value in indexes list
        max_index = indexes.index(max(indexes))

        return v_bar_list[max_index]

    def get_Best_vector(self, _not_ambig_dic):
        """
        it takes a dictionary of all not ambiguous pairs and find the most prefered one according to their labels
        :param _not_ambig_dic: dictionary of all pairs in V_bar_list_d with their labels
        :return: the best v_d^* in V_bar_list_d
        """
        v_bar_list = self.V_bar_list_d
        vector_indexes = range(len(v_bar_list))

        for key, val in _not_ambig_dic.items():
            if val == -1:
                if key[0] in vector_indexes:
                    vector_indexes.remove(key[0])
            elif val == 1:
                if key[1] in vector_indexes:
                    vector_indexes.remove(key[1])

        assert len(vector_indexes) == 1, \
                 "there shoul be a problem in removing dominated elements"

        return v_bar_list[vector_indexes[0]]

    def union(self, a_list, b_list):
        """
        union two lists of pairs
        :param a: first list of pairs
        :param b: second list of pairs
        :return: list of their union
        """
        for e in b_list:
            if e not in a_list:
                a_list.append(e)

        return a_list

    def markdefined(self, _pair, labe,  _ambig_list):
        """
        :param _pair: a pair of vector like (v_i, v_j)
        :param _ambig_list: list of ambiguous pairs
        :return: returns new list of ambiguous list after removing all useless pairs
        """
        _ambig_list_tempo = copy.copy(_ambig_list)
        if labe == 1:
            j = _pair[1]
        else:
            j = _pair[0]

        #counts all pairs either with v_j as a first element of pair or second element of pair
        for element in _ambig_list:
            if (element[0] == j or element[1] == j):
                _ambig_list_tempo.remove(element)

        return _ambig_list_tempo

    """Francois Algorithm"""
    def v_optimal(self, _random_lambda_number):

        V_bar_list = self.V_bar_list_d
        V_bar_len = len(self.V_bar_list_d)

        ambig_list = []
        not_ambig_dic = {}

        """make list of all pairs"""
        for j in range(0, V_bar_len):
            for i in range(0, j):
                ambig_list.append((i,j))

        """define all comparable pairs in the first glance"""
        for j in range(0, V_bar_len):
            for i in range(0, j):
                tempo = self.IsComparable(V_bar_list[i], V_bar_list[j], pareto_check= True)
                #if a pair is comparable
                if tempo[1]:
                    not_ambig_dic[(i,j)] = (tempo[0])
                    ambig_list = self.markdefined((i,j), tempo[0] , ambig_list)

        print 'ambig_list before', ambig_list

        iteration = 0
        # """until ambig_list is not empty"""
        while ambig_list:
            iteration += 1

            temp = self.find_closer_pair(ambig_list, _random_lambda_number)
            closer = temp[0]
            print 'closer', closer
            ambig_list = temp[1]

            label = self.Query(V_bar_list[closer[0]], V_bar_list[closer[1]], None)
            print 'label for closer', label
            not_ambig_dic[(closer)] = label
            ambig_list = self.markdefined(closer, label, ambig_list)
            print 'ambig_list after new closer', ambig_list

            ambig_notAmbig_pair = self.clean_ambiguity(ambig_list, not_ambig_dic)
            ambig_list = ambig_notAmbig_pair[0]
            not_ambig_dic = ambig_notAmbig_pair[1]

            print 'ambig_list final', ambig_list
            print 'not ambig final', not_ambig_dic

        assert not ambig_list ,\
                        "ambig_list should be null. then there is a problem in the code"

        #it returns the V_d at the top of our ranking!!
        v_best = self.get_Best_vector(not_ambig_dic)
        print "v_best", v_best
        return (v_best, self.query_counter_)

    "this function returns back set of optimal V_bars, if I receive more than one It should have a theoretical error"
    def v_optimal_1(self, _random_lambda_number):
        """
        this function returns the optimal V_bar for given set of optimal V_ds: self.V_bar
        :return: optimal V_bar of dimension d
        """
        V_bar_list = self.V_bar_list_d
        V_bar_len = len(self.V_bar_list_d)

        ambig_list = []
        not_ambig_dic = {}

        """define all comparable and not comparable pairs"""
        for j in range(0, V_bar_len):
            for i in range(0, j):
                tempo = self.IsComparable(V_bar_list[i], V_bar_list[j], pareto_check= True)
                #if a pair is comparable
                if tempo[1]:
                    not_ambig_dic[(i,j)] = (tempo[0])
                #if a pair is not comparable
                else:
                    ambig_list.append((i,j))

        iteration = 0

        print 'ambig_list', ambig_list
        print 'not_ambig_dic', not_ambig_dic

        #TODO check it later to be sure about the while loop execution occurency
        """until ambig_list is not empty"""
        while ambig_list:

            iteration+= 1
            closer = self.find_closer_pair(ambig_list, _random_lambda_number)

            print 'closer', closer

            #define a label among {-1,1} for closer pair (i,j). 1 if v_i>v_j otherwise -1
            label = self.Query(V_bar_list[closer[0]], V_bar_list[closer[1]], None)
            #add closer pair to not_ambig_dic dictionary
            not_ambig_dic[(closer)] = label

            #remove closer pair from ambig_list, because it is clear now
            ambig_list.remove(closer)

            """ambig_list and not_ambig_dic have been updated until now, we just need clean ambig_list. it means if other
            pairs inside not_ambig_dic are comparable thanks to `closer`pair, we will remove them from not_ambig_dic too.
            """
            ambig_notAmbig_pair = self.clean_ambiguity(ambig_list, not_ambig_dic)
            ambig_list = ambig_notAmbig_pair[0]
            not_ambig_dic = ambig_notAmbig_pair[1]

        #print 'iteration', iteration

        assert not ambig_list ,\
                        "ambig_list should be null. then there is a problem in the code"

        #it returns the V_d at the top of our ranking!!
        print 'not_ambig_dic', not_ambig_dic
        v_best = self.get_Best_vector(not_ambig_dic)
        return (v_best, self.query_counter_)
