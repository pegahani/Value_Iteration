import copy
import itertools
import random
import cplex
import numpy as np
import operator
import sys
from random_polytope import random_point_polytop

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
        self.prob.write("show-Ldominance.lp")
        self.prob.solve()

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
        False. The label including 1 or -1 represent respectively V_d is suoerior to U_d or U_d is suoerior to V_d
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
            # for i in range(2 * self.mdp.d, len(inequality_list)):
            for i in range(len(inequality_list)):  # the 2 * self.mdp.d spared testing the initial square, which is now in bounds
                first = True
                for x, y in zip(inequality_list[i][1:], new_constraint)[1:]:
                    if x == 0 and y == 0:
                        continue
                    if first:
                        u, v = x , y
                        first = False
                    elif (u * y != x * v):
                        break
                else :
                    return True
        print "new_constraint", new_constraint
        return False

    def Query(self, V_d, U_d):

        """
        cmpare two d dimensional vectors and returns back the most prefered one regarding the selected lambda (in other
        word: user)
        :param V_d: d dimensional vector
        :param U_d: d dimensional vector
        :return: 1 if V_d > U_d otherwise -1
        """

        bound = [0.0]
        _d = len(V_d)

        constr = []
        rhs = []

        if self.lam_random.dot(V_d) > self.lam_random.dot(U_d):
            new_constraints = bound+map(operator.sub, V_d, U_d)
            #if not self.is_already_exist(self.Lambda_ineqalities, new_constraints):
            #TODO may be we can change vector type from float to float32. like that is_already_exist function will be more effective
            c = [(j, float(V_d[j] - U_d[j])) for j in range(0, _d)]
            constr.append(zip(*c))
            rhs.append(0.0)
            self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)
            self.Lambda_ineqalities.append(new_constraints)
            self.query_counter_ += 1
            return 1

        else:
            new_constraints = bound+map(operator.sub, U_d, V_d)

            #TODO check is_already_exist function: it seems that it has a problem
            #if not self.is_already_exist(self.Lambda_ineqalities, new_constraints):
            c = [(j, float(U_d[j] - V_d[j])) for j in range(0, _d)]
            constr.append(zip(*c))
            rhs.append(0.0)
            self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

            self.Lambda_ineqalities.append(new_constraints)
            self.query_counter_ += 1
            return -1

        #return None

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

        """count number of times that V_i is better than V_j for random_lambda_numbers times of selecting random lambda
        in Lambda polytope"""
        for i in ambig_list:
            for j in xrange(random_lambda_numbers):
                selected_lambda = random_point_polytop(self.Lambda_ineqalities)
                if(np.dot(selected_lambda, v_bar_list[i[0]]) - np.dot(selected_lambda, v_bar_list[i[1]]) >=  0 ):
                    probability_dic[i] += 1

        """compute probability of eah pair. it means for each pair (v-i,v-j) what is probability of  preferring v_i to
        v_j after choosing random_lambda_numbers inside Lambda polytope. after we calculate how each probability close
        to 0.5"""
        for key,val in probability_dic.items():
            probability_dic[key] = np.abs(val/random_lambda_numbers - 0.5)

        """select a cut like \lambda.(v_i - v_j) generated from (v_i, v_j) in ambig_list that devides Lambda polytope
        approximately to two parts """
        closer_pair = min(probability_dic, key=probability_dic.get)
        """this method choose the first pair, if there are some pairs with the same probability"""
        #TODO may be add a method to select randomly among pairs with the same probability

        return closer_pair

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
                    not_ambig_dic[pair] = tempo[0]
                    ambig_list.remove(pair)

        return (ambig_list, not_ambig_dic)

    def get_Best_vector(self, _not_ambig_dic):
        """
        it takes a dictionary of all not ambiguous pairs and find the most prefered one according to their labels
        :param _not_ambig_dic: dictionary of all pairs in V_bar_list_d with their labels
        :return: the best v_d^* in V_bar_list_d
        """
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

        #index of maximum value in indexes list
        max_index = indexes.index(max(indexes))
        return v_bar_list[max_index]

    "this function returns back set of optimal V_bars, if I receive more than one It should have a theoretical error"
    def v_optimal(self, _random_lambda_number):
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
                if tempo[1]:
                    not_ambig_dic[(i,j)] = (tempo[0])
                else:
                    ambig_list.append((i,j))

        iteration = 0

        #TODO check it later to be sure about the while loop execution occurency
        """until ambig_list is not empty"""
        while ambig_list:

            iteration+= 1
            closer = self.find_closer_pair(ambig_list, _random_lambda_number)

            #define a label among {-1,1} for closer pair (i,j). 1 if v_i>V-j otherwise -1
            label = self.Query(V_bar_list[closer[0]], V_bar_list[closer[1]])
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
        v_best = self.get_Best_vector(not_ambig_dic)
        return v_best