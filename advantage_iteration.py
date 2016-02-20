import random
import cplex
import numpy as np
import itertools
import operator
import advantage


try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
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

    def reset(self, _mdp, _lambda, _lambda_inequalities):
        self.mdp = _mdp
        self.Lambda = np.zeros(len(_lambda), dtype=ftype)
        self.Lambda[:] = _lambda

        self.Lambda_inequalities = _lambda_inequalities

        self.query_counter_ = 0
        self.query_counter_with_advantages = 0

    def setStateAction(self):
        self.n = self.mdp.nstates
        self.na = self.mdp.nactions

    def get_Lambda(self):
        return self.mdp.get_lambda()

    # *********************** comparison part **************************

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

    def is_already_exist(self, inequality_list, new_constraint):
        """

        :param inequality_list: list of inequalities. list of lists of dimension d+1
        :param new_constraint: new added constraint to list of inequalities of dimension d+1
        :return: True if new added constraint is already exist in list of constraints for Lambda polytope
        """
        if new_constraint in inequality_list:
            return True
        else:
            for i in range(2 * self.mdp.d, len(inequality_list)):

                devision_list = [np.float32(x / y) for x, y in zip(inequality_list[i], new_constraint)[1:]]
                if all(x == devision_list[0] for x in devision_list):
                    return True

        return False

    def generate_noise(self, _d, _noise_deviation):
        vector_noise = np.zeros(_d, dtype=ftype)
        for i in range(_d):
            vector_noise[i] = np.random.normal(0.0, _noise_deviation)

        return vector_noise

    def Query_policies(self, _V_best, Q, noise):

        bound = [0.0]
        _d = len(_V_best[1])

        constr = []
        rhs = []

        if not noise:
            if self.Lambda.dot(_V_best[1]) > self.Lambda.dot(Q[1]):
                new_constraints = bound + map(operator.sub, _V_best[1], Q[1])
                if not self.is_already_exist(self.Lambda_inequalities, new_constraints):
                    c = [(j, float(_V_best[1][j] - Q[1][j])) for j in range(0, _d)]
                    constr.append(zip(*c))
                    rhs.append(0.0)
                    self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

                    self.Lambda_inequalities.append(new_constraints)

                return _V_best

            else:
                #TODO we have change Q with _V_best
                new_constraints = bound + map(operator.sub, Q[1], _V_best[1])
                if not self.is_already_exist(self.Lambda_inequalities, new_constraints):
                    c = [(j, float(Q[1][j] - _V_best[1][j])) for j in range(0, _d)]
                    constr.append(zip(*c))
                    rhs.append(0.0)
                    self.prob.linear_constraints.add(lin_expr=constr, senses="G" * len(constr), rhs=rhs)

                    self.Lambda_inequalities.append(new_constraints)

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

    def value_iteration_with_advantages(self, k, noise, cluster_error, threshold, exact):

        gather_query = []
        gather_diff = []
        self.adv = advantage.Advantage(self.mdp, cluster_error)

        d = self.mdp.d
        matrix_nd = np.zeros((self.n, d), dtype=ftype)
        v_d = np.zeros(d, dtype=ftype)

        best_p_and_v_d = ({s: [random.randint(0, self.na - 1)] for s in range(self.n)}, np.zeros(d, dtype=ftype))

        # k = 1
        for t in range(k):

            advantages_pair_vector_dic = self.mdp.calculate_advantages_labels(matrix_nd, True)
            cluster_advantages = self.adv.accumulate_advantage_clusters(matrix_nd, advantages_pair_vector_dic,
                                                                    cluster_error)
            policies = self.adv.declare_policies(cluster_advantages, best_p_and_v_d[0])

            for val in policies.itervalues():
                best_p_and_v_d = self.get_best_policies(best_p_and_v_d, val, noise)

            matrix_nd = self.mdp.update_matrix(policy_p=best_p_and_v_d[0], _Uvec_nd=matrix_nd)
            best_v_d = best_p_and_v_d[1]

            delta = linfDistance([np.array(best_v_d)], [np.array(v_d)], 'chebyshev')[0, 0]

            gather_query.append(self.query_counter_with_advantages)
            gather_diff.append(abs(np.dot(self.get_Lambda(), best_v_d) - np.dot(self.get_Lambda(), exact)))

            if delta < threshold:
                return best_v_d, gather_query, gather_diff
            else:
                v_d = best_v_d

        return (best_v_d, gather_query, gather_diff)

# ********************************************
def generate_inequalities(_d):
    inequalities = []

    for x in itertools.combinations(xrange(_d), 1):
        inequalities.append([0] + [1 if i in x else 0 for i in xrange(_d)])
        inequalities.append([1] + [-1 if i in x else 0 for i in xrange(_d)])

    return inequalities


def interior_easy_points(dim):
    # dim = len(self.points[0])
    l = []
    for i in range(dim):
        l.append(random.uniform(0.0, 1.0))
    return l
