import m_mdp
import advantage_iteration
import pylab as plt

if __name__ == '__main__':
    _d = 2

    _Lambda_inequalities = advantage_iteration.avi.generate_inequalities(_d)
    _lambda_rand = advantage_iteration.avi.interior_easy_points(_d)
    print _lambda_rand
    m = m_mdp.make_grid_VVMDP(_lambda_rand, n=3)

    # _state, _action = 4, 5
    # m = m_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand, None)

    w = advantage_iteration.avi(m, _lambda_rand, _Lambda_inequalities)
    # w.setStateAction()  # should be in w.__init__ and implicitely performed by the previous line

    # m.set_Lambda(_lambda_rand)  # should rather be set as argument of make_grid_VVMDP() line 11
    Uvec = m.policy_iteration()  # returns the matrix of vectorial values of the best policy reached, starting from

    # values all equal to 0 and iterating until actions dont change any more
    exact = m.initial_states_distribution().dot(Uvec)  # expected vectorial value for this best policy

    sol = w.value_iteration_with_advantages(limit=100000, noise=None,
                                            cluster_threshold=0.00001, min_change=0.001, exact=exact)

    # sol = w.value_iteration_weng(k=100000, noise= 0.5, threshold=0.0001, exact = exact)

    # print 'error', sol[2][len(sol[2]) - 1]
    print sol

    # ax = plt.subplot(111)
    # ax.plot(sol[1], sol[2], 'b')
    # plt.show()
