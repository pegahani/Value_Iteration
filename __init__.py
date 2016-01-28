from Weng import weng
import m_mdp
from advantage_iteration import avi
from matplotlib import pylab as plt

if __name__ == '__main__':

    _d = 4

    _Lambda_inequalities = avi.generate_inequalities(_d)
    _lambda_rand = avi.interior_easy_points(_d)
    print _lambda_rand
    # m = m_mdp.make_grid_VVMDP(_lambda_rand, n=3)

    # m = m_mdp.make_grid_VVMDP()

    _state, _action = 4, 5
    m = m_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand, None)

    print 'rewards', m.rewards

    w = avi(m, _lambda_rand, _Lambda_inequalities)
    Uvec = m.policy_iteration()  # returns the matrix of vectorial values of the best policy reached, starting from
    # values all equal to 0 and iterating until actions dont change any more
    exact = m.initial_states_distribution().dot(Uvec)  # expected vectorial value for this best policy
    sol_avi = w.value_iteration_with_advantages(limit=100000, noise=None,
                                            cluster_threshold=0.00001, min_change=0.001, exact=exact)
    print 'avi error', sol_avi[2][len(sol_avi[2])-1]

    w = weng(m, _lambda_rand, _Lambda_inequalities)
    sol_weng = w.value_iteration_weng(k=100000, noise= None, threshold=0.001, exact = exact)

    print 'weng error', sol_weng[2][len(sol_weng[2])-1]

    #print "weng result", sol_weng
    # print "avi result", sol_avi

    ax = plt.subplot(211)
    ax.plot(sol_weng[1], sol_weng[2],'b', marker='o')
    ax.set_title("Weng")
    ax = plt.subplot(212)
    ax.plot(sol_avi[1], sol_avi[2],'g',marker='o')
    ax.set_title("avi")

    plt.show()
