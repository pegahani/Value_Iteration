from Weng import weng
import m_mdp
from advantage_iteration import avi
from matplotlib import pylab as plt
import pickle
import time

if __name__ == '__main__':
    start = time.time()
    starts = time.clock()
    _d = 3

    # _lambda_rand = avi.interior_easy_points(_d)
    # pickle.dump(_lambda_rand,open("lambda.dmp", 'w'))
    _lambda_rand = pickle.load(open("lambda.dmp", 'r'))
    print _lambda_rand

    # m = m_mdp.make_grid_VVMDP(_lambda_rand, n=3)


    # _state, _action = 8, 5
    # m = m_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand, None)
    # m.save('mdp.dmp')
    m = m_mdp.reload('mdp.dmp')

    Uvec = m.value_iteration(epsilon=0.00001)
    # Uvec = m.policy_iteration()  # returns the matrix of vectorial values of the best policy reached, starting from
    # # values all equal to 0 and iterating until actions dont change any more
    # print Uvec
    #
    # Uvec1 = m.policy_iteration(200)
    # print Uvec - Uvec1
    # Uvec = m.policy_iteration(50)
    # print Uvec
    # Uvec1 = m.policy_iteration(50)
    # print Uvec - Uvec1

    exact = m.initial_states_distribution().dot(Uvec)  # expected vectorial value for this best policy

    #w = avi(m, _lambda_rand, _Lambda_inequalities)

    w = avi(m, _lambda_rand, [])
    sol_avi = w.value_iteration_with_advantages(limit=100000, noise=None,
                                            cluster_threshold=0.01, min_change=0.000001, exact=exact)
    print 'avi error', sol_avi[2][-1]
    print "Iterations", sol_avi[5],
    print "Queries", sum(sol_avi[1])

    # sol_AIweng = w.value_iteration_weng(k=100000, noise= None, threshold=0.001, exact = exact)
    # print 'AIweng error', sol_AIweng[2][len(sol_AIweng[2])-1]
    # print "Queries", sol_AIweng[1]

    # w = weng(m, _lambda_rand, _Lambda_inequalities)

    # w = weng(m, _lambda_rand, [])
    # sol_weng = w.value_iteration_weng(k=100000, noise= None, threshold=0.001, exact = exact)
    #
    # print 'weng error', sol_weng[2][-1]
    # print "Iterations", sol_weng[3],
    # print "Queries", sol_weng[1][-1]

    print "Pareto finds", w.pareto, "kDominance finds", w.kd, "queries performed", w.queries
    print "lastly generated clusters", w.nbclusters,
    print "hull used", sol_avi[3], "hull skipped", sol_avi[4]
    stop = time.time()
    stops = time.clock()
    print "wall clock time used", stop - start, "system time used", stops - starts

    #print "weng result", sol_weng
    # print "avi result", sol_avi

    # ax = plt.subplot(211)
    # ax.plot(sol_weng[1], sol_weng[2],'b', marker='o')
    # ax.set_title("Weng")
    # ax = plt.subplot(212)
    # ax.plot(sol_avi[1], sol_avi[2],'g',marker='o')
    # ax.set_title("avi")
    #
    # plt.show()
