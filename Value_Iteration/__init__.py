from Weng import weng
import m_mdp
from advantage_iteration import avi
from matplotlib import pylab as plt
import pickle
import time


def create_save(s, a, _id = ""):
    """
    Creates a new mdp, initialize related global variables and saves what is needed for reuse
    :type _id: string e.g. 80-1 to save in param80-1.dmp
    """
    name = "param" + _id + ".dmp"
    pp = pickle.Pickler(open(name, 'w'))
    global _lambda_rand
    global _state
    global _action
    global m
    global Uvec

    _lambda_rand = avi.interior_easy_points(_d)
    _state, _action = (s, a)
    #m = m_mdp.make_grid_VVMDP(_lambda_rand, n=2)
    m = m_mdp.make_simulate_mdp_Yann(_state, _action, _lambda_rand, None)
    Uvec = m.value_iteration(epsilon=0.00001)

    pp.dump(_lambda_rand)
    pp.dump((_state,_action))
    pp.dump(m)
    pp.dump(Uvec)

def reload(_id = ""):
    """
    Reloads a saved mdp and initialize related global variables
    :type _id: string e.g. 80-1 to reload param80-1.dmp
    """
    name = "param" + _id + ".dmp"
    pup = pickle.Unpickler(open("param80-1.dmp", 'r'))
    global _lambda_rand
    global _state
    global _action
    global m
    global Uvec

    _lambda_rand = pup.load()
    state, action = pup.load()
    m = pup.load()
    Uvec = pup.load()

def aviexec():
    global _lambda_rand
    global m
    global sol_avi
    w = avi(m, _lambda_rand, [])
    sol_avi = w.value_iteration_with_advantages(limit=100000, noise=None,
                                           cluster_threshold=0.01, min_change=0.000001, exact=exact)
    print 'avi error', sol_avi[2][-1]
    print "Iterations", sol_avi[5],
    print "Queries", sol_avi[1][-1]

    print "Pareto finds", w.pareto, "kDominance finds", w.kd, "queries performed", w.queries
    print "lastly generated clusters", w.nbclusters,
    print "hull used", sol_avi[3], "hull skipped", sol_avi[4]
    # print "avi result", sol_avi

def wengexec():
    global _lambda_rand
    global m
    global sol_weng
    w = weng(m, _lambda_rand, [])
    sol_weng = w.value_iteration_weng(k=100000, noise= None, threshold=0.001, exact = exact)

    print 'weng error', sol_weng[2][-1]
    print "Iterations", sol_weng[3],
    print "Queries", sol_weng[1][-1]

    print "Pareto finds", w.pareto, "kDominance finds", w.kd, "queries performed", w.queries
    #print "weng result", sol_weng


if __name__ == '__main__':

    start = time.time()
    starts = time.clock()
    _d = 2

    create_save(8, 5, _id = "8-2")
    #reload(_id = "8-2")

    print _lambda_rand
    print Uvec
    exact = m.initial_states_distribution().dot(Uvec)  # expected vectorial value for this best policy
    print exact

    aviexec()
    wengexec()

    stop = time.time()
    stops = time.clock()
    print "wall clock time used", stop - start, "system time used", stops - starts


    ax = plt.subplot(211)
    ax.plot(sol_weng[1], sol_weng[2],'b', marker='o')
    ax.set_title("Weng")
    ax = plt.subplot(212)
    ax.plot(sol_avi[1], sol_avi[2],'g',marker='o')
    ax.set_title("avi")

    plt.show()
