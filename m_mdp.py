from collections import defaultdict
from itertools import product, izip, starmap, repeat, islice, ifilter
from operator import add
import math
import numpy as np
from toolz import first
import random

try:
    from scipy.sparse import csr_matrix, dok_matrix
    from scipy.spatial.distance import cityblock as l1distance
except:
    from sparse_mat import dok_matrix,csr_matrix,l1distance

ftype = np.float32

""" Vectorial mdp where rewards are vectors of size d, and scalar values are obtained by scalar product of rewards or utility functions
    with the lambda vector.
    The initial distribution is equal probability for each initial state. Two functions value_iteration and policy_iteration compute solutions
    with the help of callical algorithms.
    Note that a policy is an array of (state, action) pairs (stationary) or (state, list-of-actions) pairs (non stationary)"""

class VVMdp:

    def __init__(self,
                 _startingstate,
                 _transitions,  # dictionary of key:values   (s,a,s):proba
                 _rewards,  # dictionary of key:values   s: vector of rewards
                 _gamma=.9, _lambda = None):

        self._lambda = _lambda

        try:
            states = sorted(
                {st for (s, a, s2) in _transitions.iterkeys() for st in (s, s2)}
            )

            actions = sorted(
                {a for (s, a, s2) in _transitions.iterkeys()}
            )

            n , na = len(states) , len(actions)

            stateInd = {s: i for i, s in enumerate(states)}
            actionInd = {a: i for i, a in enumerate(actions)}

            assert set(_startingstate).issubset(stateInd.keys()), \
                 "initial states are not subset of total states"

            self.startingStateInd = [stateInd[x] for x in _startingstate]

            d = len(_rewards[first(_rewards.iterkeys())])
            assert all(d == len(np.array(v,dtype=ftype)) for v in _rewards.itervalues()),\
                    "incorrect reward vectors"


            assert set(_rewards.keys()).issubset(states) ,\
                    "states appearing in rewards should also appear in transitions"

        except ValueError,TypeError:

            print "transitions or rewards do not have the correct structure"
            raise

        # convert rewards to nstates x d matrix
        rewards = np.zeros((n,d), dtype=ftype)
        for s, rv in _rewards.iteritems():
            rewards[stateInd[s], :] = rv

        self.rmax = np.max( [sum(abs(rewards[s,:])) for s in range(n)] )

        transitions = np.array(
        [[ dok_matrix((1, n), dtype=ftype) for _ in actions ] for _ in states ], dtype=object )

        rev_transitions = defaultdict(set)

        for (s, a, s2), p in _transitions.iteritems():
            si, ai, si2 = stateInd[s], actionInd[a], stateInd[s2]
            transitions[si,ai][0, si2] = p
            rev_transitions[si2].add(si)

        for s, a in product(range(n), range(na)):
            transitions[s,a] = transitions[s,a].tocsr()
            assert 0.99 <= transitions[s,a].sum() <= 1.01, "probability transitions should sum up to 1"

        # autoprobability[s,a] = P(s|s,a)
        self.auto_probability = np.array( [[transitions[s,a][0,s] for a in range(na)] for s in range(n)] ,dtype=ftype )

        # copy local variables in object variables
        self.states , self.actions , self.nstates , self.nactions, self.d = states,actions,n,na,d
        self.stateInd,self.actionInd = stateInd,actionInd
        self.rewards , self.transitions, self.rev_transitions = rewards , transitions, rev_transitions
        self.gamma = _gamma

        #E_test = np.zeros((n*na, n), dtype=ftype)
        E_test = dok_matrix((n*na, n), dtype=ftype)

        for s in range(n):
            for a in range(na):
                E_test[s*na+a, :] = [ transitions[s,a][0,i] for i in range(n) ]

        self.E_test = E_test

    def T(self, state, action):
        """Transition model.  From a state and an action, return all
        of (state , probability) pairs."""
        _tr = self.transitions[state,action]
        return izip(_tr.indices, _tr.data)

    def set_Lambda(self,l):
        self.Lambda = np.array(l, dtype=ftype)

    def get_lambda(self):
        return self.Lambda

    def initial_states_distribution(self):
        n = self.nstates
        _init_states = np.zeros(n, dtype=ftype)

        init_n = len(self.startingStateInd)

        for i in range(n):
            if i in self.startingStateInd:
                _init_states[i] = ftype(1)/ftype(init_n)
                #_init_states[i] = 1.0
            else:
                _init_states[i] = 0.0

        return _init_states

    def expected_vec_utility(self,s,a, Uvec):
        "The expected vector utility of doing a in state s, according to the MDP and U."
        # Uvec is a (nxd) matrix
        return np.sum( (p*Uvec[s2] for s2,p in self.T(s,a)) )

    def get_vec_Q(self,s,a,Uvec):
        # Uvec is a (nxd) matrix
        return self.rewards[s] + self.gamma * self.expected_vec_utility(s,a,Uvec)

    def expected_scalar_utility(self,s,a,U):
        # U is a n-dimensional vector
        return self.transitions[s,a].dot(U)

    def expected_dot_utility(self,s,a,Uvec):
        # assumes self.Lambda numpy array exists
        # Uvec is a (nxd) matrix
        return sum( (p*(Uvec[s2].dot(self.Lambda)) for s2,p in self.T(s,a)) )

    def value_iteration(self, epsilon=0.001,policy=None,k=100000,_Uvec=None, _stationary= True):
        "Solving an MDP by value iteration. [Fig. 17.4]. Stops when the improvement is less than epsilon"
        n , na, d , Lambda = self.nstates , self.nactions, self.d , self.Lambda
        gamma , R , expected_scalar_utility = self.gamma , self.rewards , self.expected_scalar_utility

        Udot = np.zeros( n , dtype=ftype)
        _uvec= np.zeros( d , dtype=ftype)
        Uvec = np.zeros( (n,d) , dtype=ftype)
        if _Uvec is not None:
            Uvec[:] = _Uvec

        Q    = np.zeros( na , dtype=ftype )

        for t in range(k): # bounds the number of iterations if the break condition is too weak

            delta = 0.0
            for s in range(n):

                # Choose the action
                if policy != None:
                    if _stationary:
                        act = random.choice(policy[s])
                    else:
                        act = policy[s]

                else:
                    Q[:]    = [expected_scalar_utility(s, a, Udot) for a in range(na)]
                    act     = np.argmax(Q)

                # Compute the update
                _uvec[:] = R[s] + gamma * self.expected_vec_utility(s,act,Uvec) # vectorial utility of the best action 
                _udot    = Lambda.dot(_uvec) # its scalar utility

                if policy != None:
                    delta = max(delta, l1distance(_uvec , Uvec[s]) )
                else:
                    delta = max(delta, abs(_udot-Udot[s]) )

                Uvec[s] = _uvec
                Udot[s] = _udot

            if delta < epsilon * (1 - gamma) / gamma:
                return Uvec
        return Uvec

    def best_action(self,s,U):
        # U is a (nxd) matrix
        # Lambda has to be defined
        return np.argmax( [self.expected_dot_utility(s,a,U) for a in range(self.nactions)] )


    def policy_iteration(self,_Uvec=None):
        """Solve an MDP by policy iteration [Fig. 17.7]. Tries 20 value iterations, then test if the policiy has changed
        and stops if not."""
        if _Uvec == None:
            U = np.zeros( (self.nstates,self.d) , dtype=ftype)
        else:
            U = _Uvec

        pi = {s:random.randint(0,self.nactions-1) for s in range(self.nstates)}
        while True:
            U = self.value_iteration(epsilon=0.0,policy=pi, k=20,_Uvec=U, _stationary=False)
            unchanged = True
            for s in range(self.nstates):
                a = self.best_action(s,U)
                if a != pi[s]:
                    pi[s] = a
                    unchanged = False
            if unchanged:
                return U

    def calculate_advantages_labels(self, _matrix_nd, _IsInitialDistribution, policy):
        """
        This function get a matrix and finds all |S|x|A| advantages.  It is unused in the class (17/1/2016)
        :param _matrix_nd: a matrix of dimension nxd which is required to calculate advantages (the actual vectorial utility function)
        :param _IsInitialDistribution: if initial distribution should be considered in advantage calculation or not
        :param policy: unused
        :return: a dictionary of all advantages for our MDP. keys are pairs and values are advantages vectros
                for instance: for state s and action a and d= 3 we have: (s,a): [0.1,0.2,0.4]
        """

        n, na = self.nstates, self.nactions
        advantage_dic = {}
        init_distribution = self.initial_states_distribution()

        for s in range(n):
            for a in range(na):
                advantage_d = self.get_vec_Q(s,a,_matrix_nd)-_matrix_nd[s]
                if _IsInitialDistribution:
                    advantage_dic[(s,a)] = init_distribution[s]*( advantage_d)
                    #advantage_dic[(s,a)] = advantage_d
                else:
                    advantage_dic[(s,a)]= advantage_d

        return advantage_dic

    def update_matrix(self, policy_p,_Uvec_nd):

        """
        This function receives an updated policy after considering advantages and the old nxd matrix
        and it returns the updated matrix related to new policy. Unused in this class (17/1/2016)
        :param policy_p: a given policy
        :param _Uvec_nd: nxd matrix before implementing new policy
        :return: nxd matrix after improvement
        """

        n, d = self.nstates, self.d
        gamma , R  = self.gamma , self.rewards

        _uvec_nd= np.zeros((n,d) , dtype=ftype)

        for s in range(n):
                act = random.choice(policy_p[s])
                # Compute the update
                _uvec_nd[s,:] = R[s] + gamma * self.expected_vec_utility(s, act, _Uvec_nd)

        return _uvec_nd


#********************************************************************************************

def make_simulate_mdp_Yann(n_states, n_actions, _lambda, _r=None):
    """ Builds a random MDP.
        Each state has ceil(log(nstates)) successors.
        Reward vectors are permutations of [1,0,...,0]
    """

    nsuccessors = int(math.ceil(math.log1p(n_states)))
    gauss_iter  = starmap(random.gauss,repeat((0.5,0.5)))
    _t = {}

    for s,a in product(range(n_states), range(n_actions)):

        next_states = random.sample(range(n_states), nsuccessors)
        probas =  np.fromiter(islice(ifilter(lambda x: 0 < x < 1 ,gauss_iter),nsuccessors), ftype)

        _t.update(  {(s,a,s2):p for s2,p in izip(next_states, probas/sum(probas) )  }  )

    if _r is None:
        #_r = {i:np.random.permutation([random.randint(1,5)]+[0]*(len(_lambda)-1)) for i in range(n_states)}
        _r = {i:np.random.permutation([1]+[0]*(len(_lambda)-1)) for i in range(n_states)}

    assert len(_r[0])==len(_lambda),"Reward vectors should have same length as lambda"

    return VVMdp(
        _startingstate= set(range(n_states)),
        _transitions= _t,
        _rewards= _r ,
        _gamma= 0.95,
        _lambda=_lambda)
#********************************************************************************************

def make_grid_VVMDP(n=2):
    _t =       { ((i,j),'v',(min(i+1,n-1),j)):0.9 for i,j in product(range(n),range(n)) }
    _t.update( { ((i,j),'v',(max(i-1,0),j)):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'^',(max(i-1,0),j)):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'^',(min(i+1,n-1),j)):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'>',(i,min(j+1,n-1))):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'>',(i,max(j-1,0))):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'<',(i,max(j-1,0))):0.9 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'<',(i,min(j+1,n-1))):0.1 for i,j in product(range(n),range(n)) } )
    _t.update( { ((i,j),'X',(i,j)):1 for i,j in product(range(n),range(n)) } )

    _r = { (i,j):[0.0,0.0] for i,j in product(range(n),range(n))}
    _r[(n-1,0)] = [1.0,0.0]
    _r[(0,n-1)] = [0.0,1.0]
    _r[(n-1,n-1)] = [1.0,1.0]

    gridMdp = VVMdp (
        _startingstate= {(0,1)}, #{(0,0)},
        _transitions= _t,
        _rewards=_r
    )
    return gridMdp



