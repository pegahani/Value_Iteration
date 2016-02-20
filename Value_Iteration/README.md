# Value_Iteration using advantages



...........[***Look at the TODO list***]......... 

This module implements _advantage-based value iteration_. The `avi` class is in `advantage_iteration.py` ; it embeds 
a VVMDP which is in `m_mdp.py`.  Instances are created and the execution launched in \__init__.py

In the following description, instance variables are written in **boldface** 

## The VVMDP code
There are **nstates** states and **nactions** actions, indexed by integers. Data passed to the instance creation may
use human readable names for states and actions ; they are converted to an integer representation during initialization.
**states** and **actions** store these form (in index order) while the dictionaries **stateInd** and **actionInd** 
allow to retrieve the index from the readable form.
 
 **transitions** is an _nstates x nactions_ array of vectors. At a given [state, action] position, one finds a vector 
 of size s ; the 
 index in the vector represents a state, the value is the probability of transiting to that state, knowing the current 
 position.
 **rev_transitions** is an inversed index (without actions).  **auto_probability** is the probability that (s,a) leads 
 to a. **startingStateInd** is the list of starting states.
  
 Each state has a vectorial value of size **d**. The **rewards** and the vectorial value function (set of 1 vectorial 
 value  per state) are each represented as a _nstates x d_ matrix. 

  Policies are vectors of _nstates_ size, and which  contain integer values smaller than **nactions** (the action 
  associated to the state of this index). Policies may also contain a list of actions when *\_stationay* is positionned
  (in value_iteration)
  
## The avi code
  Instance variables are **mdp**, **Lambda**, **Lambda_inequalities**, **query_counter**, 
  **query_counter_with_advantages**.
  prob is a global variable refering a cplex instance.
  
  An inequality is a list of size _d + 1_. 
  
  An advantages dictionnary is a dictionary where keys are pairs (state, action) and values are advantages (vectors 
  of size d).  
  The set of clusters (clusters dictionary, or clusters advantages dictionary) is a dictionary of advantages 
  dictionaries where the key of each cluster is its id, that is to say a number.
  A clusters policy-values dictionary has the same cluster ids as keys, but values are pairs, with the 
  policy-list (list of pairs (state, action) in the cluster) as first element (each state at most one) and a list of 
  vectors, either the values or the advantages of these pairs, as second element
  Finally, a clusters policy-expval dictionary is very similar to the policy-values case, but the second element of the 
  value is a single vector (the expected value of the list of vectors in the policy-values case)
  
  A policy-value-pair is a pair formed of: a policy-dict, which is a dictionary of state:action correspondance, and the 
  value vector(size d) associated to this policy.
  
  