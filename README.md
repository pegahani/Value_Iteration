# Value_Iteration using advantages

...........[***Look at the TODO list***].........

This module implements _advantage-based value iteration_. The avi class is in advantage_iteration.py ; it embeds 
a VVMDP which is in m_mdp.py.  Instances are created and the execution launched in \__init__.py

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
  
  An advantages dictionnary is a dictionary where keys are pairs (staate, action) and values are advantages (vectors 
  of size d).  The set of clusters is a dictionary of advantages dictionaries where the key of each cluster is its 
  number.
  
  _It seems that we could merge Query and QueryPolicies on tne one side, get_best and get_best_policies on the 
  other side, since the \[1] can be used at calling time_