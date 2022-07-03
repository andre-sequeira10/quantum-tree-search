# quantum-tree-search

Quantum tree search algorithm for non-constant branching factor trees. <br>
Developed in Qiskit (qiskit v0.37)<br>

Example: <br>

### Import

from qTreeSearch import quantumTreeSearch as QTS

states=5
actions=4
action_set=list(range(actions))

transition_kernel= np.zeros(states,dtype=object)
steps = 1

#action, sprime
transition_kernel[0] = [(0,1),(1,2)]
transition_kernel[1] = [(1,3) , (2,4)]
transition_kernel[2] = [(0,2),(1,2)]

for i in range(3,states):
	transition_kernel[i] = []

qts = QTS(tree=transition_kernel, n_states=states, action_set=action_set)

q_tree = qts.traverse(depth=2, mode="depth")
counts = qts.measure(goal_state=3, iterations=1)



