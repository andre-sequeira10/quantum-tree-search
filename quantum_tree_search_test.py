from qTreeSearch import quantumTreeSearch as QTS
from executeCircuit import execute_circuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np 
'''
states=5
actions=4
action_set=list(range(actions))

transition_kernel= np.zeros(states,dtype=object)
steps = 1

#action, sprime
transition_kernel[0] = [(0,1),(1,2)]
transition_kernel[1] = [(1,3) , (2,4)]

for i in range(2,states):
	transition_kernel[i] = []
'''

states=5
actions=4
action_set=list(range(actions))

transition_kernel= np.zeros(states,dtype=object)
steps = 1

#action, sprime
transition_kernel[0] = [(0,1),(1,2),(2,3)]
transition_kernel[1] = [(1,4) , (3,5)]
transition_kernel[2] = []
transition_kernel[3] = [(1,6),(2,7),(3,8)]

for i in range(4,states):
	transition_kernel[i] = []

	
qts = QTS(tree=transition_kernel, n_states=states, action_set=action_set)

q_tree = qts.traverse(depth=2, mode="depth")
counts, actions = qts.measure(goal_state=4, iterations=1)


print(actions)
plot_histogram(counts)

plt.show()

cnot_count = qts.count_ops(op="cx")

print("Number of CNOTs - {}".format(cnot_count))


