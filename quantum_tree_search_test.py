from qTreeSearch import quantumTreeSearch as QTS
from executeCircuit import execute_circuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np 

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
'''

states=7
actions=3
action_set=list(range(actions))

transition_kernel= np.zeros(states,dtype=object)
steps = 1

#action, sprime
transition_kernel[0] = [(0,1),(1,2)]
transition_kernel[1] = [(0,6), (1,3) , (2,4)]
transition_kernel[2] = [(0,4),(1,5)]

for i in range(3,states):
	transition_kernel[i] = []
'''

'''
states=7
actions=3
action_set=list(range(actions))

transition_kernel= np.zeros(states,dtype=object)
steps = 1

#action, sprime
transition_kernel[0] = [(0,1),(1,2)]
transition_kernel[1] = [(0,6), (1,3) , (2,4)]
transition_kernel[2] = [(0,4),(1,5)]

for i in range(3,states):
	transition_kernel[i] = []
	
qts = QTS(tree=transition_kernel, n_states=states, action_set=action_set)

#circuit = qts.A([0], constant_branching=False)

q_tree = qts.traverse(depth=2, mode="depth")

#q_tree = qts.traverse(mode="iterative_deepning")

counts, actions = qts.measure(goal_state=4, iterations=None, shots=5000)#, noise_model_device="custom", prob_1=0.001, prob_2=0.001)

#counts, actions, depth = qts.measure(goal_state=7, iterations=None, noise_model_device="custom")

print("ACTIONS - {}".format(actions))
'''

states=15
actions=2
action_set=list(range(actions))

transition_kernel= np.zeros(states,dtype=object)

#action, sprime
transition_kernel[0] = [(0,1),(1,2)]
transition_kernel[1] = [(0,3) , (1,4)]
transition_kernel[2] = [(0,5),(1,6)]
transition_kernel[3] = [(0,7),(1,8)]
transition_kernel[4] = [(0,9),(1,10)]
transition_kernel[5] = [(0,11),(1,12)]
transition_kernel[6] = [(0,13),(1,14)]

for i in range(7,states):
	transition_kernel[i] = []

qts = QTS(tree=transition_kernel, n_states=states, action_set=action_set, constant_branching=True)

q_tree = qts.traverse(depth=3, mode="depth")

counts, actions = qts.measure(goal_state=12, iterations=None, shots=5000)

plot_histogram(counts)
plt.show()

qts.draw_circuit()

#cnot_number = qts.count_ops(op="cx")
#print("Number of CNOTS - {}".format(cnot_number))


