from this import d
import numpy as np 
from qiskit import * 
from qiskit.circuit.library import StatePreparation
import matplotlib.pyplot as plt
from executeCircuit import execute_circuit, basis_states_probs
from itertools import chain
from qiskit.converters import circuit_to_dag
from qiskit.compiler import transpile


class quantumTreeSearch:
	def __init__(self, tree=None, n_states=None, n_actions=None, action_set=None, constant_branching=False) -> None:

		self.tree = tree
		self.n_states = n_states
		self.action_set = action_set
		self.n_actions = len(action_set)
		self.constant_b = constant_branching
		self.branching = []
		
		self.mode = None

		if self.tree is None:
			raise ValueError("Tree must be specified")
		if self.n_states is None:
			raise ValueError("number of states must be specidified")
		if self.n_actions is None and self.action_set is None:
			raise ValueError("number of actions or action set must be specidified")

		self.a_qubits = int(np.ceil(np.log2(self.n_actions)))
		#self.s_qubits = int(np.ceil(np.log2(self.n_states)))
		self.s_qubits = 1


	def A(self, state, constant_branching=True):

		if state is None:
			raise ValueError("reachable states should be specified")

		s = QuantumRegister(self.s_qubits)
		a = QuantumRegister(self.a_qubits)

		circuit = QuantumCircuit(s,a, name=r"$\mathcal{A}$")
		
		if not constant_branching:
			
			a_s = len(self.tree[state])
			self.branching.append(a_s)

			state_a = [complex(0.0,0.0) for i in range(2**self.a_qubits)]

			#create uniform superposition over the set of admissible actions
			for (a_d,sprime) in self.tree[state]:
				state_a[a_d] += complex(1/np.sqrt(a_s) , 0.0)

			#sbin=bin(state)[2:].zfill(self.s_qubits)
			sbin='1'

			ctrl_init_a = StatePreparation(state_a, label=r"$\mathcal{A}$").control(self.s_qubits, ctrl_state=sbin)

			circuit = circuit.compose(ctrl_init_a, [i for i in s]+[i for i in a])

		return circuit

	def T(self, state, a_d):
		
		s = QuantumRegister(self.s_qubits)
		a = QuantumRegister(self.a_qubits)
		sprime = QuantumRegister(self.s_qubits)

		circuit = QuantumCircuit(s,a,sprime, name=r"$\mathcal{T}$")
		
		sbin='1'

		abin = bin(a_d)[2:].zfill(self.a_qubits)
		state_v = [complex(0.0,0.0), complex(1.0,0.0)]
		
		ctrl_init_t = StatePreparation(state_v, label=r"$\mathcal{T}$").control(self.s_qubits+self.a_qubits, ctrl_state=abin+sbin)
		
		circuit = circuit.compose(ctrl_init_t, [i for i in s]+[i for i in a]+[i for i in sprime])

		return circuit

	def traverse(self, depth=None, mode="depth"):
		
		if self.mode == None:
			self.mode = mode
		
		self.branching = []
		if self.constant_b:
			self.branching.append(self.n_actions)

		if mode == "iterative_deepning":

			self.q_tree = None #self.traverse(depth=depth, mode="depth")
		

		elif mode == "depth":
			if depth is None:
				raise ValueError("Depth is missing")
			
			self.depth = depth 

			self.actions_d = {}
			for d in range(depth):
				self.actions_d["action{0}".format(d)] = QuantumRegister(self.a_qubits,"a{0}".format(d))

			self.states_d = {}
			for i in range(self.n_states):
				self.states_d["states_d{0}".format(i)]=QuantumRegister(self.s_qubits,name="s{}".format(i))

	
			self.q_tree = QuantumCircuit()
			for i in range(self.n_states):
				self.q_tree.add_register(self.states_d["states_d{0}".format(i)])

			self.q_tree.x(self.states_d["states_d{0}".format(0)])

			neighbours = [0]
			n=[]

			if self.constant_b:
				for d in range(1,depth+1):
					self.q_tree.add_register(self.actions_d["action{0}".format(d-1)])
					self.q_tree.h(self.actions_d["action{0}".format(d-1)])
			
			for d in range(1,depth+1):
				if not self.constant_b:
					self.q_tree.add_register(self.actions_d["action{0}".format(d-1)])
			 
				for s in neighbours:
					n.append([n_s for (_,n_s) in self.tree[s]])

				for state in neighbours:
					if self.tree[state] != []:
						
						a_d = self.A(state, constant_branching=self.constant_b)
					
						regs = [i for i in self.states_d["states_d{0}".format(state)]] + [i for i in self.actions_d["action{0}".format(d-1)]]

						self.q_tree = self.q_tree.compose(a_d, regs)
						self.q_tree.barrier()

						for (a_d,sp) in self.tree[state]:
							regs = [i for i in self.states_d["states_d{0}".format(state)]] + [i for i in self.actions_d["action{0}".format(d-1)]] + [i for i in self.states_d["states_d{0}".format(sp)]]


							#t_d = self.T(neighbours)
							t_d = self.T(state, a_d)

							self.q_tree = self.q_tree.compose(t_d, regs)
							self.q_tree.barrier()

				neighbours = list(chain(*n))
				
				#check if we have only leafs
				states_list = [self.tree[state] == [] for state in neighbours]
				if states_list[0] and len(states_list):
					self.leafs = True
					break
		return self.q_tree

	def measure(self, goal_state=None, iterations=None, shots=1024, noise_model_device=None, prob_1=0.001, prob_2=0.001):

		self.goal_state = goal_state

		if self.goal_state == None:
			self.a_classical = {}
			for d in range(self.depth):
				self.a_classical["actions{0}".format(d)]=ClassicalRegister(self.a_qubits,name="ac{}".format(d))
				self.q_tree.add_register(self.a_classical["actions{0}".format(d)])
				self.q_tree.measure(self.actions_d["action{0}".format(d)], self.a_classical["actions{0}".format(d)])
			counts = execute_circuit(self.q_tree,shots=shots)
			return counts

		else:
			
			self.leafs = False
			inc = 1
			while not self.leafs:

				if self.mode == "iterative_deepning":

					self.q_tree = self.traverse(depth=inc, mode="depth")
				else:
					self.leafs = True
				
				self.avg_branching = np.round(np.mean(self.branching))
				#print("avg - {}".format(self.avg_branching))

				if iterations == None or self.mode == 'iterative_deepning':
					iterations = int(np.floor(np.pi/4 * np.sqrt(self.avg_branching**self.depth)))

				#############################
				########## ORACLE ###########
				#############################

				oracle = QuantumCircuit(self.s_qubits)
				oracle.z(0)

				self.q_tree_inverse = self.q_tree.inverse()

				self.diffusion_circuit = QuantumCircuit()
				#self.diffusion_circuit.add_register(self.states_d["states_d{0}".format(0)])

				#regs = [i for i in self.states_d["states_d{0}".format(0)]]\
				regs=[]
				for i in range(self.n_states):
					regs += [i for i in self.states_d["states_d{0}".format(i)]]
					self.diffusion_circuit.add_register(self.states_d["states_d{0}".format(i)])

				regs_diffusion = []
				for d in range(self.depth):
					self.diffusion_circuit.add_register(self.actions_d["action{0}".format(d)])
					regs += [i for i in self.actions_d["action{0}".format(d)]]
					regs_diffusion += [i for i in self.actions_d["action{0}".format(d)]]

				self.diffusion_circuit = self.diffusion_circuit.compose(self.q_tree_inverse, regs)
				self.diffusion_circuit.x(regs_diffusion)
				#self.diffusion_circuit.x(regs)
				self.diffusion_circuit.h(regs_diffusion[-1])
				#self.diffusion_circuit.h(regs[-1])
				self.diffusion_circuit.mct(regs_diffusion[:-1], regs_diffusion[-1])
				#self.diffusion_circuit.mct(regs[:-1], regs[-1])
				self.diffusion_circuit.h(regs_diffusion[-1])
				#self.diffusion_circuit.h(regs[-1])
				self.diffusion_circuit.x(regs_diffusion)
				#self.diffusion_circuit.x(regs)
				self.diffusion_circuit = self.diffusion_circuit.compose(self.q_tree, regs)

				for i in range(iterations):
					self.q_tree.barrier()
					self.q_tree = self.q_tree.compose(oracle, [i for i in self.states_d["states_d{0}".format(self.goal_state)]])
					self.q_tree.barrier()
					self.q_tree = self.q_tree.compose(self.diffusion_circuit, regs)
				
				self.q_tree.barrier()
				self.a_classical = {}
				for d in range(self.depth):
					self.a_classical["actions{0}".format(d)]=ClassicalRegister(self.a_qubits,name="ac{}".format(d))
					self.q_tree.add_register(self.a_classical["actions{0}".format(d)])
					self.q_tree.measure(self.actions_d["action{0}".format(d)], self.a_classical["actions{0}".format(d)])
				if self.mode == "iterative_deepning":
					self.s_classical = ClassicalRegister(self.s_qubits)
					self.q_tree.add_register(self.s_classical)
					self.q_tree.measure(self.states_d["states_d{0}".format(self.goal_state)], self.s_classical)
				

				counts = execute_circuit(self.q_tree, shots=shots, noise_model_device=noise_model_device, prob_1=prob_1, prob_2=prob_2)


				new_counts = {}
				optimal_action_seq = None
				optimal_action_counter = 0
				optimal_goal_state = 0

				for k in counts:
					k_list = k.split()
					k_reversed = list(map(lambda x: x[::-1], k_list))
					k_reversed_int = list(map( lambda x: int(x,2), k_reversed))
					state = k_reversed_int[-1]
					k_reversed_str = list(map( lambda x: str(x), k_reversed_int))
					k_new = ' '.join(k_reversed_str)
					new_counts[k_new] = counts[k]
					if new_counts[k_new] > optimal_action_counter:
						optimal_action_seq = k_new
						optimal_action_counter = new_counts[k_new]
						optimal_goal_state = state
				
				if optimal_goal_state: #== self.goal_state:
					self.leafs = True
				else:
					inc+=1


			if self.mode == 'iterative_deepning':
				return new_counts, optimal_action_seq, inc
			else:
				return new_counts, optimal_action_seq
	
	def draw_circuit(self, output="mpl", block=None):

		#add option for parts of the circuit only
		if block is not None:
			pass
		else:
			self.q_tree.draw(output=output)
			plt.show()


	def count_ops(self, op="cx", device=None):
        
		if device is None:
			device = Aer.get_backend("qasm_simulator")

		qc_transpiled = transpile(self.q_tree, backend=device)
		qc_transpiled_dag = circuit_to_dag(qc_transpiled)

		return qc_transpiled_dag.count_ops()[op]


	
