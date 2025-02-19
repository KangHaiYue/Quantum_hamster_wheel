# -*- coding: utf-8 -*-
"""
Last Updated on 7 Feburary, 2024
@author: Haiyue Kang
"""

# Standard libraries
import queue as Q
import itertools
import copy
from ast import literal_eval
from datetime import datetime
from math import ceil
import random
from collections import Counter
from multiprocessing import Pool
#other installed libraries
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
import pickle
from time import sleep
import numpy as np
import numpy.linalg as la
import networkx as nx
# Qiskit libraries
from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute, transpile
from qiskit.quantum_info import partial_trace, Statevector, DensityMatrix, Operator, PauliList

#pytket libraries
from pytket.circuit import Circuit, BitRegister, if_bit, reg_eq
from pytket.extensions.qiskit import qiskit_to_tk,tk_to_qiskit
from pytket.circuit.display import render_circuit_jupyter
from pytket.backends import ResultHandle

# Local modules
from utilities import pauli_n, bit_str_list, run_cal, load_cal, pauli_product
from teleportation import calc_n, calc_f, calc_s, ptrans



# Two-qubit Pauli basis
basis_list = ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
ext_basis_list = ['II', 'IX', 'IY', 'IZ',
                  'XI', 'XX', 'XY', 'XZ',
                  'YI', 'YX', 'YY', 'YZ',
                  'ZI', 'ZX', 'ZY', 'ZZ']
#I tensor H operator in matrix
H1 = np.kron(np.array([[1, 0],[0, 1]]), 1/np.sqrt(2)*np.array([[1, 1], [1, -1]]))
#I tensor X operator in matrix
X1 = np.kron(np.array([[1, 0],[0, 1]]), np.array([[0, 1], [1, 0]]))
#Base Bell State
base_circ = QuantumCircuit(2)
base_circ.h(0)
base_circ.cx(0,1)

BS_list_odd = []

# 4 variants of Bellstate
BS_1 = base_circ.copy()
BS_list_odd.append(BS_1) #|00>+|11>
BS_2 = base_circ.copy()
BS_2.x(0)
BS_list_odd.append(BS_2) #|01>+|10>
BS_3 = base_circ.copy()
BS_3.z(0)
BS_list_odd.append(BS_3) #|00>-|11>
BS_4 = base_circ.copy()
BS_4.x(0)
BS_4.z(0)
BS_list_odd.append(BS_4) #|01>-|10>

BS_list_even = copy.deepcopy(BS_list_odd)
#Bellstate up to local transformation (this is what we should obtain from qst)
for circuit in BS_list_even:
    circuit.h(0)

States_odd = []
States_even = []
for circuit in BS_list_odd:
    state = Statevector(circuit)
    States_odd.append(state)
for circuit in BS_list_even:
    state = Statevector(circuit)
    States_even.append(state)
#Direct Matrix form of the 8 variants (4 variants up to Hadamard)
BS_list_odd[0] = 1/np.sqrt(2)*np.array([[1],[0],[0],[1]])
BS_list_odd[1] = 1/np.sqrt(2)*np.array([[0],[1],[1],[0]])
BS_list_odd[2] = 1/np.sqrt(2)*np.array([[1],[0],[0],[-1]])
BS_list_odd[3] = 1/np.sqrt(2)*np.array([[0],[1],[-1],[0]])
BS_list_even[0] = 1/2*np.array([[1],[1],[1],[-1]])
BS_list_even[1] = 1/2*np.array([[1],[-1],[1],[1]])
BS_list_even[2] = 1/2*np.array([[1],[1],[-1],[1]])
BS_list_even[3] = 1/2*np.array([[-1],[1],[1],[1]])

# ref: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetTable
POPCOUNT_TABLE16 = [0, 1] * 2**15
for index in range(2, len(POPCOUNT_TABLE16)):  # 0 and 1 are trivial
    POPCOUNT_TABLE16[index] += POPCOUNT_TABLE16[index >> 1]





class quantinuum_teleportation:
    def __init__(self, backend, path_to_teleport):
        self.backend = backend
        self.device_name = backend._device_name
        self.path_to_teleport = path_to_teleport
        self.nqubits = len(path_to_teleport)

        info = backend.backend_info
        self.info = info
        self.device_size = len(info.architecture.nodes)

        self.name_list = None
        self.shots = None
        self.qrem_shots = None
        self.qrem = None

        self.M_list = None
        self.qrem_circuits = None
        
        self.teleported_BellState_circuit = None
        #self.qreg = None
        self.teleported_BellState_circuits_qst = None

        #self.circuit = self.gen_chain_graphstate_circuit()

    def gen_chain_graphstate_circuit(self, chain):
        
        circ = QuantumCircuit(self.device_size)#, self.nqubits)

        # Apply Hadamard gates to every qubit
        circ.h(chain)
        # Connect every edge with cz gates
        #pi = 3.141592653589793
        for i in range(0,len(chain)-1,2):
            circ.cz(chain[i], chain[i+1])
            #circ.cp(pi, self.path_to_teleport[i], self.path_to_teleport[i+1])
        for i in range(1,len(chain)-1,2):
            circ.cz(chain[i], chain[i+1])
            #circ.cp(pi,self.path_to_teleport[i], self.path_to_teleport[i+1])
        
        return circ
    
    def gen_two_qubit_graphstate_circuit(self, qubit_a, qubit_b):
        circ = QuantumCircuit(self.device_size)
        connection_order = [qubit_a] + [qubit_b]

        # Apply Hadamard gates to every qubit
        circ.h(connection_order)
        # Connect every edge with cz gates
        circ.cz(qubit_a, qubit_b)

        return circ
    
    def gen_teleported_BellState_circuit(self, paths, mode = 'post select'):
        self.mode = mode
        #max_gap = self.nqubits - 2
        
        #BellState_circuits = {gap: {} for gap in range(1, max_gap+1)}
        BellState_circuits = {gap: {} for gap in paths.keys()}
        
        for gap, path_list in paths.items():
            for j in range(len(path_list)):
                path = paths[gap][j]
                #path = tuple(range(gap+2))
                #path = random.sample(range(self.path_to_teleport), gap+2)
            
                if mode == 'post select':
                    circ = self.gen_chain_graphstate_circuit(chain=path)
                    circ.name = f'{gap}-{path}'
                    #circ = self.circuit.copy(f'{gap}-{self.path_to_teleport}')
                    circ.barrier()
                    crX = ClassicalRegister(gap)
                    circ.add_register(crX)
        
                    #circ.barrier()
                    #measure intermediate qubits in X basis
                    X_basis_qubits = path[1:-1]
                    circ.h(np.array(X_basis_qubits).flatten())
                    circ.measure(np.array(X_basis_qubits).flatten(), crX)
                    #circ.measure(np.array(X_basis_qubits).flatten(), X_basis_qubits)

                elif mode == 'extended post select':
                    circ = self.gen_chain_graphstate_circuit(chain=path[:20])
                    circ.name = f'{gap}-{path}'
                    #circ = self.circuit.copy(f'{gap}-{self.path_to_teleport}')
                    circ.barrier()
                    
                    c_reg_dict = {}
                    for i in range(gap):
                        #vars()[f'x{i}'] = ClassicalRegister(1)
                        c_reg_dict[i] = ClassicalRegister(1)
                        circ.add_register(c_reg_dict[i])
                    
                    #crX = ClassicalRegister(gap)
                    #circ.add_register(crX)
                    
                    X_basis_qubits = path[1:19]
                    for qubit in X_basis_qubits:
                        circ.h(qubit)
                    for qubit in X_basis_qubits:
                        #circ.Measure(qreg[X_basis_qubits[i]], vars()[f'X{i}'][0])
                        circ.measure(qubit, c_reg_dict[qubit-1])

                    
                    loops = ceil((len(path)-20)/18)
                    for loop in range(1, loops+1):
                        # Re-initialize mid qubits
                        for qubit in X_basis_qubits:
                            circ.reset(qubit)
        
                        if loop == loops:
                            #last scan
                            for qubit in path[2+18*loop:]:
                                circ.h(qubit)
                            for i in range(1+18*loop, gap+1, 2):
                                circ.cz(path[i], path[i+1])
                            for i in range(2+18*loop, gap+1, 2):
                                circ.cz(path[i], path[i+1])
                            circ.barrier()
                    
                            X_basis_qubits = path[1+18*loop:-1]
                            for qubit in X_basis_qubits:
                                circ.h(qubit)
                            for i in range(len(X_basis_qubits)):
                                #circ.Measure(qreg[X_basis_qubits[i]], vars()[f'X{i}'][0])
                                circ.measure(X_basis_qubits[i], c_reg_dict[i+18*loop])
                    
                        else:
                            #next scan
                            for qubit in path[2+18*loop: 2+18*(loop+1)]:
                                circ.h(qubit)
                            for i in range(1+18*loop, 1+18*(loop+1), 2):
                                circ.cz(path[i], path[i+1])
                            for i in range(2+18*loop, 2+18*(loop+1), 2):
                                circ.cz(path[i], path[i+1])
                            circ.barrier()

                            X_basis_qubits = path[1+18*loop:1+18*(loop+1)]
                            for qubit in X_basis_qubits:
                                circ.h(qubit)
                            for i in range(len(X_basis_qubits)):
                                #circ.Measure(qreg[X_basis_qubits[i]], vars()[f'X{i}'][0])
                                circ.measure(X_basis_qubits[i], c_reg_dict[i+18*loop])
                    
                    
                elif mode == 'swap':
                    circ = self.gen_two_qubit_graphstate_circuit(path[0], path[1])
                    circ.name = f'{gap}-{path}'
                    #circ = self.circuit.copy(f'{gap}-{self.path_to_teleport}')
                    circ.barrier()
                    X_basis_qubits = path[1:-1]
                    swap_controls = X_basis_qubits
                    swap_targets = path[2:]
                    #move the information in the second qubit away using SWAP gates
                    for i in range(len(swap_controls)):
                        circ.swap(swap_controls[i], swap_targets[i])
            
                elif mode == 'dynamic':
                    circ = Circuit(name=f'{gap}-{path}')
                    qreg = circ.add_q_register('qreg', self.device_size)
                    for qubit in path:
                        circ.H(qreg[qubit])
                    for i in range(0,len(path)-1,2):
                        circ.CZ(qreg[path[i]], qreg[path[i+1]])
                    for i in range(1,len(path)-1,2):
                        circ.CZ(qreg[path[i]], qreg[path[i+1]])
                    circ.add_barrier(qreg)

            
                    #for i in range(gap):
                    #    name = f'X{i}'
                    #    vars()[name] = circ.add_c_register(name,1)
            
                    c_reg_dict = {}
                    for i in range(gap):
                        name = f'x{i}'
                        c_reg_dict[name] = circ.add_c_register(name,1)

            
                    X_basis_qubits = path[1:-1]
                    for qubit in X_basis_qubits:
                        circ.H(qreg[qubit])
                    for i in range(len(X_basis_qubits)):
                        #circ.Measure(qreg[X_basis_qubits[i]], vars()[f'X{i}'][0])
                        circ.Measure(qreg[X_basis_qubits[i]], c_reg_dict[f'x{i}'][0])
            
            
                    if gap > 3:
                        #XOR operation dynamic circuit
                        reg_z = circ.add_c_register('reg_z', 1)
                        reg_x = circ.add_c_register('reg_x', 1)
                
                        XOR_Z = c_reg_dict['x0']
                        XOR_X = c_reg_dict['x1']
                        for i in range(2, gap, 2):
                            XOR_Z ^= c_reg_dict[f'x{i}']
                        for i in range(3, gap, 2):
                            XOR_X ^= c_reg_dict[f'x{i}']
                        circ.add_classicalexpbox_register(XOR_Z, reg_z)
                        circ.add_classicalexpbox_register(XOR_X, reg_x)
                        if gap % 2 != 0:
                            circ.H(qreg[path[-1]]) # add this if gap is odd
                        circ.Z(qreg[path[-1]], condition=reg_eq(reg_z,1))
                        circ.X(qreg[path[-1]], condition=reg_eq(reg_x,1))

                    elif gap == 1:
                        circ.H(qreg[path[-1]]) # add this if gap is odd
                        circ.Z(qreg[path[-1]], condition=reg_eq(c_reg_dict['x0'],1))
                    elif gap == 2:
                        circ.Z(qreg[path[-1]], condition=reg_eq(c_reg_dict['x0'],1))
                        circ.X(qreg[path[-1]], condition=reg_eq(c_reg_dict['x1'],1))
                        #circ.X(qreg[self.path_to_teleport[-1]], condition=reg_eq(reg_x,1))
                    elif gap == 3:
                        reg_z = circ.add_c_register('reg_z', 1)
                        XOR_Z = c_reg_dict['x0']
                        for i in range(2, gap, 2):
                            XOR_Z ^= c_reg_dict[f'x{i}']
                        circ.add_classicalexpbox_register(XOR_Z, reg_z)
                        circ.H(qreg[path[-1]]) # add this if gap is odd
                        circ.Z(qreg[path[-1]], condition=reg_eq(reg_z,1))
                
                        circ.X(qreg[path[-1]], condition=reg_eq(c_reg_dict['x1'],1))
                
                elif mode == 'extended dynamic':
                    #First time scan
                    circ = Circuit(name=f'{gap}-{path}')
                    qreg = circ.add_q_register('qreg', self.device_size)
                    
                    c_reg_dict = {}
                    for i in range(1, gap+1):
                        name = f'x{i}'
                        c_reg_dict[name] = circ.add_c_register(name,1)
                    
                    for qubit in path[:20]:
                        circ.H(qreg[qubit])
                    for i in range(0,19,2):
                        circ.CZ(qreg[i], qreg[i+1])
                    for i in range(1,19,2):
                        circ.CZ(qreg[i], qreg[i+1])
                    circ.add_barrier(qreg)
                    
                    X_basis_qubits = path[1:19]
                    for qubit in X_basis_qubits:
                        circ.H(qreg[qubit])
                    for qubit in X_basis_qubits:
                        #circ.Measure(qreg[X_basis_qubits[i]], vars()[f'X{i}'][0])
                        circ.Measure(qreg[qubit], c_reg_dict[f'x{qubit}'][0])

                    
                    loops = ceil((len(path)-20)/18)
                    for loop in range(1, loops+1):
                        # Re-initialize mid qubits
                        for qubit in X_basis_qubits:
                            circ.Reset(qreg[qubit])
                            
                        if loop == loops:
                            #last scan
                            for qubit in path[2+18*loop:]:
                                circ.H(qreg[qubit])
                            for i in range(1+18*loop, gap+1, 2):
                                circ.CZ(qreg[path[i]], qreg[path[i+1]])
                            for i in range(2+18*loop, gap+1, 2):
                                circ.CZ(qreg[path[i]], qreg[path[i+1]])
                            circ.add_barrier(qreg)
                    
                            X_basis_qubits = path[1+18*loop:-1]
                            for qubit in X_basis_qubits:
                                circ.H(qreg[qubit])
                            for i in range(len(X_basis_qubits)):
                                #circ.Measure(qreg[X_basis_qubits[i]], vars()[f'X{i}'][0])
                                circ.Measure(qreg[X_basis_qubits[i]], c_reg_dict[f'x{i+1+18*loop}'][0])
                    
                        else:
                            #next scan
                            for qubit in path[2+18*loop: 2+18*(loop+1)]:
                                circ.H(qreg[qubit])
                            for i in range(1+18*loop, 1+18*(loop+1), 2):
                                circ.CZ(qreg[path[i]], qreg[path[i+1]])
                            for i in range(2+18*loop, 2+18*(loop+1), 2):
                                circ.CZ(qreg[path[i]], qreg[path[i+1]])
                            circ.add_barrier(qreg)
                    
                            X_basis_qubits = path[1+18*loop:1+18*(loop+1)]
                            for qubit in X_basis_qubits:
                                circ.H(qreg[qubit])
                            for i in range(len(X_basis_qubits)):
                                #circ.Measure(qreg[X_basis_qubits[i]], vars()[f'X{i}'][0])
                                circ.Measure(qreg[X_basis_qubits[i]], c_reg_dict[f'x{i+1+18*loop}'][0])
                    
                    reg_z = circ.add_c_register('reg_z', 1)
                    reg_x = circ.add_c_register('reg_x', 1)
                
                    XOR_Z = c_reg_dict['x1']
                    XOR_X = c_reg_dict['x2']
                    for i in range(3, gap+1, 2):
                        XOR_Z ^= c_reg_dict[f'x{i}']
                    for i in range(4, gap+1, 2):
                        XOR_X ^= c_reg_dict[f'x{i}']
                    circ.add_classicalexpbox_register(XOR_Z, reg_z)
                    circ.add_classicalexpbox_register(XOR_X, reg_x)
                    if gap % 2 != 0:
                        circ.H(qreg[path[-1]]) # add this if gap is odd
                    circ.Z(qreg[path[-1]], condition=reg_eq(reg_z,1))
                    circ.X(qreg[path[-1]], condition=reg_eq(reg_x,1))
                    
                BellState_circuits[gap][path] = circ
                
        self.teleported_BellState_circuit = BellState_circuits
        return circ
    
    
    def gen_teleported_qst_circuits(self):

        qst_circuits = {gap: {path:{} 
                              for path in path_dict.keys()} 
                        for gap, path_dict in self.teleported_BellState_circuit.items()}
        name_list = []
        
        for gap, path_dict in qst_circuits.items():
            for circ_path in path_dict.keys():
                targets = np.array([circ_path[0], circ_path[-1]]).flatten()
                if self.mode == 'post select' or self.mode == 'extended post select' or self.mode == 'swap':
                    #change the measurement basis to desired pauli-operators
                    circxx = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-XX')
                    circxx.h(targets)
                    qst_circuits[gap][circ_path]['XX'] = circxx
                
                    circxy = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-XY')
                    circxy.sdg(targets[1])
                    circxy.h(targets)
                    qst_circuits[gap][circ_path]['XY'] = circxy
                
                    circxz = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-XZ')
                    circxz.h(targets[0])
                    qst_circuits[gap][circ_path]['XZ'] = circxz
                
                    circyx = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-YX')
                    circyx.sdg(targets[0])
                    circyx.h(targets)
                    qst_circuits[gap][circ_path]['YX'] = circyx
                
                    circyy = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-YY')
                    circyy.sdg(targets)
                    circyy.h(targets)
                    qst_circuits[gap][circ_path]['YY'] = circyy
                
                    circyz = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-YZ')
                    circyz.sdg(targets[0])
                    circyz.h(targets[0])
                    qst_circuits[gap][circ_path]['YZ'] = circyz
                
                    circzx = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-ZX')
                    circzx.h(targets[1])
                    qst_circuits[gap][circ_path]['ZX'] = circzx
                
                    circzy = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-ZY')
                    circzy.sdg(targets[1])
                    circzy.h(targets[1])
                    qst_circuits[gap][circ_path]['ZY'] = circzy
                
                    circzz = self.teleported_BellState_circuit[gap][circ_path].copy(f'{gap}-{circ_path}-ZZ')
                    qst_circuits[gap][circ_path]['ZZ'] = circzz
                    #another measurement circuit for QST (distinct from teleportation measurment)
                    for circ in qst_circuits[gap][circ_path].values():
                        name_list.append(circ.name)
                        cr3 = ClassicalRegister(2)
                        circ.add_register(cr3)
                        circ.measure(targets, cr3)
                        
        
                elif self.mode == 'dynamic' or self.mode == 'extended dynamic':
                    circxx = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circxx.name = f'{gap}-{circ_path}-XX'
                    qreg = circxx.get_q_register('qreg')
                    for targ in targets:
                        circxx.H(qreg[targ])
                    qst_circuits[gap][circ_path]['XX'] = circxx
            
                    circxy = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circxy.name = f'{gap}-{circ_path}-XY'
                    qreg = circxy.get_q_register('qreg')
                    circxy.Sdg(qreg[targets[1]])
                    for targ in targets:
                        circxy.H(qreg[targ])
                    qst_circuits[gap][circ_path]['XY'] = circxy

                    circxz = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circxz.name = f'{gap}-{circ_path}-XZ'
                    qreg = circxz.get_q_register('qreg')
                    circxz.H(qreg[targets[0]])
                    qst_circuits[gap][circ_path]['XZ'] = circxz
            
                    circyx = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circyx.name = f'{gap}-{circ_path}-YX'
                    qreg = circyx.get_q_register('qreg')
                    circyx.Sdg(qreg[targets[0]])
                    for targ in targets:
                        circyx.H(qreg[targ])
                    qst_circuits[gap][circ_path]['YX'] = circyx
            
                    circyy = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circyy.name = f'{gap}-{circ_path}-YY'
                    qreg = circyy.get_q_register('qreg')
                    for targ in targets:
                        circyy.Sdg(qreg[targ])
                        circyy.H(qreg[targ])
                    qst_circuits[gap][circ_path]['YY'] = circyy
            
                    circyz = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circyz.name = f'{gap}-{circ_path}-YZ'
                    qreg = circyz.get_q_register('qreg')
                    circyz.Sdg(qreg[targets[0]])
                    circyz.H(qreg[targets[0]])
                    qst_circuits[gap][circ_path]['YZ'] = circyz
            
                    circzx = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circzx.name = f'{gap}-{circ_path}-ZX'
                    qreg = circzx.get_q_register('qreg')
                    circzx.H(qreg[targets[1]])
                    qst_circuits[gap][circ_path]['ZX'] = circzx
            
                    circzy = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circzy.name = f'{gap}-{circ_path}-ZY'
                    qreg = circzy.get_q_register('qreg')
                    circzy.Sdg(qreg[targets[1]])
                    circzy.H(qreg[targets[1]])
                    qst_circuits[gap][circ_path]['ZY'] = circzy
            
                    circzz = self.teleported_BellState_circuit[gap][circ_path].copy()
                    circzz.name = f'{gap}-{circ_path}-ZZ'
                    qreg = circzz.get_q_register('qreg')
                    qst_circuits[gap][circ_path]['ZZ'] = circzz
            
            
            
                    #another measurement circuit for QST (distinct from teleportation measurment)
                    for circ in qst_circuits[gap][circ_path].values():
                        name_list.append(circ.name)
                        cr3 = circ.add_c_register('cr3', 2)
                        for i in range(len(targets)):
                            circ.Measure(qreg[targets[i]], cr3[i])
                        #circ.measure(targets, targets)

        self.teleported_BellState_circuits_qst = qst_circuits
        self.name_list = name_list
        return qst_circuits
    
    def run_teleported_qst_circuits(self, shots=100, qrem=False):
        
        self.shots = shots
        self.qrem = qrem
        
        now = datetime.now()
        time_str = ('%02d-%02d-%02d-%02d-%02d-%02d'%(now.year, now.month, now.day, now.hour, now.minute,now.second))
        print(time_str)
        
        if qrem is True:
            qrem_circuits = self.gen_qrem_circuits()
        qrem_circuits_tk = []
        for circ in qrem_circuits:
            circ_tk = qiskit_to_tk(circ)
            circ_tk_compiled = self.backend.get_compiled_circuit(circ_tk, optimisation_level=2)
            qrem_circuits_tk.append(circ_tk_compiled)
            
        total_qrem_cost = 0
        for circ in qrem_circuits_tk:
            cost = self.backend.cost(circ, n_shots=shots, syntax_checker="H1-1SC")
            total_qrem_cost += cost
        print(f'The total cost of QREM job is {total_qrem_cost} HQCs')
        inp=input('Continue? (y/n): ')
        if inp=='y':
            handle_0 = self.backend.process_circuit(qrem_circuits_tk[0], n_shots=shots)
            print(f'job handle QREM 0: {handle_0}')
            if self.mode == 'post select':
                filename = f'quantinuum results/post select/handle QREM 0 {time_str}.pkl'
            elif self.mode == 'extended post select':
                filename = f'quantinuum results/extended post select/handle QREM 0 {time_str}.pkl'
            elif self.mode == 'dynamic':
                filename = f'quantinuum results/dynamic/handle QREM 0 {time_str}.pkl'
            elif self.mode == 'extended dynamic':
                filename = f'quantinuum results/extended dynamic/handle QREM 0 {time_str}.pkl'
            elif self.mode == 'swap':
                filename = f'quantinuum results/swap/handle QREM 0 {time_str}.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(str(handle_0), file, protocol=pickle.HIGHEST_PROTOCOL)
            #dump(str(handle_0),'quantinuum results/handle QREM 0.dat', 'wb')
            
            handle_1 = self.backend.process_circuit(qrem_circuits_tk[1], n_shots=shots)
            print(f'job handle QREM 1: {handle_1}')
            if self.mode == 'post select':
                filename = f'quantinuum results/post select/handle QREM 1 {time_str}.pkl'
            elif self.mode == 'extended post select':
                filename = f'quantinuum results/extended post select/handle QREM 1 {time_str}.pkl'
            elif self.mode == 'dynamic':
                filename = f'quantinuum results/dynamic/handle QREM 1 {time_str}.pkl'
            elif self.mode == 'extended dynamic':
                filename = f'quantinuum results/extended dynamic/handle QREM 1 {time_str}.pkl'
            elif self.mode == 'swap':
                filename = f'quantinuum results/swap/handle QREM 1 {time_str}.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(str(handle_1), file, protocol=pickle.HIGHEST_PROTOCOL)
            #dump(str(handle_1),'quantinuum results/handle QREM 1.dat', 'wb')


        #circ_list = []
        #for circ in self.teleported_BellState_circuits_qst.values():
        #    circ_list.append(circuit)
        circ_list_tk = []
        if self.mode == 'dynamic' or self.mode == 'extended dynamic':
            for path_dict in self.teleported_BellState_circuits_qst.values():
                for basis_dict in path_dict.values():
                    for circ in basis_dict.values():
                        circ_tk_compiled = self.backend.get_compiled_circuit(circ, optimisation_level=2)
                        circ_list_tk.append(circ_tk_compiled)
        else:
            for path_dict in self.teleported_BellState_circuits_qst.values():
                for basis_dict in path_dict.values():
                    for circ in basis_dict.values():
                        circ_tk = qiskit_to_tk(circ)
                        circ_tk_compiled = self.backend.get_compiled_circuit(circ_tk, optimisation_level=2)
                        circ_list_tk.append(circ_tk_compiled)
        
        #total_cost = 0
        #for i in range(len(circ_list_tk)):
        #    cost = self.backend.cost(circ_list_tk[i], n_shots=shots, syntax_checker="H1-1SC")
        #    total_cost += cost
        #print(f'The total cost of QST job is {total_cost} HQCs')
        
        inp=input('QST job Continue? (y/n): ')
        if inp=='y':
            for i in range(len(circ_list_tk)):
                handle = self.backend.process_circuit(circ_list_tk[i], n_shots=shots)#postprocess=True, simplify_initial=True, noisy_simulation=True
                print(f'job handle {self.name_list[i]}: {handle}')
                if self.mode == 'post select':
                    filename = f'quantinuum results/post select/handle {self.name_list[i]} {time_str}.pkl'
                elif self.mode == 'extended post select':
                    filename = f"quantinuum results/extended post select/handle {self.name_list[i].split('-')[0]}-{self.name_list[i].split('-')[2]} {time_str}.pkl"
                elif self.mode == 'dynamic':
                    filename = f'quantinuum results/dynamic/handle {self.name_list[i]} {time_str}.pkl'
                elif self.mode == 'extended dynamic':
                    filename = f"quantinuum results/extended dynamic/handle {self.name_list[i].split('-')[0]}-{self.name_list[i].split('-')[2]} {time_str}.pkl"
                elif self.mode == 'swap':
                    filename = f'quantinuum results/swap/handle {self.name_list[i]} {time_str}.pkl'
                with open(filename, 'wb') as file:
                    pickle.dump(str(handle), file, protocol=pickle.HIGHEST_PROTOCOL)
                #dump(str(handle),f'quantinuum results/handle {self.name_list[i]}.dat', 'wb')
    
    
    def save_counts_from_handles(self, mode='post select', time_str='', qrem_time_str=''):
        self.mode = mode
        if self.name_list is None:
            self.gen_teleported_BellState_circuit(mode=mode)
            self.gen_teleported_qst_circuits()
        
        qrem_counts = []
        for i in range(2):
            if self.mode == 'post select':
                filename = f'quantinuum results/post select/handle QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'extended post select':
                filename = f'quantinuum results/extended post select/handle QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'dynamic':
                filename = f'quantinuum results/dynamic/handle QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'extended dynamic':
                filename = f'quantinuum results/extended dynamic/handle QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'swap':
                filename = f'quantinuum results/swap/handle QREM {i} {qrem_time_str}.pkl'
                
            with open(filename, 'rb') as file:
                handle = pickle.load(file)
            job = ResultHandle.from_str(handle)
            while not self.backend.circuit_status(job).status.name=='COMPLETED':
                sleep(10)
            result=self.backend.get_result(job)
            counts=result.get_counts()
            print(counts)
            qrem_counts.append(qrem_counts)
            
            if self.mode == 'post select':
                filename = f'quantinuum results/post select/counts QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'extended post select':
                filename = f'quantinuum results/extended post select/counts QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'dynamic':
                filename = f'quantinuum results/dynamic/counts QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'extended dynamic':
                filename = f'quantinuum results/extended dynamic/counts QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'swap':
                filename = f'quantinuum results/swap/counts QREM {i} {qrem_time_str}.pkl'
                
            with open(filename, 'wb') as file:
                pickle.dump(counts, file, protocol=pickle.HIGHEST_PROTOCOL)
                
                
        for name in self.name_list:
            if self.mode == 'post select':
                filename = f'quantinuum results/post select/handle {name} {time_str}.pkl'
            elif self.mode == 'extended post select':
                filename = f"quantinuum results/extended post select/handle {name.split('-')[0]}-{name.split('-')[2]} {time_str}.pkl"
            elif self.mode == 'dynamic':
                filename = f'quantinuum results/dynamic/handle {name} {time_str}.pkl'
            elif self.mode == 'extended dynamic':
                filename = f"quantinuum results/extended dynamic/handle {name.split('-')[0]}-{name.split('-')[2]} {time_str}.pkl"
            elif self.mode == 'swap':
                filename = f'quantinuum results/swap/handle {name} {time_str}.pkl'
                
            with open(filename, 'rb') as file:
                handle = pickle.load(file)
            job = ResultHandle.from_str(handle)
            while not self.backend.circuit_status(job).status.name=='COMPLETED':
                sleep(10)
            result=self.backend.get_result(job)
            counts=result.get_counts()
            print(counts)
            
            
            if self.mode == 'post select':
                filename = f'quantinuum results/post select/counts {name} {time_str}.pkl'
                with open(filename, 'wb') as file:
                    pickle.dump(counts, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            elif self.mode == 'extended post select':
                filename = f"quantinuum results/extended post select/counts {name.split('-')[0]}-{name.split('-')[2]} {time_str}.pkl"
                
                indexs = [int(cbit[0][1:]) for cbit in result.to_dict()['bits']]
                indexs_sorted = sorted(indexs)
                order = []
                for idx in indexs_sorted:
                    order.append(indexs.index(idx))
                order[-1] += 1
            
                counts_new = {}
                for outcome, count in counts.items():
                    outcome_new = tuple([outcome[i] for i in order])
                    counts_new[outcome_new] = count
                print(counts_new)
            
                with open(filename, 'wb') as file:
                    pickle.dump(counts_new, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            elif self.mode == 'dynamic':
                filename = f'quantinuum results/dynamic/counts {name} {time_str}.pkl'
                with open(filename, 'wb') as file:
                    pickle.dump(counts, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            elif self.mode == 'extended dynamic':
                filename = f"quantinuum results/extended dynamic/counts {name.split('-')[0]}-{name.split('-')[2]} {time_str}.pkl"
                with open(filename, 'wb') as file:
                    pickle.dump(counts, file, protocol=pickle.HIGHEST_PROTOCOL)
                    
            elif self.mode == 'swap':
                filename = f'quantinuum results/swap/counts {name} {time_str}.pkl'
                with open(filename, 'wb') as file:
                    pickle.dump(counts, file, protocol=pickle.HIGHEST_PROTOCOL)
                
        #if self.shots is None:
        #    self.shots = result.metadata[0]['shots']
    
    def counts_from_saves(self, mode='post select', time_str='', qrem_time_str=''):
        self.mode = mode
        if self.name_list is None:
            self.gen_teleported_BellState_circuit(mode=mode)
            self.gen_teleported_qst_circuits()
        
        qrem_counts = []
        for i in range(2):
            if self.mode == 'post select':
                filename = f'quantinuum results/post select/counts QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'extended post select':
                filename = f'quantinuum results/extended post select/counts QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'dynamic':
                filename = f'quantinuum results/dynamic/counts QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'extended dynamic':
                filename = f'quantinuum results/extended dynamic/counts QREM {i} {qrem_time_str}.pkl'
            elif self.mode == 'swap':
                filename = f'quantinuum results/swap/counts QREM {i} {qrem_time_str}.pkl'
            with open(filename, 'rb') as file:
                counts = pickle.load(file)
            qrem_counts.append(counts)

        self.shots = 0
        for count in qrem_counts[0].values():
            self.shots += count
        
        M_list = [np.zeros((2, 2)) for i in range(self.device_size)]
        for jj, counts in enumerate(qrem_counts):
            for outcome, count in counts.items():
                bit_str = ''
                for digit in outcome:
                    bit_str += str(digit)
                for i, q in enumerate(bit_str):
                    ii = int(q)
                    M_list[i][ii, jj] += count

        # Normalise
        norm = 1/self.shots
        for M in M_list:
            M *= norm
        self.M_list = M_list
        
        counts_dict = copy.deepcopy(self.teleported_BellState_circuits_qst)
        pvecs_dict = copy.deepcopy(self.teleported_BellState_circuits_qst)
        
        for name in self.name_list:
            if self.mode == 'post select':
                filename = f'quantinuum results/post select/counts {name} {time_str}.pkl'
            elif self.mode == 'extended post select':
                filename = f"quantinuum results/extended post select/counts {name.split('-')[0]}-{name.split('-')[2]} {time_str}.pkl"
            elif self.mode == 'dynamic':
                filename = f'quantinuum results/dynamic/counts {name} {time_str}.pkl'
            elif self.mode == 'extended dynamic':
                filename = f"quantinuum results/extended dynamic/counts {name.split('-')[0]}-{name.split('-')[2]} {time_str}.pkl"
            elif self.mode == 'swap':
                filename = f'quantinuum results/swap/counts {name} {time_str}.pkl'
            with open(filename, 'rb') as file:
                counts = pickle.load(file)
            gap, path, basis = name.split('-')
            print(basis)
            gap = int(gap)
            path = literal_eval(path)
            if mode == 'dynamic' or mode == 'extended dynamic':
                new_counts = {}
                for outcome, count in counts.items():
                    new_outcome = outcome[:2]
                    if new_outcome in new_counts:
                        new_counts[new_outcome] += count
                    else:
                        new_counts[new_outcome] = count
                counts_dict[gap][path][basis] = new_counts
                
            else:
                counts_dict[gap][path][basis] = counts
                

            if mode == 'swap':
                pvecs_dict[gap][path][basis] = np.zeros(4)
                for outcome, count in counts.items():
                    bit_str = ''
                    for digit in outcome:
                        bit_str += str(digit)
                    idx = int(bit_str, 2)
                    pvecs_dict[gap][path][basis][idx] += count
                    
            elif mode == 'post select':
                pvecs_dict[gap][path][basis] = np.zeros(2**(gap+2))
                for outcome, count in counts.items():
                    bit_str = ''
                    for digit in outcome:
                        bit_str += str(digit)
                    idx = int(bit_str, 2)
                    pvecs_dict[gap][path][basis][idx] += count
                    
            elif mode == 'extended post select':
                pvecs_dict[gap][path][basis] = np.zeros(2)
            
            else:
                pvecs_dict[gap][path][basis] = np.zeros(4)
                for outcome, count in counts.items():
                    bit_str = ''
                    for digit in outcome:
                        bit_str += str(digit)
                    bit_str = bit_str[:2]
                    idx = int(bit_str, 2)
                    pvecs_dict[gap][path][basis][idx] += count
            pvecs_dict[gap][path][basis] /= self.shots
            
        return counts_dict, pvecs_dict
    
    
    def apply_qrem_teleported_counts(self, counts_dict, pvecs_dict, qrem='QREM', mitigate_qubits = [1,3,4,5]):
        
        if self.mode == 'post select':
            counts_dict_mit = copy.deepcopy(counts_dict)
            pvecs_dict_mit = copy.deepcopy(pvecs_dict)
            #counts_dict_mit = {basis: {} for basis in basis_list}
            #pvecs_dict_mit = {basis: np.zeros(self.nqubits) for basis in basis_list}
            
            for gap, path_dict in pvecs_dict.items():
                for path, basis_dict in path_dict.items():
                    measurements_order = path
                    if qrem == 'QREM':
                        M_inv = la.inv(self.calc_M_multi(measurements_order))
                    for basis, pvec in basis_dict.items():
                        #correct pvec according to mitigation mode
                        if qrem == 'QREM':
                            pvec_mit = np.matmul(M_inv, pvec)
                        #pvec_mit = find_closest_pvec(pvec_mit)
                        elif qrem =='reduced_QREM':
                            pvec_mit = self.apply_reduced_qrem_to_pvec(pvec, gap, measurements_order, mitigate_qubits=mitigate_qubits)
    
                        pvec_mit = quantinuum_teleportation.find_closest_pvec(pvec_mit)
                        pvec_mit /= np.sum(pvec_mit)
                        pvecs_dict_mit[gap][path][basis] = pvec_mit
                        for j, prob in enumerate(pvec_mit):
                            bit_str = bin(j)[2:].zfill(gap+2)
                            outcome_list = []
                            for bit in bit_str:
                                outcome_list.append(int(bit))
                            counts_dict_mit[gap][path][basis][tuple(outcome_list)] = prob*self.shots
                print(f'{gap} done')
        
        elif self.mode == 'extended post select':
            counts_dict_mit = copy.deepcopy(counts_dict)
            pvecs_dict_mit = copy.deepcopy(pvecs_dict)
            #counts_dict_mit = {basis: {} for basis in basis_list}
            #pvecs_dict_mit = {basis: np.zeros(self.nqubits) for basis in basis_list}
            
            for gap, path_dict in counts_dict.items():
                for path, basis_dict in path_dict.items():
                    measurements_order = path
                    with Pool(9) as p:
                        basis_list = list(basis_dict.keys())
                        results = p.map(quantinuum_teleportation.apply_reduced_qrem_to_counts_physical_single_input, [(counts, measurements_order, self.M_list) for counts in basis_dict.values()])
                        for i in range(len(results)):
                            scale_factor = sum(results[i].values())
                            for outcome in results[i].keys():
                                results[i][outcome] *= self.shots/scale_factor
                            print(len(results[i]))
                            basis = basis_list[i]
                            counts_dict_mit[gap][path][basis] = results[i]
                            
                    
                    
                    #for basis, counts in basis_dict.items():
                    #    #pvec_mit = find_closest_pvec(pvec_mit)
                    #    counts_mit = self.apply_reduced_qrem_to_counts(counts, gap, measurements_order)
                    #    counts_mit = quantinuum_teleportation.find_closest_counts(counts_mit)
                    #    scale_factor = sum(counts_mit.values())
                    #    for outcome in counts_mit.keys():
                    #        counts_mit[outcome] *= self.shots/scale_factor
                    #        
                    #    counts_dict_mit[gap][path][basis] = counts_mit
                        
                print(f'{gap} done')
                
                
        else:
            counts_dict_mit = copy.deepcopy(counts_dict)
            pvecs_dict_mit = copy.deepcopy(pvecs_dict)
            for gap, path_dict in pvecs_dict.items():
                for path, basis_dict in path_dict.items():
                    measurements_order = [path[0], path[-1]]
                    M_inv = la.inv(self.calc_M_multi(measurements_order))
                    for basis, pvec in basis_dict.items():
                        if qrem == 'QREM':
                            pvec_mit = np.matmul(M_inv, pvec)
                        elif qrem == 'reduced_QREM':
                            pvec_mit = self.apply_reduced_qrem_to_pvec(pvec, measurements_order, mitigate_qubits=mitigate_qubits)
                        #pvec_mit /= np.sum(pvec_mit)
                        pvec_mit = quantinuum_teleportation.find_closest_pvec(pvec_mit)
                        pvec_mit /= np.sum(pvec_mit)
                        pvecs_dict_mit[gap][path][basis] = pvec_mit
                        for j, prob in enumerate(pvec_mit):
                            bit_str = bin(j)[2:].zfill(2)
                            outcome_list = []
                            for bit in bit_str:
                                outcome_list.append(int(bit))
                            counts_dict_mit[gap][path][basis][tuple(outcome_list)] = prob*self.shots
        
        return counts_dict_mit, pvecs_dict_mit
    
    @staticmethod
    def apply_reduced_qrem_to_counts_physical_single_input(inputs):
        """Apply Reduced Quantum Readout Error Mitigation to zipped counts
        """
        (counts, measurements_order, M_list) = inputs
        #mitigate_qubits = measurements_order.copy()
        threshold = 0.00003#set minimum threshold, zero out the count if below it
        
        corrected_counts = copy.deepcopy(counts)
        #iterate over each qubit
        for idx in range(len(measurements_order)):
            #idx = measurements_order.index(q)
            calibration_M = la.inv(M_list[measurements_order[idx]])
            applied_names = set([])
            corrected_bit_strings = [k for k in corrected_counts.keys()]
            
            for bit_string in corrected_bit_strings:
                bit_string_true = ''
                for digit in bit_string:
                    bit_string_true += str(digit)
                bit_string_int = int(bit_string_true, 2)
                #bit_string = bin(bit_string_int)[2:].zfill(self.nqubits)
                bit_string_list = list(bit_string_true)
                bit_string_list[idx] = '_'
                #check if the bit-string (except digit q) is already been corrected
                name = "".join(bit_string_list)
                if name not in applied_names:
                    applied_names.add(name)
                    #check the digit is 0 or 1, then flip it
                    if (bit_string_int & (1 << idx)) != 0:
                        bit_string_list[idx] = '0'
                    else:
                        bit_string_list[idx] = '1'
                    bit_string_flip = tuple([int(bit) for bit in bit_string_list])
                    bit_string_int_flip = int("".join(bit_string_list), 2)
                            
                    reduced_pvec = np.zeros(2)
                    # if 0->1
                    if bit_string_int < bit_string_int_flip:
                        if bit_string in corrected_counts:
                            reduced_pvec[0] += corrected_counts[bit_string]
                        if bit_string_flip in corrected_counts:
                            reduced_pvec[1] += corrected_counts[bit_string_flip]
                        reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                        if abs(reduced_pvec_mit[0]) > threshold:
                            corrected_counts[bit_string] = reduced_pvec_mit[0]
                        #zero-out if below threshold
                        else:
                            corrected_counts[bit_string] = 0
                            del corrected_counts[bit_string]
                        if abs(reduced_pvec_mit[1]) > threshold:
                            corrected_counts[bit_string_flip] = reduced_pvec_mit[1]
                        #zero-out if below threshold
                        else:
                            corrected_counts[bit_string_flip] = 0
                            del corrected_counts[bit_string_flip]
                    # if 1->0
                    else:
                        if bit_string in corrected_counts:
                            reduced_pvec[1] += corrected_counts[bit_string]
                        if bit_string_flip in corrected_counts:
                            reduced_pvec[0] += corrected_counts[bit_string_flip]
                        reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                        if abs(reduced_pvec_mit[0]) > threshold:
                            corrected_counts[bit_string_flip] = reduced_pvec_mit[0]
                        #zero-out if below threshold
                        else:
                            corrected_counts[bit_string_flip] = 0
                            del corrected_counts[bit_string_flip]
                        if abs(reduced_pvec_mit[1]) > threshold:
                            corrected_counts[bit_string] = reduced_pvec_mit[1]
                        #zero-out if below threshold
                        else:
                            corrected_counts[bit_string] = 0
                            del corrected_counts[bit_string]

        print(f'{len(corrected_counts)}')
        
        corrected_counts = quantinuum_teleportation.find_closest_counts(corrected_counts)
        return corrected_counts
    
    

    def apply_reduced_qrem_to_counts(self, counts, gap, measurements_order):
        """Apply Reduced Quantum Readout Error Mitigation to zipped counts

        Args:
            counts_list (list): list of counts dictionaries
            mitigate_qubits (list, optional): list of qubits indicies to mitigate. Defaults to [1,3,4,5,6].
            threshold (float, optional): minimum value of a bit-string below it will be zero out. Defaults to 0.1.

        Returns:
            list: error mitigated counts_list
        """
        #mitigate_qubits = measurements_order.copy()
        threshold = 0.00003#set minimum threshold, zero out the count if below it
        
        corrected_counts = copy.deepcopy(counts)
        #iterate over each qubit
        for idx in range(len(measurements_order)):
            #idx = measurements_order.index(q)
            calibration_M = la.inv(self.M_list[measurements_order[idx]])
            applied_names = set([])
            corrected_bit_strings = [k for k in corrected_counts.keys()]
            
            for bit_string in corrected_bit_strings:
                bit_string_true = ''
                for digit in bit_string:
                    bit_string_true += str(digit)
                bit_string_int = int(bit_string_true, 2)
                #bit_string = bin(bit_string_int)[2:].zfill(self.nqubits)
                bit_string_list = list(bit_string_true)
                bit_string_list[idx] = '_'
                #check if the bit-string (except digit q) is already been corrected
                name = "".join(bit_string_list)
                if name not in applied_names:
                    applied_names.add(name)
                    #check the digit is 0 or 1, then flip it
                    if (bit_string_int & (1 << idx)) != 0:
                        bit_string_list[idx] = '0'
                    else:
                        bit_string_list[idx] = '1'
                    bit_string_flip = tuple([int(bit) for bit in bit_string_list])
                    bit_string_int_flip = int("".join(bit_string_list), 2)
                            
                    reduced_pvec = np.zeros(2)
                    # if 0->1
                    if bit_string_int < bit_string_int_flip:
                        if bit_string in corrected_counts:
                            reduced_pvec[0] += corrected_counts[bit_string]
                        if bit_string_flip in corrected_counts:
                            reduced_pvec[1] += corrected_counts[bit_string_flip]
                        reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                        if abs(reduced_pvec_mit[0]) > threshold:
                            corrected_counts[bit_string] = reduced_pvec_mit[0]
                        #zero-out if below threshold
                        else:
                            corrected_counts[bit_string] = 0
                            del corrected_counts[bit_string]
                        if abs(reduced_pvec_mit[1]) > threshold:
                            corrected_counts[bit_string_flip] = reduced_pvec_mit[1]
                        #zero-out if below threshold
                        else:
                            corrected_counts[bit_string_flip] = 0
                            del corrected_counts[bit_string_flip]
                    # if 1->0
                    else:
                        if bit_string in corrected_counts:
                            reduced_pvec[1] += corrected_counts[bit_string]
                        if bit_string_flip in corrected_counts:
                            reduced_pvec[0] += corrected_counts[bit_string_flip]
                        reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                        if abs(reduced_pvec_mit[0]) > threshold:
                            corrected_counts[bit_string_flip] = reduced_pvec_mit[0]
                        #zero-out if below threshold
                        else:
                            corrected_counts[bit_string_flip] = 0
                            del corrected_counts[bit_string_flip]
                        if abs(reduced_pvec_mit[1]) > threshold:
                            corrected_counts[bit_string] = reduced_pvec_mit[1]
                        #zero-out if below threshold
                        else:
                            corrected_counts[bit_string] = 0
                            del corrected_counts[bit_string]

        print(f'{len(corrected_counts)}')
                
        return corrected_counts
    
    
    
    def apply_reduced_qrem_to_pvec(self, pvec, gap, measurements_order, mitigate_qubits = [1,3,4,5]):
        """Apply QREM qubit-wisely on selected qubits

        Args:
            pvec (numpy array): probability vector
            gap (int): number of hops in teleportation
            measurements_order (_type_): the indicies of qubits in the same order of their measurement order
            mitigate_qubits (list, optional): list of qubits to mitigate. Defaults to [1,3,4,5].

        Returns:
            numpy array: corrected probability vector
        """
        mitigate_qubits_true = list(set(mitigate_qubits).intersection(measurements_order))
        threshold = 0.0001/self.shots#set minimum threshold, zero out the count if below it
        #threshold = 0
        #iterate over each qubit
        for q in mitigate_qubits_true:
            idx = measurements_order.index(q)
            calibration_M = la.inv(self.M_list[q])
            applied_names = set([])
            bit_string_ints = np.flatnonzero(pvec.copy())#non-zero indicies of the pvec elements
            for num in bit_string_ints:
                if self.mode == 'post select':
                    bit_string = bin(num)[2:].zfill(gap+2)
                else:
                    bit_string = bin(num)[2:].zfill(2)
                bit_string_list = list(bit_string)
                bit_string_list[idx] = '_'
                name = "".join(bit_string_list)
                #check if the bit-string (execpt digit q) is already been corrected
                if name not in applied_names:
                    applied_names.add(name)
                    #check the digit is 0 or 1, then flip it
                    if (num & (1 << idx)) != 0:
                        bit_string_list[idx] = '0'
                    else:
                        bit_string_list[idx] = '1'
                    bit_string_flip = "".join(bit_string_list)
                    num_flip = int(bit_string_flip, 2)
                    
                    reduced_pvec = np.zeros(2)
                    # if 0->1
                    if num < num_flip:
                        reduced_pvec[0] += pvec[num]
                        reduced_pvec[1] += pvec[num_flip]
                        #calibrate the two elemnts with bit-string only differ at this digit idx
                        reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                        if abs(reduced_pvec_mit[0]) > threshold:
                            pvec[num] = reduced_pvec_mit[0]
                        else:
                            pvec[num] = 0
                        if abs(reduced_pvec_mit[1]) > threshold:
                            pvec[num_flip] = reduced_pvec_mit[1]
                        else:
                            pvec[num_flip] = 0
                    #if 1->0          
                    else:
                        reduced_pvec[1] += pvec[num]
                        reduced_pvec[0] += pvec[num_flip]
                        reduced_pvec_mit = np.matmul(calibration_M, reduced_pvec)
                                
                        if abs(reduced_pvec_mit[0]) > threshold:
                            pvec[num_flip] = reduced_pvec_mit[0]
                        else:
                            pvec[num_flip] = 0
                        if abs(reduced_pvec_mit[1]) > threshold:
                            pvec[num] = reduced_pvec_mit[1]
                        else:
                            pvec[num] = 0
        #pvec = find_closest_pvec(pvec)
        print(f'{np.count_nonzero(pvec)}')
        return pvec
    
    def bin_teleported_pvecs(self, counts_dict):    
        """bin pvces list from gap-pair-QST basis-bit string(X-measurements+pair measurements) structure to gap-pair-BellState(X 
        measurements)-QST basis structure

        Args:
            qst_counts_list (list): list of counts dictionaries
            pvecs_list (list): list of pvecs dictionaries

        Returns:
            list: binned list of pvecs dictionaries
        """
        # bin the pvecs into corresponding Bell State
        BellState_names = ['BS_1', 'BS_2', 'BS_3', 'BS_4']
        pvecs_binned = {}
        for gap, path_dict in counts_dict.items():
            pvecs_binned[gap] = {path: {bellstate:{basis: np.zeros(4) for basis in basis_list} 
                                        for bellstate in BellState_names} 
                                 for path in path_dict.keys()}
            
            for path, basis_dict in path_dict.items():
                for basis, counts in basis_dict.items():
                    for outcome, count in counts.items():
                        bit_str = ''
                        for digit in outcome:
                            bit_str += str(digit)
                        #print(bit_str)
                        pair_str = bit_str[-2:]
                        idx = int(pair_str, 2)
                        X_str = bit_str[:-2]
                        BellState_name = quantinuum_teleportation.bit_str_to_BellState_simp(X_str)
                        pvecs_binned[gap][path][BellState_name][basis][idx] += count
            
                    for bellstate in BellState_names:
                        pvec = pvecs_binned[gap][path][bellstate][basis]
                        if pvec.sum() != 0:
                            norm = 1/pvec.sum()
                        else:
                            norm = 1
                        pvecs_binned[gap][path][bellstate][basis] = pvec*norm
        
        return pvecs_binned
    
    def recon_teleported_density_mats(self, mode = 'post select', time_str='', qrem_time_str='', apply_mit='QREM', mitigate_qubits = [1,3,4,5],
                                      bootstrap=False, resample_num=10):
        """
        Reconstruct density matrices of the teleported two-qubit graph state
        """
        qst_counts, pvecs = self.counts_from_saves(mode=mode, time_str=time_str, qrem_time_str=qrem_time_str)

        #cateogrise variants of teleported state

        #Apply QREM
        if apply_mit == 'QREM':
            qst_counts, pvecs = self.apply_qrem_teleported_counts(qst_counts, pvecs,
                                                                    qrem = 'QREM')
            print('QREM done')
        elif apply_mit == 'reduced_QREM':
            qst_counts, pvecs = self.apply_qrem_teleported_counts(qst_counts, pvecs, 
                                                                  qrem = 'reduced_QREM',
                                                                  mitigate_qubits = mitigate_qubits)
            print('reduced_QREM done')

        if bootstrap is True:
            pvecs_new = {gap: {path: {basis: [] 
                                      for basis in basis_dict.keys()} 
                               for path, basis_dict in path_dict.items()} 
                         for gap, path_dict in pvecs.items()}
            
            for gap, path_dict in qst_counts.items():
                for path, basis_dict in path_dict.items():
                    for basis, counts in basis_dict.items():
                        outcome_tuple, count = zip(*counts.items())
                        new_samples = random.choices(outcome_tuple, count, k=self.shots*resample_num)
                        new_samples_dict_list = [Counter(new_samples[i*self.shots: (i+1)*self.shots]) for i in range(resample_num)]
                        new_pvecs_list = []
                        
                        for i in range(resample_num):
                            pvec = np.zeros(4)
                            for outcome, count in new_samples_dict_list[i].items():
                                bit_str = ''
                                for digit in outcome:
                                    bit_str += str(digit)
                                idx = int(bit_str, 2)
                                pvec[idx] += count
                            pvec /= self.shots
                            new_pvecs_list.append(pvec)
                        
                        pvecs_new[gap][path][basis] = new_pvecs_list
        
        
        #calculate density matrices
        if mode == 'post select' or mode == 'extended post select':
            #categorise
            pvecs_binned = self.bin_teleported_pvecs(qst_counts)
            print('Categorisation done')
            rho_dict = {}
            for gap, path_dict in pvecs_binned.items():
                rho_dict[gap] = {path: {} for path in path_dict.keys()}
                for path, BS_dict in path_dict.items():
                    for BellState, basis_dict in BS_dict.items():
                        rho_dict[gap][path][BellState] = quantinuum_teleportation.calc_rho(basis_dict)
            return rho_dict
        
        else:
            if bootstrap is True:
                rho_dict = {}
                for gap, path_dict in pvecs_new.items():
                    rho_dict[gap] = {}
                    for path, basis_dict in path_dict.items():
                        rho_dict[gap][path] = []
                        for i in range(resample_num):
                            basis_dict_i = {}
                            for basis, pvecs_list in basis_dict.items():
                                basis_dict_i[basis] = pvecs_list[i]
                            rho_dict[gap][path].append(quantinuum_teleportation.calc_rho(basis_dict_i))
            
            else:
                rho_dict = {}
                for gap, path_dict in pvecs.items():
                    rho_dict[gap] = {}
                    for path, basis_dict in path_dict.items():
                        rho_dict[gap][path] = quantinuum_teleportation.calc_rho(basis_dict)
            return rho_dict
    
    @staticmethod
    def bit_str_to_BellState_simp(bit_str):
        if len(bit_str) == 1:
            if bit_str == '0':
                return 'BS_1'
            else:
                return 'BS_2'
        even = int(bit_str[::2],2)
        odd = int(bit_str[1::2],2)
        if quantinuum_teleportation.hamming_weight(even) % 2 == 0:
            if quantinuum_teleportation.hamming_weight(odd) % 2 == 0:
                return 'BS_1'
            else:
                return 'BS_3'
        else:
            if quantinuum_teleportation.hamming_weight(odd) % 2 == 0:
                return 'BS_2'
            else:
                return 'BS_4'
    
    @staticmethod
    def calc_rho(pvecs):
        """Calculate density matrix from probability vectors

        Args:
            pvecs (dictionary): key are the basis, values are pvec

        Returns:
            numpy2d array: list of density matrices
        """
        rho = np.zeros([4, 4], dtype=complex)

        # First calculate the Stokes parameters
        s_dict = {basis: 0. for basis in ext_basis_list}
        s_dict['II'] = 1.  # S for 'II' always equals 1

        # Calculate s in each experimental basis
        for basis, pvec in pvecs.items():
            # s for basis not containing I
            s_dict[basis] = pvec[0] - pvec[1] - pvec[2] + pvec[3]
            # s for basis 'IX' and 'XI'
            s_dict['I' + basis[1]] += (pvec[0] - pvec[1] + pvec[2] - pvec[3])/3 #+ or - is only decided by whether 2nd qubit is measured
                                                                                #to be |0> or |1> because only identity is applied to 1st
                                                                                #qubit so its result is not important
            s_dict[basis[0] + 'I'] += (pvec[0] + pvec[1] - pvec[2] - pvec[3])/3
            
        # Weighted sum of basis matrices
        for basis, s in s_dict.items():
            rho += 0.25*s*pauli_n(basis)

        # Convert raw density matrix into closest physical density matrix using
        # Smolin's algorithm (2011)
        #print(rho)
        rho = quantinuum_teleportation.find_closest_physical(rho)

        return rho
    
    def gen_qrem_circuits(self):
        """Generate QREM circuits

        Returns:
            list: list of two circuits from QREM
        """
        circ0 = QuantumCircuit(self.device_size, self.device_size, name='qrem0')
        circ0.measure(sorted(self.path_to_teleport), sorted(self.path_to_teleport))

        circ1 = QuantumCircuit(self.device_size, self.device_size, name='qrem1')
        circ1.x(sorted(self.path_to_teleport))
        circ1.measure(sorted(self.path_to_teleport), sorted(self.path_to_teleport))

        self.qrem_circuits = [circ0, circ1]

        return [circ0, circ1]
    
    def calc_M_multi(self, qubits):
        """Compose n-qubit calibration matrix by tensoring single-qubit matrices

        Args:
            qubits (list): list of qubits indecies

        Returns:
            numpy2d array: calibration matrix on those qubits
        """
        M = self.M_list[qubits[0]]
        for q in qubits[1:]:
            M_new = np.kron(M, self.M_list[q])
            M = M_new

        return M
    
    @staticmethod
    def hamming_weight(n):
        """return the Hamming weight of an integer (check how many '1's for an integer after converted to binary)

        Args:
            n (int): any integer

        Returns:
            int: number of ones of the integer n after converted to binary
        """
        c = 0
        while n:
            c += POPCOUNT_TABLE16[n & 0xffff]
            n >>= 16
        return c
    
    @staticmethod
    def find_closest_pvec(pvec_array):
        """Find closest probability vector
        [C Michelot - Journal of Optimization Theory and Applications, 1986]
        works for both prob. vectors and shot count vectors
        Args:
            pvec_array (numpy array of dimension 2^N): probability vector, in principle non-physical
            (may have negative values)

        Returns:
            numpy array of same size as input: corrected physical probability vector
        """
        # placeholder vector
        v = lil_matrix(pvec_array)
        q = Q.PriorityQueue()

        cv = v.tocoo()
        #cv = vector_as_sparse_matrix.tocoo()
        count = 0

        for i,j,k in zip(cv.row, cv.col, cv.data):
            q.put((k, (i,j)))
            count += 1
        # q now stores eigenvalues in increasing order with corresponding index as value.
        # note that we don't need to have count = vector_as_sparse_matrix.shape[0] because count is not important for
        # negative values (since (item[0] + a / count) < 0 is always true) and after processing the negative and zero 
        # values count would be decremented by #(zero values) anyway. So we can ignore zero values when calculating count

        a = 0
        continue_zeroing = True
        while not q.empty():
            item = q.get()
            if continue_zeroing:
                if (count > 0) and ((item[0] + a / count) < 0):
                    v[item[1][0], item[1][1]] = 0
                    a += item[0] # add up all the pvec elements xi that has xi+a/count < 0
                    count -= 1 #eventually count will be reduced to number of elements left in the pvec that has xi +a/count > 0
                else:
                    continue_zeroing = False #once the judgement fails, it never get back to True
            if not continue_zeroing:
                v[item[1][0], item[1][1]] = item[0] + a / count #update the rest of the pvec items by distributing the taking away the -ves 
                #that has been set to 0


        return v.toarray()[0]
    
    @staticmethod
    def find_closest_counts(counts):

        #scale_factor = shots/sum(counts.values())
        #counts.update((k,v*scale_factor/shots) for k,v in counts.items())
        
        counts_new = copy.deepcopy(counts)
        q = Q.PriorityQueue()
    
        count = 0
        for bit_string, v in counts.items():
            q.put((v, bit_string))
            count += 1
    
        a = 0
        continue_zeroing = True
        while not q.empty():
            item = q.get()
            if continue_zeroing:
                if (count > 0) and ((item[0] + a/count) < 0):
                    counts_new[item[1]] = 0
                    a += item[0]
                    count -= 1
                else:
                    continue_zeroing = False
            if not continue_zeroing:
                counts_new[item[1]] = item[0] + a/count
    
        #counts.update((k, v*shots) for k,v in counts.items())
        return counts_new

    @staticmethod
    def find_closest_physical(rho):
        """Algorithm to find closest physical density matrix from Smolin et al.

        Args:
            rho (numpy2d array): (unphysical) density matrix

        Returns:
            numpy2d array: physical density matrix
        """
        rho = rho/rho.trace()
        rho_physical = np.zeros(rho.shape, dtype=complex)
        # Step 1: Calculate eigenvalues and eigenvectors
        eigval, eigvec = la.eig(rho)
        # Rearranging eigenvalues from largest to smallest
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        eigval_new = np.zeros(len(eigval), dtype=complex)

        # Step 2: Let i = number of eigenvalues and set accumulator a = 0
        i = len(eigval)
        a = 0

        while (eigval[i-1] + a/i) < 0:
            a += eigval[i-1]
            i -= 1

        # Step 4: Increment eigenvalue[j] by a/i for all j <= i
        # Note since eigval_new is initialized to be all 0 so for those j>i they are already set to 0
        for j in range(i):
            eigval_new[j] = eigval[j] + a/i
            # Step 5 Construct new density matrix
            rho_physical += eigval_new[j] * \
                np.outer(eigvec[:, j], eigvec[:, j].conjugate()) #rho = Sum(lambdai*|lambdai><lambdai|)

        return rho_physical
    
def calc_teleported_negativities(rho_dict, output='all', mode = 'post select',
                                 witness = 'negativity', bootstrap = False):
    
    ent_all = copy.deepcopy(rho_dict)# Negativities for each bin per experiment
    if bootstrap is True:
        ent_mean = {gap: {path: [] for path in path_dict.keys()} 
                    for gap, path_dict in rho_dict.items()} # Mean negativity between bins per experiment
    else:
        ent_mean = {gap: {path: 0 
                          for path in path_dict.keys()} 
                    for gap, path_dict in rho_dict.items()} # Mean negativity between bins per experiment
    ent_max = copy.deepcopy(ent_mean) # Max negativity between bins per experiment
    ent_min = copy.deepcopy(ent_mean)
    
    if witness == 'negativity':
        if mode == 'post select' or mode == 'extended post select':
            for gap, path_dict in rho_dict.items():
                for path, bellstate_dict in path_dict.items():
                    ent_sum = 0
                    ent_list = []
                    for bellstate, rho in bellstate_dict.items():
                        ent = calc_n(rho)
                        ent_all[gap][path][bellstate] = ent #n_all stores each BS outcome
                        ent_list.append(ent)
                        ent_sum += ent
                
                    ent_mean[gap][path] = np.mean(ent_list)
                    ent_max[gap][path] = max(ent_list)
                    ent_min[gap][path] = min(ent_list)
                
        else:
            if bootstrap is True:
                for gap, path_dict in rho_dict.items():
                    for path, rho_list in path_dict.items():
                        for rho in rho_list:
                            ent = calc_n(rho)
                            ent_mean[gap][path].append(ent)
            else:
                for gap, path_dict in rho_dict.items():
                    for path, rho in path_dict.items():
                        ent = calc_n(rho)
                        ent_mean[gap][path] = ent
    
    elif witness == 'fidelity':
        if mode == 'post select' or mode == 'extended post select':
            for gap, path_dict in rho_dict.items():
                for path, bellstate_dict in path_dict.items():
                    ent_sum = 0
                    ent_list = []
                    for bellstate, rho in bellstate_dict.items():
                        ent = calc_f(rho, gap, bellstate)
                        ent_all[gap][path][bellstate] = ent #n_all stores each BS outcome
                        ent_list.append(ent)
                        ent_sum += ent
                
                    ent_mean[gap][path] = np.mean(ent_list)
                    ent_max[gap][path] = max(ent_list)
                    ent_min[gap][path] = min(ent_list)
                
        else:
            if bootstrap is True:
                for gap, path_dict in rho_dict.items():
                    for path, rho_list in path_dict.items():
                        for rho in rho_list:
                            ent = calc_f(rho)
                            ent_mean[gap][path].append(ent)
            else:
                for gap, path_dict in rho_dict.items():
                    for path, rho in path_dict.items():
                        ent = calc_f(rho)
                        ent_mean[gap][path] = ent
            
    elif witness == 'entropy':
        if mode == 'post select' or mode == 'extended post select':
            for gap, path_dict in rho_dict.items():
                for path, bellstate_dict in path_dict.items():
                    ent_sum = 0
                    ent_list = []
                    for bellstate, rho in bellstate_dict.items():
                        ent = calc_s(rho)
                        ent_all[gap][path][bellstate] = ent #n_all stores each BS outcome
                        ent_list.append(ent)
                        ent_sum += ent
                
                    ent_mean[gap][path] = np.mean(ent_list)
                    ent_max[gap][path] = max(ent_list)
                    ent_min[gap][path] = min(ent_list)
                
        else:
            if bootstrap is True:
                for gap, path_dict in rho_dict.items():
                    for path, rho_list in path_dict.items():
                        for rho in rho_list:
                            ent = calc_s(rho)
                            ent_mean[gap][path].append(ent)
            else:
                for gap, path_dict in rho_dict.items():
                    for path, rho in path_dict.items():
                        ent = calc_s(rho)
                        ent_mean[gap][path] = ent
            
    if output == 'all':
        return ent_all
    elif output == 'mean':
        return ent_mean
    elif output == 'max':
        return ent_max
    elif output == 'min':
        return ent_min
    
    return None

def calc_teleported_n_mean_gap(n_list, bellstate):
    """Calculate mean negativity dict for gap only

    Args:
        n_list (list): list of negativities dictionary
        bellstate (str): name of which bellstate

    Returns:
        two dictioanries: mean and standard errors of the negativity on bellstate
    """
    # construct the dictionary
    n_dict = {gap: [] for gap in n_list.keys()}
    #if dynamic circuit
    if bellstate is None:
        for gap, path_dict in n_list.items():
            for path, value in path_dict.items():
                n_dict[gap].append(value)
    #take average of 4 variants
    elif bellstate == 'ignore':
        for gap, path_dict in n_list.items():
            for path, bellstates_dict in path_dict.items():
                if gap == 1:
                    #if len(bellstates_dict) > 2:
                        #print(f'error: expected a maximum of 2 bellstates... got: {bellstates_dict}')
                    n_dict[gap].append((bellstates_dict['BS_1']+bellstates_dict['BS_2'])/2)
                else:
                    n_dict[gap].append((bellstates_dict['BS_1']+bellstates_dict['BS_2']+
                                        bellstates_dict['BS_3']+bellstates_dict['BS_4'])/4)
    #find negativity of each variant bellstate
    else:
        for gap, path_dict in n_list.items():
            for path, bellstates_dict in path_dict.items():
                n_dict[gap].append(bellstates_dict[bellstate])
    
    n_mean = {gap: np.mean(negativities) for gap, negativities in n_dict.items()}
    
    n_std_err = {gap: np.std(negativities)/np.sqrt(len(negativities)) for gap, negativities in n_dict.items()}

    return n_mean, n_std_err