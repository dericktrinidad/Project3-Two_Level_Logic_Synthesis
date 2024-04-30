import numpy as np
from collections import defaultdict
import copy

class minterm_matrix:
    def __init__(self, matrix, prime_implicants, minterms):
        self.matrix = matrix #-> np.array matrix 0 or 1
        self.prime_implicants = prime_implicants #-> List of integers
        self.minterms = minterms #-> List of tuples
        self.cost = None #-> integer

    def __lt__(self, other) -> bool:
        if isinstance(other, minterm_matrix):
            return self.cost < other.cost
        return NotImplemented

class prime_implicant:
    def __init__(self, pi_table, prev_leftovers=None):

        self.pi_table = pi_table
        self.prev_leftovers = prev_leftovers
        self.parent = None
        self.child = None
        self.leftover_check = None

class branch_bound_tree:

    def __init__(self, prime_implicants):
        self.prime_implicants = prime_implicants
        self.pi_idx_table = {idx:pi for idx, pi in enumerate(self.prime_implicants)}
        self.max_bit = format(max(prime_implicants), 'b')
        self.most_significant_bit = len(self.max_bit)
        binary_table = self.build_binary_table(prime_implicants)
        initial_pi = self.build_initial_pi(binary_table)
        final_pi = self.spi_shell(initial_pi)
        matrix = self.build_minterm_matrix(final_pi)
        print(matrix.matrix)
        final_equation = self.prune_matrix_shell(matrix)
        print(final_equation)

    def prune_matrix_shell(self,matrix):
        #TODO: Find essential Prime Implicants in matrix
        print("Remove Essential Prime Implicants")
        current_minterms = ''
        while matrix.matrix.any():
            essential_pi_table, matrix, essential_minterms = self.essential_prime_implicants(matrix)
            
            print("Next Matrix: \n", matrix.matrix)
            print("Next Prime Implicants: ", matrix.prime_implicants)
            print("Essential Prime Implicants", essential_pi_table)
            print("Essential Minterms1: ", essential_minterms)
            for minterms in essential_minterms:
                current_minterms += minterms
            #TODO: Prune matrix
            print("Prune matrix using col dominance")
            matrix = self.prune_matrix(matrix, essential_pi_table)
            print("Next Matrix: \n", matrix.matrix)
            print("Next Prime Implicants: \n", matrix.prime_implicants)
            conditional_minterms = self.conditional_minterms(matrix)
            if conditional_minterms:
                print("Conditional Minterms: ", conditional_minterms)

                # essential_minterms = essential_minterms.union(conditional_minterms)
                essential_minterms_results = essential_minterms.join("+")
                print(essential_minterms_results)
                minterm_results =  [essential_minterms+ "+" + condition for condition in conditional_minterms]
                print('Final Minterm Results: ', minterm_results)
                return minterm_results
            
        
    def conditional_minterms(self, matrix: minterm_matrix):
        current_matrix = matrix.matrix
        # current_prime_implicants = matrix.prime_implicants
        essential_minterms = set()
        for row_idx in range(current_matrix.shape[0]):
            row = current_matrix[row_idx, :]
            if np.all(row == 1):
                essential_minterms.add(row_idx)
            elif np.any(row == 1) and np.any(row ==  0):
                return
        return essential_minterms

    def find_essential_prime_implicants(self, matrix):
        EPI_row_axis = []
        processed_minterms = set()
        for col_idx in range(matrix.matrix.shape[1]):
            count = np.sum(matrix.matrix[:, col_idx])
            if count == 1:
                minterm_idx = np.where(matrix.matrix[:, col_idx] == 1)[0][0]
                minterm = matrix.minterms[minterm_idx]
                if minterm not in processed_minterms:
                    EPI_row_axis.append(minterm)
                    processed_minterms.add(minterm)
        return EPI_row_axis


    def prune_matrix(self, matrix: minterm_matrix, essential_pi_table)-> np.array:
        current_matrix = matrix.matrix
        current_prime_implicants = matrix.prime_implicants
        intersections = set()
        for curr_col_idx, ess_col in essential_pi_table.items():
            ess_col_indicies = np.where(ess_col == 1)[0]
            ess_col_indicies_set = set(ess_col_indicies)
            for col_idx in range(current_matrix.shape[1]):
                if curr_col_idx == col_idx: continue
                col = current_matrix[:, col_idx]
                col_indicies = set(np.where(col == 1)[0])
                intersection = ess_col_indicies_set.intersection(col_indicies)
                if intersection:
                    intersections.add(col_idx)

        if intersections:
            next_matrix = np.delete(current_matrix, list(intersections), axis=1)
            next_prime_implicants = np.delete(current_prime_implicants, list(intersections), axis = 0)
        return minterm_matrix(next_matrix, next_prime_implicants, matrix.minterms)

    def essential_prime_implicants(self, matrix: minterm_matrix) -> tuple:
        current_matrix = matrix.matrix
        current_prime_implicants = matrix.prime_implicants
        current_minterms = matrix.minterms
        col_sums = np.sum(current_matrix, axis = 0)
        min_col_sums = min(col_sums)
        essential_column_idx = np.where(col_sums == min_col_sums)[0]
        essential_minterm_idxs = set(np.where(current_matrix[:, col_idx] == min_col_sums)[0][0] for col_idx in essential_column_idx)
        essential_column_idx_table = {index: current_matrix[:,index] for index in essential_column_idx}
        next_matrix = np.delete(current_matrix, essential_column_idx, axis = 1)
        next_prime_implicants = np.delete(current_prime_implicants, essential_column_idx, axis = 0)
        return essential_column_idx_table, minterm_matrix(next_matrix, next_prime_implicants, current_minterms), essential_minterm_idxs


    def build_minterm_matrix(self, pi):
        current_pi_table = pi.pi_table
        parent_pi_table = pi.parent.pi_table
        leftover_minterms = pi.parent.leftover_check
        
        print('leftover_minterms: ', leftover_minterms)
        # current_minterms = set(minterm for minterms in current_pi_table.values() for minterm in minterms.keys())
        current_minterm_table = [(minterm, bits) for minterms in current_pi_table.values() for minterm, bits in minterms.items()]
        if leftover_minterms:
            leftover_minterm_table = [(minterm, bits) for minterms in parent_pi_table.values() for minterm, bits in minterms.items() if minterm in leftover_minterms]
            current_minterm_table.extend(leftover_minterm_table)

        minterm_table_sorted = sorted(current_minterm_table, key=lambda x: (len(x[0]), x[0]))
        print('Row Axis: ', minterm_table_sorted)
        print('Column Axis: ', self.prime_implicants)
        matrix = np.zeros((len(minterm_table_sorted), len(self.prime_implicants)), dtype='int')
        for idx, minterm_tuple in enumerate(minterm_table_sorted):
            minterm = minterm_tuple[0]
            row = matrix[idx, :]
            for bit in minterm:
                row_idx = self.prime_implicants.index(bit)
                row[row_idx] = 1
        return minterm_matrix(matrix, self.prime_implicants, minterm_table_sorted)
    
    def build_binary_table(self, prime_implicants):
        max_int = max(prime_implicants)
        max_bit = format(max_int, 'b')
        bit_format = str(len(max_bit))+'b'
        binary_representations = {integer: format(integer, bit_format).replace(' ', '0') for integer in prime_implicants}
        print(binary_representations)
        return binary_representations
    
    def build_initial_pi(self, binary_lists):
        pi_table = defaultdict(lambda: defaultdict(str))
        for minterm, bits in binary_lists.items():
            uninverted_bits_count = sum([int(char) for char in bits])
            pi_table[uninverted_bits_count][(minterm,)]=bits
        pi = prime_implicant(pi_table)
        return pi

    def spi_shell(self, pi):
        best_pi = None
        while (pi.pi_table):
            pi = self.simplify_prime_implicants(pi)
            if pi.pi_table:
                best_pi = copy.deepcopy(pi)
        return best_pi
    
    def simplify_prime_implicants(self, pi):
        pi_table = pi.pi_table
        uninv_bits_counts = list(pi_table.keys())
        
        next_pi_table = defaultdict(lambda: defaultdict(str))
        next_pi = prime_implicant(next_pi_table)
        next_pi.parent = pi
        pi.child = next_pi

        unique_bits = set()
        pi.leftover_check = set(minterms for group_minterms in pi_table.values() for minterms in group_minterms.keys())
        for idx in range(len(uninv_bits_counts)-1):
            current_group, next_group = uninv_bits_counts[idx], uninv_bits_counts[idx+1]
            for current_minterms, current_bits in pi_table[current_group].items():
                for next_minterms, next_bits in pi_table[next_group].items():
                    absorb_count = 0
                    variable = ''
                    for byte_idx in range(0,self.most_significant_bit):
                        current_byte = current_bits[byte_idx]
                        next_byte = next_bits[byte_idx]
                        if current_byte == next_byte:
                            variable += current_bits[byte_idx]
                        else:
                            variable += '-'
                            absorb_count += 1
                    if absorb_count == 1:
                        if  variable not in unique_bits:
                            next_pi_table[idx][current_minterms+next_minterms] = variable
                            unique_bits.add(variable)
                        pi.leftover_check.discard(current_minterms), pi.leftover_check.discard(next_minterms)
        return next_pi

    def get_minterm_matrix(self) -> np.array:
        max_int = max(self.input_array)
        max_bit = format(max_int, 'b')
        bit_format = str(len(max_bit))+'b'
        minterm_matrix = np.zeros((len(max_bit), len(input)), dtype='int')
        for idx, integer in enumerate(input):
            minterm = format(integer, bit_format).replace(' ', '0')
            minterm_array = np.array([int(char) for char in minterm])
            minterm_matrix[:, idx] = minterm_array
        return minterm_matrix

    def MiS_quick(self, minterm_matrix)-> np.array:
        current_matrix = minterm_matrix.matrix
        min_idx = self.Best_MiS(current_matrix)
        lb_table = self.get_lbtable(current_matrix, min_idx)


    def get_lbtable(self, matrix, min_idx):
        best_column  = matrix[:, min_idx]
        best_indicies = set(np.where(best_column == 1)[0])
        lb_table = matrix.copy()
        
        for col_idx in range(matrix.shape[1]):
            if col_idx == min_idx:
                continue
            column = matrix[:, col_idx]
            indicies = set(np.where(column == 1)[0])
            intersections = best_indicies.intersection(indicies)
            if intersections:
                np.delete(matrix, col_idx, axis=1)
        return lb_table

    def get_minterm_indicies(self, matrix):
        minterm_indicies = []
        for col_idx in range(matrix.shape[1]):
            column = matrix[:, col_idx]
            indicies = set(np.where(column == 1)[0])
            minterm_indicies.append(indicies)
        return minterm_indicies

    def Best_MiS(self, matrix):
        best_cost = float('inf') 
        for col_idx in range(matrix.shape[1]):
            column = matrix[:, col_idx]
            cost = np.sum(column)
            if column < best_cost:
                best_cost = cost
                MiS_idx = col_idx
        return MiS_idx
    
    def cost_function(self,) -> float:
        return np.sum(self.matrix)