import numpy as np
from collections import defaultdict
import copy

class minterm_matrix:
    def __init__(self, matrix, prime_implicants):
        self.matrix = matrix #-> np.array matrix 0 or 1
        self.prime_implicants = prime_implicants #-> List of integers
        self.cost = None #-> integer

    def __lt__(self, other) -> bool:
        if isinstance(other, minterm_matrix):
            return self.cost < other.cost
        return NotImplemented

class prime_implicant:
    def __init__(self, group_table, minterm_table, prev_leftovers=None):
        self.group_table = group_table
        self.minterm_table = minterm_table
        self.prev_leftovers = prev_leftovers
        self.parent = None
        self.child = None
        self.last_group_check = None

class branch_bound_tree:
    def __init__(self, prime_implicants):
        self.prime_implicants = prime_implicants
        self.pi_idx_table = {idx:pi for idx, pi in enumerate(self.prime_implicants)}
        self.max_bit = format(max(prime_implicants), 'b')
        self.most_significant_bit = len(self.max_bit)

        binary_table = self.build_binary_table(prime_implicants)
        group_table, minterm_table = self.build_initial_tables(binary_table)
        initial_pi = prime_implicant(group_table, minterm_table)
        final_pi = self.spi_shell(initial_pi)
        final_minterm_table = final_pi.minterm_table
        final_group_table = final_pi.group_table
        print("final group table: ", final_group_table)
        print("final minterm table: ", final_minterm_table)
        print("parent group table: ", final_pi.parent.group_table)
        print("parent minterm table: ", final_pi.parent.minterm_table)
        print("parent final leftover minterms: ", final_pi.parent.last_group_check)
        leftover = set()
        parent_minterm_table = final_pi.parent.minterm_table
        parent_group_table = final_pi.parent.group_table
        parent_group_check = final_pi.parent.last_group_check
        for group in parent_group_check:
            # print(group)
            leftover.add((group, parent_minterm_table[group]))
        # print(final_minterm_table)
        matrix = self.build_minterm_matrix(final_minterm_table)
        # print(matrix.matrix)
        #TODO: Find essential Prime Implicants in matrix
        # essential_pi_table, next_matrix = self.essential_prime_implicants(matrix)
        #TODO: Prune matrix
        # next_matrix = self.prune_matrix(next_matrix, essential_pi_table)
        # print(next_matrix.matrix)
        # print(next_matrix.prime_implicants)

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
        return minterm_matrix(next_matrix, next_prime_implicants)

    def essential_prime_implicants(self, matrix: minterm_matrix) -> tuple:
        current_matrix = matrix.matrix
        current_prime_implicants = matrix.prime_implicants
        col_sums = np.sum(current_matrix, axis = 0)
        min_col_sums = min(col_sums)
        essential_column_idx = np.where(col_sums == min_col_sums)[0]
        essential_column_idx_table = {index: current_matrix[:,index] for index in essential_column_idx}
        next_matrix = np.delete(current_matrix, essential_column_idx, axis = 1)
        next_prime_implicants = np.delete(current_prime_implicants, essential_column_idx, axis = 0)
        return essential_column_idx_table, minterm_matrix(next_matrix, next_prime_implicants)


    def build_minterm_matrix(self, pi):
        minterm_table = pi.minterm_table
        matrix = np.zeros((self.most_significant_bit, len(self.prime_implicants)), dtype='int')
        for idx, minterm in enumerate(minterm_table.keys()):
            row = matrix[idx, :]
            for bit in minterm:
                row_idx = self.prime_implicants.index(bit)
                row[row_idx] = 1
        return minterm_matrix(matrix, self.prime_implicants)
    
    def build_binary_table(self, prime_implicants):
        max_int = max(prime_implicants)
        max_bit = format(max_int, 'b')
        bit_format = str(len(max_bit))+'b'
        binary_representations = {integer: format(integer, bit_format).replace(' ', '0') for integer in prime_implicants}
        print(binary_representations)
        return binary_representations
    
    def build_initial_tables(self, binary_lists):
        minterm_table = defaultdict(str)
        group_table = defaultdict(list)
        for minterm, bits in binary_lists.items():
            uninverted_bits_count = sum([int(char) for char in bits])
            group_table[uninverted_bits_count].append((minterm,))
            for byte in bits:
                minterm_table[(minterm,)]+=str(byte)
        return group_table, minterm_table

    def spi_shell(self, pi):
        best_pi = None
        while (pi.minterm_table or pi.group_table):
            pi = self.simplify_prime_implicants(pi)
                
            if pi.minterm_table and pi.group_table:
                best_pi = copy.deepcopy(pi)
        return best_pi
    
    def simplify_prime_implicants(self, pi):
        group_table, minterm_table = pi.group_table, pi.minterm_table

        uninv_bits_counts = list(group_table.keys())
        
        next_group_table = defaultdict(list)
        next_minterm_table = {}
        next_pi = prime_implicant(next_group_table, next_minterm_table)
        next_pi.parent = pi
        pi.child = next_pi

        unique_bits = set()
        pi.last_group_check = set(group for group_vals in group_table.values() for group in group_vals)

        for idx in range(len(uninv_bits_counts)-1):
            current_group, next_group = uninv_bits_counts[idx], uninv_bits_counts[idx+1]
            for current_minterms in group_table[current_group]:
                current_bits = minterm_table[current_minterms]

                for next_minterms in group_table[next_group]:
                    absorb_count = 0
                    variable = ''
                    
                    next_bits = minterm_table[next_minterms]

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
                            next_pi.group_table[idx].append((current_minterms+next_minterms))
                            next_pi.minterm_table[current_minterms+next_minterms] = variable
                            unique_bits.add(variable)
                        pi.last_group_check.discard(current_minterms)
                        pi.last_group_check.discard(next_minterms)
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