import numpy as np
import copy
class minterm_matrix:
    def __init__(self, matrix, prime_implicants, minterms):
        self.matrix = matrix #-> np.array matrix 0 or 1
        self.prime_implicants = prime_implicants #-> List of integers
        self.minterms = minterms #-> List of tuples
        self.cost = None #-> integer
        self.essential_pi_table = None
        self.parent = None
        self.essential_minterm_idxs = None

class BB_tree:
    def __init__(self, pi, prime_implicants):
        self.prime_implicants = prime_implicants
        matrix = self.build_minterm_matrix(pi)

        print(matrix.matrix)
        self.final_equation = self.BCP(matrix)
        print(self.final_equation)

    def BCP(self,matrix, best_cost=float('inf'), current_minterms = []):
        #TODO: Find essential Prime Implicants in matrix
        print("Remove Essential Prime Implicants")
        curr_matrix_node = copy.deepcopy(matrix)
        next_matrix_node = self.reduce_matrix(curr_matrix_node)
        print("Next Matrix: \n", next_matrix_node.matrix)
        parent_node = next_matrix_node.parent
        for minterms_idx in parent_node.essential_minterm_idxs:
            minterm = parent_node.minterms[minterms_idx]
            current_minterms.append(minterm)
        print("Current Minterms: ", current_minterms)
        curr_cost = self.cost_function(next_matrix_node.matrix)
        if (len(next_matrix_node.prime_implicants) == 1): #Terminal Case
            if (curr_cost < best_cost):
                best_cost = curr_cost
                return next_matrix_node
            else:
                return None # No solution for this branch
        else: # not terminal case
            LB = self.MiS_quick()
            if (LB + curr_cost > best_cost): return None #No solution on this branch
            else: #solution found
                S1 = self.BCP(next_matrix_node, best_cost=best_cost, current_minterms=current_minterms)
                S1_cost = self.cost_function(S1.matrix)
                if(S1_cost == LB): return S1
                else:
                    S0 = self.BCP(next_matrix_node, best_cost=best_cost, current_minterms=current_minterms)
                    return S

        # conditional_minterms = self.conditional_minterms(matrix)
        # if conditional_minterms:
        #     print("Conditional Minterms: ", conditional_minterms)
        #     minterm_results =  [current_minterms + "+" + condition for condition in conditional_minterms]
        #     print('Final Minterm Results: ', minterm_results)
        #     return minterm_results
    def cost_function(matrix:np.array):
        return np.sum(matrix)
    
    def display_minterms(self, matrix_node):
        parent_node = matrix_node.parent
        for minterms_idx in parent_node.essential_minterm_idxs:
            minterm = parent_node.minterms[minterms_idx]
            print(minterm)
            matrix_node.append(minterm)

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
                print(minterm)
                if minterm not in processed_minterms:
                    EPI_row_axis.append(minterm)
                    processed_minterms.add(minterm)
        return EPI_row_axis    

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
    
    # def prune_matrix(self, matrix: minterm_matrix, essential_pi_idx)-> np.array:
    #     current_matrix = matrix.matrix
    #     current_prime_implicants = matrix.prime_implicants
    #     current_minterms = matrix.minterms
    #     col_dominance = set()
    #     print("Essential_pi_idx: ", essential_pi_idx)
    #     for row_idx in essential_pi_idx:
    #         ess_row = current_matrix[row_idx, :]
    #         ess_col_indicies = set(np.where(ess_row == 1)[0])
    #         print("essentical col_inidicies",ess_col_indicies)
    #         col_dominance.update(ess_col_indicies)

    #     print("Column Dominance: ", col_dominance)
    #     next_matrix = np.delete(current_matrix, list(col_dominance), axis=1)
    #     next_prime_implicants = np.delete(current_prime_implicants, list(col_dominance), axis = 0)

    #     non_zero_rows = np.any(next_matrix != 0, axis=1)
    #     next_matrix = next_matrix[non_zero_rows]
    #     next_minterms = [pi for idx, pi in enumerate(current_minterms) if idx in set(non_zero_rows) ]
    #     return minterm_matrix(next_matrix, next_prime_implicants, next_minterms)

    def reduce_matrix(self, matrix: minterm_matrix) -> tuple:
        current_matrix = matrix.matrix
        current_prime_implicants = matrix.prime_implicants
        current_minterms = matrix.minterms
        col_sums = np.sum(current_matrix, axis = 0)
        min_col_sums = min(col_sums)
        essential_column_idx = np.where(col_sums == min_col_sums)[0]
        essential_minterms_idxs = set(np.where(current_matrix[:, col_idx] == min_col_sums)[0][0] for col_idx in essential_column_idx)
        essential_column_idx_table = {index: current_matrix[:,index] for index in essential_column_idx}
        matrix.essential_pi_table = essential_column_idx_table
        matrix.essential_minterm_idxs = essential_minterms_idxs
        next_matrix = np.delete(current_matrix, essential_column_idx, axis = 1)
        col_dominance = set()

        for row_idx in essential_minterms_idxs:
            ess_row = current_matrix[row_idx, :]
            ess_col_indicies = set(np.where(ess_row == 1)[0])
            col_dominance.update(ess_col_indicies)
        col_dominance.update(set(essential_column_idx))
        next_matrix = np.delete(current_matrix, list(col_dominance), axis=1)
        next_prime_implicants = np.delete(current_prime_implicants, list(col_dominance), axis = 0)

        non_zero_rows = np.any(next_matrix != 0, axis=1)
        next_matrix = next_matrix[non_zero_rows]
        next_minterms = [pi for idx, pi in enumerate(current_minterms) if idx in set(non_zero_rows) ]
        next_matrix_node = minterm_matrix(next_matrix, next_prime_implicants, next_minterms)
        next_matrix_node.parent = matrix
        return next_matrix_node
    
    
        