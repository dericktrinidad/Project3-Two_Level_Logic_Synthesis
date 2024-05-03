import numpy as np
import copy
import random

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

    def BCP(self,matrix, best_cost=float('inf'), current_logic_equation = []):
        #TODO: Find essential Prime Implicants in matrix
        print("Remove Essential Prime Implicants")
        current_matrix_node = copy.deepcopy(matrix)
        # next_matrix_node, next_logic_equation  = self.reduce_matrix(current_matrix_node, current_logic_equation)
        print("OPTIMIZE MATRIX")
        next_matrix_node, next_logic_equation = self.optimize_matrix_processing(current_matrix_node, current_logic_equation)
        next_matrix = next_matrix_node.matrix
        print("Next Matrix: \n", next_matrix)
        print("Next Minterms: \n", next_logic_equation)

        # next_upperbound_cost = self.upperbound_cost(next_matrix)
        # if (len(next_matrix_node.minterms) == 1): #Terminal Case
        #     if (next_upperbound_cost < best_cost):
        #         best_cost = next_upperbound_cost
        #         return next_matrix_node
        #     else:
        #         return None # No solution for this branch
        # else: # not terminal case
        #     next_lowerbound_cost = len(self.MiS_quick(next_matrix))
        #     if (next_lowerbound_cost + next_upperbound_cost > best_cost): return None #No solution on this branch
            
        #     Pi = self.choose_var(next_logic_equation)

        #     S1_node, S1_equation, S0_node, S0_equation = self.split_logic_matrix(matrix, Pi, current_logic_equation)
        #     #solution found
        #     S1_bcp = self.BCP(S1_node, best_cost=best_cost, current_logic_equation=S1_equation)
        #     S1_cost = self.upperbound_cost(S0_bcp.matrix) if S1_bcp is not None else float('inf')
        #     if S1_cost == next_lowerbound_cost: return S1_bcp
        #     S0_bcp = self.BCP(S0_node, best_cost=best_cost, current_logic_equation=S0_equation)
        #     S0_cost = self.upperbound_cost(S0_bcp.matrix) if S0_bcp is not None else float('inf')
        #     if S1_cost < S0_cost: 
        #         return S0_bcp
        #     else: 
        #         return S0_bcp
    
    def MiS_quick(self, matrix:np.array):
        MiS = set()
        matrix = matrix.copy()
        row_axis = np.arange(matrix.shape[0])
        while matrix.size > 0 and matrix.shape[1] > 0:
            print(matrix)
            row_sums = np.sum(matrix, axis = 1)
            min_sum = min(row_sums)
            min_row_indicies = np.where(row_sums == min_sum)[0]
            
            best_rows = []
            for row_idx in min_row_indicies:
                delete_rows = {row_idx}
                col_indicies = np.where(matrix[row_idx, :] == 1)[0]
                for col_idx in col_indicies:
                    row = matrix[:, col_idx]
                    row_indicies = np.where(row == 1)[0]
                    delete_rows.update(set(row_indicies))
                row_to_delete = list(delete_rows)
                row_score = np.sum(matrix[row_to_delete, :])
                best_rows.append((row_idx, row_to_delete, row_score))

            selected_row = min(best_rows, key=lambda x: x[2])
            selected_row_idx, selected_row_to_delete, _  = selected_row
            # print("selected column ", row_axis[selected_row_idx])
            MiS.add(row_axis[selected_row_idx])
            # print("cols to delete: ", [row_axis[idx] for idx in selected_row_to_delete])
            matrix = np.delete(matrix, selected_row_to_delete, axis = 0)
            row_axis = np.delete(row_axis, selected_row_to_delete)
        return MiS

    def choose_var(self, matrix:np.array):
        col_zi = []
        for col_idx in range(matrix.shape[1]):
            col = matrix[col_idx, :]
            zi = 1 / (np.sum(col) - 1)
            col_zi.append(zi)

        row_weights = []
        for row_idx in range(matrix.shape[0]):
            row = matrix[:, row_idx]
            row_ones = np.where(row == 1)[0]
            weight = np.sum([col_zi[ones_idx] for ones_idx in row_ones])
            row_weights.append(weight)
        max_weight = max(row_weights)
        print(max_weight)
        selected_columns = [col_idx for col_idx, weight in enumerate(row_weights) if weight == max_weight]
        print(selected_columns)
        return random.choice(selected_columns)


    def split_logic_matrix(self, matrix: minterm_matrix, Pi, curr_equation):
        curr_matrix = matrix.matrix
        curr_prime_implicants = matrix.prime_implicants
        
        next_matrix = np.delete(curr_matrix, Pi, axis=0)
        Pi_column = next_matrix[:, Pi]  
        S1_indicies = np.where(Pi_column == 0)[0]
        S1_prime_implicants = [curr_prime_implicants[s1_idx] for s1_idx in  S1_indicies]
        S1_equation = [curr_prime_implicants[Pi]]
        S1_equation.extend(curr_equation)
        S1_matrix = next_matrix[S1_indicies, :]
        S1_node = minterm_matrix(S1_matrix, S1_prime_implicants, matrix.minterms)

        S0_indicies = np.where(Pi_column == 1)[0]
        # S0_equation = [curr_prime_implicants[s0_idx] for s0_idx in  S0_indicies]
        S0_prime_implicants = [curr_prime_implicants[s0_idx] for s0_idx in S0_indicies]
        S0_equation = [curr_prime_implicants[Pi]]
        S0_equation.extend(curr_equation)
        S0_matrix = next_matrix[S0_indicies, :]
        S0_node = minterm_matrix(S0_matrix, S0_prime_implicants, matrix.minterms) 
        # S0_equation, S0_matrix, S1_equation, S1_matrix
        return S1_node, S1_equation, S0_node, S0_equation
    
    def upperbound_cost(self, matrix:np.array):
        return matrix.shape[1] + 1
    
    def display_minterms(self, matrix_node):
        parent_node = matrix_node.parent
        for minterms_idx in parent_node.essential_minterm_idxs:
            minterm = parent_node.minterms[minterms_idx]
            print(minterm)
            matrix_node.append(minterm)

    def conditional_minterms(self, matrix: minterm_matrix):
        current_matrix = matrix.matrix
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
    
    def optimize_matrix_processing(self, matrix: minterm_matrix, current_minterm_equation = []):
        current_matrix = matrix.matrix
        current_prime_implicants = matrix.prime_implicants
        current_minterms =  matrix.minterms
        col_sums = np.sum(current_matrix, axis=0)
        if np.any(col_sums == 1):
            essential_column_idx = np.where(col_sums == 1)[0]
            essential_minterms_idxs = set(np.where(current_matrix[:, col_idx] == 1)[0][0] for col_idx in essential_column_idx)
            col_dominance = set(essential_column_idx)
            for row_idx in essential_minterms_idxs:
                ess_row = current_matrix[row_idx, :]
                col_dominance.update(np.where(ess_row == 1)[0])
            next_matrix = np.delete(current_matrix, list(col_dominance), axis=1)
            next_prime_implicants = np.delete(current_prime_implicants, list(col_dominance), axis=0)
            non_zero_rows = np.any(next_matrix != 0, axis=1)
            next_matrix = next_matrix[non_zero_rows]
            next_minterms = [minterm for idx, minterm in enumerate(current_minterms) if non_zero_rows[idx]]
        else:
            next_matrix = current_matrix.copy()
            next_prime_implicants = matrix.prime_implicants.copy()
            next_minterms = current_minterms.copy()
        print("Matrix After Essential Check: ")
        print(next_matrix)
        # next_matrix_node = minterm_matrix(next_matrix, next_prime_implicants, next_minterms)
        md_next_matrix, md_prime_implicants, md_minterms = self.minterm_dominance(next_matrix, next_prime_implicants, next_minterms)
        print("Matrix After Minterm Dominance: ")
        print(md_next_matrix)
        print("minterms After Minterm Dominance: ")
        print(md_minterms)
        next_matrix_node = minterm_matrix(md_next_matrix, md_prime_implicants, md_minterms)
        next_matrix_node.parent = matrix
        current_minterm_equation.extend(current_minterms[idx] for idx in essential_minterms_idxs)
        return next_matrix_node, current_minterm_equation
    
    def reduce_matrix(self, curr_matrix:np.array, curr_prime_implicants:np.array, curr_minterms:np.array):
        col_sums = np.sum(current_matrix, axis=0)
        if np.any(col_sums == 1):
            essential_column_idx = np.where(col_sums == 1)[0]
            essential_minterms_idxs = set(np.where(current_matrix[:, col_idx] == 1)[0][0] for col_idx in essential_column_idx)
            col_dominance = set(essential_column_idx)
            for row_idx in essential_minterms_idxs:
                ess_row = current_matrix[row_idx, :]
                col_dominance.update(np.where(ess_row == 1)[0])
            next_matrix = np.delete(current_matrix, list(col_dominance), axis=1)
            next_prime_implicants = np.delete(current_prime_implicants, list(col_dominance), axis=0)
            non_zero_rows = np.any(next_matrix != 0, axis=1)
            next_matrix = next_matrix[non_zero_rows]
            next_minterms = [minterm for idx, minterm in enumerate(current_minterms) if non_zero_rows[idx]]
        else:
            next_matrix = current_matrix.copy()
            next_prime_implicants = matrix.prime_implicants.copy()
            next_minterms = current_minterms.copy()


    def minterm_dominance(self, curr_matrix:np.array, curr_prime_implicants:list, curr_minterms:list):
        unique_rows = set()
        row_sums = np.sum(curr_matrix, axis = 1)
        rows_to_delete = set()
        for i in range(curr_matrix.shape[0]):
            
            row_i = curr_matrix[i, :]
            if tuple(row_i) in unique_rows:
                        rows_to_delete.add(i)
                        continue
            
            for j in range(curr_matrix.shape[0]):
                if i == j: continue
                if row_sums[i] < row_sums[j] and self.check_dominance(curr_matrix[i, :], curr_matrix[j, :]):
                    rows_to_delete.add(i)
                elif row_sums[i] > row_sums[j] and self.check_dominance(curr_matrix[j, :], curr_matrix[i, :]):
                    rows_to_delete.add(j)
        delete_rows = list(rows_to_delete)
        next_matrix = np.delete(curr_matrix, delete_rows, axis=0)
        next_minterm = [minterm for idx, minterm in  enumerate(curr_minterms) if idx not in rows_to_delete]
        return next_matrix, curr_prime_implicants, next_minterm
    
    def check_dominance(self, row1, row2):
        return np.all((row1|row2) == row2)
    

