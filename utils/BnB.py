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
        self.final_node, self.final_equation = self.BCP(matrix)
        print("Final Equation: ", self.final_equation)
        self.finalresult(self.final_equation)
        print("Branch and Bound Algorithm Complete!")

    def BCP(self,matrix, best_cost=float('inf'), current_logic_equation = []):
        #TODO: Find essential Prime Implicants in matrix
        current_matrix_node = copy.deepcopy(matrix)
        next_matrix_node, next_logic_equation = self.optimize_matrix_processing(current_matrix_node, current_logic_equation)
        next_matrix = next_matrix_node.matrix

        next_upperbound_cost = self.upperbound_cost(next_matrix)
        if (len(next_matrix_node.minterms) <= 1 or next_matrix.size == 0): #Terminal Case
            if (next_upperbound_cost < best_cost):
                best_cost = next_upperbound_cost
                print("Solution Found!")
                return next_matrix_node, next_logic_equation
            else:
                return None, next_logic_equation # No solution for this branch
        else: # not terminal case
            next_lowerbound_cost = len(self.MiS_quick(next_matrix))
            if (next_lowerbound_cost + next_upperbound_cost > best_cost): return None, next_logic_equation #No solution on this branch
            
            Pi = self.choose_var(next_matrix)
            S1_node, S1_equation, S0_node, S0_equation = self.split_logic_matrix(next_matrix_node, Pi, current_logic_equation)
            if S1_node is not None:
                S1_node, S1_equation = self.BCP(S1_node, best_cost=best_cost, current_logic_equation=S1_equation)
            S1_cost = self.upperbound_cost(S1_node.matrix)
            if S1_cost == next_lowerbound_cost: return S1_node, S1_equation

            if S0_node is not None:
                S0_node, S0_equation = self.BCP(S0_node, best_cost=best_cost, current_logic_equation=S0_equation)
            S0_cost = self.upperbound_cost(S0_node.matrix)
            if S1_cost < S0_cost: 
                return S1_node, S1_equation
            elif S1_cost > S0_cost: 
                return S0_node, S0_equation
            elif S1_cost == float('inf') and S0_cost == float('inf'): # No Solution In this Branch
                print("No Solution In This Branch")
                return None, next_logic_equation
            else:
                print("No Solution In This Branch")
                return None, next_logic_equation
            
    def finalresult(self, matrix: minterm_matrix):
        final_result = []
        for _, bits_pattern in matrix:
            final_result.append(bits_pattern)
        var = []
        for bits in final_result:
            variables = ''
            for i in range(len(bits)):
                if bits[i] == '0':
                    variables += "'" + chr(i+65) + "'"
                elif bits[i] == '1':
                    variables += chr(i+65)
            var.append(variables)
        print('Final Equation: F = '+' + '.join(''.join(i) for i in var))
                
    def MiS_quick(self, matrix:np.array):
        MiS = set()
        matrix = matrix.copy()
        row_axis = np.arange(matrix.shape[0])
        while matrix.size > 0:
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
            MiS.add(row_axis[selected_row_idx])
            matrix = np.delete(matrix, selected_row_to_delete, axis = 0)
            row_axis = np.delete(row_axis, selected_row_to_delete)
        return MiS

    def choose_var(self, matrix:np.array):
        col_zi = []
        for col_idx in range(matrix.shape[1]):
            col = matrix[col_idx, :]
            col_sum = np.sum(col)
            if col_sum > 1:
                zi = 1 / (col_sum - 1)
            else:
                zi = float('inf')
            col_zi.append(zi)
        row_weights = []
        for row_idx in range(matrix.shape[0]):
            row = matrix[row_idx, :]
            row_ones = np.where(row == 1)[0]
            weight = 0
            for ones_idx in row_ones:
                col_zi_sum = col_zi[ones_idx]
                if not np.isinf(col_zi_sum):
                    weight += col_zi_sum
            row_weights.append(weight)
        max_weight = max(row_weights)
        selected_columns = [col_idx for col_idx, weight in enumerate(row_weights) if weight == max_weight]
        return random.choice(selected_columns)


    def split_logic_matrix(self, matrix: minterm_matrix, Pi, curr_equation):
        curr_matrix = matrix.matrix
        curr_prime_implicants = matrix.prime_implicants
        curr_minterms =  matrix.minterms
        next_matrix = np.delete(curr_matrix, Pi, axis=0)
        next_minterms = curr_minterms.copy()
        del next_minterms[Pi]

        Pi_column = curr_matrix[Pi, :]
        S1_indicies = np.where(Pi_column == 1)[0]
        S1_equation = [curr_minterms[Pi]]
        S1_equation.extend(curr_equation)
        S1_matrix = np.delete(next_matrix, S1_indicies, axis=1)
        S1_minterms = [minterm for idx, minterm in enumerate(next_minterms) if idx not in set(S1_indicies)]
        S1_node = minterm_matrix(S1_matrix, curr_prime_implicants, S1_minterms)
        S0_indicies = np.where(Pi_column == 0)[0]
        S0_equation = []
        S0_equation.extend(curr_equation)
        S0_matrix = np.delete(next_matrix, S0_indicies, axis=1)
        S0_minterms = [minterm for idx, minterm in enumerate(next_minterms) if idx not in set(S0_indicies)]
        S0_node = minterm_matrix(S0_matrix, curr_prime_implicants, S0_minterms)
        return S1_node, S1_equation, S0_node, S0_equation
    
    def upperbound_cost(self, matrix:np.array):
        return matrix.shape[1] + 1

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
        essential_minterms_idxs = set()
        while True:
            col_sums = np.sum(current_matrix, axis=0)
            if not np.any(col_sums == 1):
                break
            minterm_lookup = current_minterms.copy()
            current_matrix, current_prime_implicants, current_minterms, essential_minterm_update = self.reduce_matrix(
                        current_matrix, current_prime_implicants, current_minterms)
            essential_minterms_idxs.update(minterm_lookup[idx] for idx in essential_minterm_update)
            if current_matrix.size < 0:
                break
            current_matrix, current_prime_implicants, current_minterms = self.minterm_dominance(
                        current_matrix, current_prime_implicants, current_minterms)

        next_matrix_node = minterm_matrix(current_matrix, current_prime_implicants, current_minterms)
        next_matrix_node.parent = matrix
        current_minterm_equation.extend(essential_minterms_idxs)
        return next_matrix_node, current_minterm_equation
    
    def reduce_matrix(self, curr_matrix:np.array, curr_prime_implicants:np.array, curr_minterms:np.array):
        col_sums = np.sum(curr_matrix, axis=0)
        essential_column_idx = np.where(col_sums == 1)[0]
        essential_minterms_idxs = set(np.where(curr_matrix[:, col_idx] == 1)[0][0] for col_idx in essential_column_idx)
        col_dominance = set(essential_column_idx)
        for row_idx in essential_minterms_idxs:
            ess_row = curr_matrix[row_idx, :]
            col_dominance.update(np.where(ess_row == 1)[0])
        next_matrix = np.delete(curr_matrix, list(col_dominance), axis=1)
        next_prime_implicants = np.delete(curr_prime_implicants, list(col_dominance), axis=0)
        non_zero_rows = np.any(next_matrix != 0, axis=1)
        next_matrix = next_matrix[non_zero_rows]
        next_minterms = [minterm for idx, minterm in enumerate(curr_minterms) if non_zero_rows[idx]]
        return next_matrix, next_prime_implicants, next_minterms, essential_minterms_idxs

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
    

