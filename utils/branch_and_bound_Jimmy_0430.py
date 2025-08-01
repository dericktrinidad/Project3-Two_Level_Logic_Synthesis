import numpy as np

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
    
    def cost_function(self,) -> float:
        return np.sum(self.matrix)

class BB_tree:
    def __init__(self, pi, prime_implicants):
        self.prime_implicants = prime_implicants
        matrix = self.build_minterm_matrix(pi)
        print('Matrix: \n', matrix.matrix)
        self.EPI = self.find_essential_prime_implicants(matrix)
        print('EPI: \n', self.EPI)
        self.final_table = self.removeTerms(matrix, self.EPI)
        print('Matrix (remove EPI): \n', self.final_table.matrix)
        print('Minterms (remove EPI): \n', self.final_table.minterms)
        self.final_list = self.finalresult(self.final_table, self.EPI)
        print('Final List: \n', self.final_list)
        self.final_equation = self.changeVariables(self.final_list)
        # self.final_equation = self.prune_matrix_shell(matrix)
        # print(self.final_equation)

#### Update 4/30
    def removeTerms(self, matrix: minterm_matrix, EPI):
        rows_to_remove = []
        for idx, (minterm, _) in enumerate(matrix.minterms):
            for epi_minterm, _ in EPI:
                if minterm == epi_minterm:
                    rows_to_remove.append(idx)
                    break
        # Remove the rows from the matrix
        matrix.matrix = np.delete(matrix.matrix, rows_to_remove, axis=0)
        # Remove the corresponding minterms from the list of minterms
        matrix.minterms = [minterm for idx, minterm in enumerate(matrix.minterms) if idx not in rows_to_remove]
        return matrix
   
    def finalresult(self, matrix: minterm_matrix, EPI):
        final_result = []
        if len(matrix.minterms) == 0:
            for _, bits_pattern in EPI:
                final_result.append(bits_pattern)
        else:
            p = [matrix.minterms]
            while len(p) > 1:
                p[1] = self.multiply(p[0], p[1])
                p.pop(0)
            final_result.append(min(p[0], key=len)[1])
            final_result.extend(bits_pattern for _, bits_pattern in EPI)
        return(final_result)

    def changeVariables(self, x):
        var = []
        for bits in x:
            variables = ''
            for i in range(len(x)):
                if bits[i] == '0':
                    variables += "'" + chr(i+65) + "'"
                elif bits[i] == '1':
                    variables += chr(i+65)
            var.append(variables)
        print('Final Equation: F = '+' + '.join(''.join(i) for i in var))
        
    def mul(self, x,y): # Multiply 2 minterms
        res = []
        for i in x:
            if i+"'" in y or (len(i)==2 and i[0] in y):
                return []
            else:
                res.append(i)
        for i in y:
            if i not in res:
                res.append(i)
        return res

    def multiply(self, x,y): # Multiply 2 expressions
        res = []
        for i in x:
            for j in y:
                tmp = self.mul(i,j)
                res.append(tmp) if len(tmp) != 0 else None
        return res
####

    # def prune_matrix_shell(self,matrix):
    #     #TODO: Find essential Prime Implicants in matrix
    #     print("Remove Essential Prime Implicants")
    #     current_minterms = ''
    #     while matrix.matrix.any():
    #         essential_pi_table, matrix, essential_minterms = self.essential_prime_implicants(matrix)
            
    #         print("Next Matrix: \n", matrix.matrix)
    #         print("Next Prime Implicants: ", matrix.prime_implicants)
    #         print("Essential Prime Implicants", essential_pi_table)
    #         print("Essential Minterms1: ", essential_minterms)
    #         for minterms_col in essential_pi_table.values():
    #             minterm_str = ''.join(str(num for num in minterms_col.flatten()))
    #             current_minterms += minterm_str
    #         #TODO: Prune matrix
    #         print("Prune matrix using col dominance")
    #         matrix = self.prune_matrix(matrix, essential_pi_table)
    #         print("Next Matrix: \n", matrix.matrix)
    #         print("Next Prime Implicants: \n", matrix.prime_implicants)
    #         conditional_minterms = self.conditional_minterms(matrix)
    #         if conditional_minterms:
    #             print("Conditional Minterms: ", conditional_minterms)

    #             # essential_minterms = essential_minterms.union(conditional_minterms)
    #             # essential_minterms_results = essential_minterms.join("+")
    #             # essential_minterms_results = ['+'.join(str(minterm) for minterm in essential_minterms) + "+" + str(condition) for condition in conditional_minterms]
    #             # print(essential_minterms_results)
    #             minterm_results =  ['+'.join(str(minterm) for minterm in essential_minterms) + "+" + str(condition) for condition in conditional_minterms]
    #             print('Final Minterm Results: ', minterm_results)
    #             return minterm_results
    #         return
            
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
                # print(minterm)
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
    
    
        