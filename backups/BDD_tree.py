from collections import defaultdict

class minterm_node:
    def __init__(self, minterms):
        self.minterms = minterms
        self.parent_minterm = None
        self.left_minterm = None
        self.right_minterm = None
        self.isLeaf = False
        self.factored_minterm = None

class bdd:
    def __init__(self, sum_of_minterms):
        # self.initial_implicants = initial_implicants
        # self.pi_idx_table = {idx:pi for idx, pi in enumerate(self.initial_implicants)}
        self.leafNodes = []
        self.max_bit = format(max(sum_of_minterms), 'b')
        self.most_significant_bit = len(self.max_bit)
        initial_minterms = self.build_initial_equation(sum_of_minterms)
        print("Initial Minterms: ")
        print(initial_minterms.minterms)
        simplified_minterms = self.bdd_shell(initial_minterms)
        print([node.minterms for node in self.leafNodes])

    def bdd_shell(self, minterm_node, current_count=0):
        curr_minterms = minterm_node.minterms
        if all(minterm.count(None) == self.most_significant_bit - 1 for minterm in curr_minterms):
            self.leafNodes.append(minterm_node)
            return minterm_node
        if current_count >= self.most_significant_bit:
            self.leafNodes.append(minterm_node)
            return minterm_node
        max_var = self.max_variable_count(curr_minterms)
        print("max variable count: ")
        print(max_var)
        left_minterms, right_minterms = self.simplify_minterms(curr_minterms, max_var)
        print("left minterms: \n", left_minterms.minterms)
        print("Right Minterms: \n", right_minterms.minterms)
        minterm_node.left_minterm, minterm_node.right_minterm = left_minterms, right_minterms
        left_minterms.parent_minterm = right_minterms.parent_minterm = minterm_node
        self.bdd_shell(left_minterms, current_count+1), self.bdd_shell(right_minterms, current_count+1)
        # return

    def simplify_minterms(self, minterms, max_var):
        left_minterm = [None] * self.most_significant_bit
        right_minterm = [None] * self.most_significant_bit

        left_minterm[max_var] = 0
        right_minterm[max_var] = 1

        
        left_minterms, right_minterms = [], []
        left_node, right_node = minterm_node(left_minterms), minterm_node(right_minterms)
        left_node.factored_minterm, right_node.factored_minterm = left_minterm, right_minterm

        for minterm in minterms:
            curr_minterm = minterm.copy()
            curr_var = curr_minterm[max_var]
            if curr_minterm:
                if curr_var == 1:
                    curr_minterm[max_var] = None
                    left_node.minterms.append(curr_minterm)
                elif curr_var == 0:
                    curr_minterm[max_var] = None
                    right_node.minterms.append(curr_minterm)
                else: #curr_var == None
                    left_node.minterms.append(curr_minterm)
                    right_node.minterms.append(curr_minterm)
        return left_node, right_node

    def build_initial_equation(self, sum_of_minterms):
        bit_format = str(self.most_significant_bit)+'b'
        minterms = []
        for value in sum_of_minterms:
            bits = format(value, bit_format).replace(' ', '0')
            minterm = [int(char) for char in bits]
            minterms.append(minterm)
        return minterm_node(minterms)
    
    def max_variable_count(self, minterms):
        variable_count = defaultdict(int)
        for minterm in minterms:
            for idx, state in enumerate(minterm):
                if state is not None:
                    variable_count[idx] += 1
        max_count_key = max(variable_count, key=variable_count.get)
        return max_count_key