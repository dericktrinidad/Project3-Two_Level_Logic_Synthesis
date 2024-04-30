import numpy as np
from collections import defaultdict
import copy

class prime_implicant:
    def __init__(self, pi_table, prev_leftovers=None):

        self.pi_table = pi_table
        self.prev_leftovers = prev_leftovers
        self.parent = None
        self.child = None
        self.leftover_check = None

class QMcC_iter:
    def __init__(self, prime_implicants):
        self.prime_implicants = prime_implicants
        self.pi_idx_table = {idx:pi for idx, pi in enumerate(self.prime_implicants)}
        self.max_bit = format(max(prime_implicants), 'b')
        self.most_significant_bit = len(self.max_bit)
        binary_table = self.build_binary_table(prime_implicants)
        initial_pi = self.build_initial_pi(binary_table)
        self.final_pi = self.spi_shell(initial_pi)

    def build_binary_table(self, prime_implicants):
        max_int = max(prime_implicants)
        max_bit = format(max_int, 'b')
        bit_format = str(len(max_bit))+'b'
        binary_representations = {integer: format(integer, bit_format).replace(' ', '0') for integer in prime_implicants}
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
    
