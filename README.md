# Feducha Mathesis Algorithm & Branch and Bound Optimization
## Project Overview

This project implements the Feducha Mathesis Algorithm combined with the Branch and Bound Optimization technique to solve complex combinatorial problems. The primary focus is on efficiently minimizing Boolean functions and their respective equations. The algorithms work together to handle various test cases, perform simplifications, and compute optimal solutions for different types of Boolean expressions.

The Feduccia Mathesis Algorithm operates as an optimization method for minimization problems, focusing on finding the simplest Boolean expression that satisfies a given truth table. On the other hand, the Branch and Bound algorithm is employed to explore and prune the search space, making the problem-solving process computationally feasible for larger datasets.

#### Key Features

Boolean minimization using the Feducha Mathesis algorithm.

Optimization through Branch and Bound techniques.

Solves Boolean functions to find minimized forms.

Can handle multiple test cases and large inputs.

Outputs final Boolean equations and their minimized forms.

## Algorithm Overview
Feduccia Mathesis Algorithm

The Feduccia Mathesis Algorithm is a specialized method for minimizing Boolean expressions, typically used in digital logic design and simplification of logic circuits. This algorithm focuses on systematically reducing Boolean functions by exploring the most efficient ways to simplify them.

#### Steps in Feducca Mathesis:

Initialization: The algorithm begins by constructing the initial truth table and generating a prime implicant chart.

Iterative Optimization: It iterates through the input set, making necessary decisions based on the established criteria to prune unnecessary terms.

Finalization: The algorithm outputs the minimal sum of products or the simplest Boolean expression.

#### Branch and Bound Algorithm

The Branch and Bound (BB) method is used to explore possible solutions for combinatorial optimization problems, pruning solutions that can't lead to better results. It works by recursively exploring subsets of the problem and bounding the suboptimal ones, ensuring that only the best solutions are considered.

Steps in Branch and Bound:

Branching: The search space is divided into smaller subproblems, or “branches”.

Bounding: The algorithm computes upper and lower bounds for each branch.

Pruning: If a branch cannot lead to a better solution than the current best, it is pruned.

Final Solution: The algorithm eventually converges on the optimal solution by iterating through the problem space.
```Bash
Project Structure
/project-directory
│
├── /utils                  # Utility functions for Feducca Mathesis and Branch and Bound algorithms.
│   ├── QMcC.py             # Core implementation of Feducca Mathesis algorithm.
│   └── branch_and_bound.py # Core implementation of Branch and Bound algorithm.
│
├── test_cases.py           # Test case definitions and algorithm executions.
├── README.md               # Project overview and documentation.
└── requirements.txt        # Python dependencies.
```
## Usage

To use the Feducha Mathesis Algorithm and Branch and Bound optimization, you can modify the test cases or create your own.

#### Running the Algorithm

The main functionality is driven by two key functions:

QMcC_iter(pi_input) — Performs the optimization and minimization of Boolean functions using the Feducha Mathesis algorithm.

BB_tree() — Applies the Branch and Bound technique for optimal solution search.

You can execute any test case by specifying the input variables (in this case, pi_input) and calling the functions accordingly.
``` Python
Example:
from utils.QMcC import QMcC_iter
from utils.branch_and_bound_final import BB_tree

# Input for a specific test case
pi_input = [0, 1, 2, 3, 6, 7, 8, 9, 14, 15]

# Execute the Feducca Mathesis Algorithm
final_pi = QMcC_iter(pi_input).final_pi

# Execute Branch and Bound on the output of Feducca Mathesis
BB = BB_tree(final_pi, pi_input)

print("Solution Found!")
print("Final Equation:", BB.final_equation)
```
#### Expected Output

After running a test case, the algorithm outputs the final minimized Boolean equation, such as:

``` Python
FINAL EQUATION:  F = 'A''B' + 'B''C' + BC
```
