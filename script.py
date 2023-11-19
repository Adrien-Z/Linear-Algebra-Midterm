# Linear Algebra Mid-term take-home assignment
# Adrien Zaradez

# Task 1
# (a) Take a square matrix as input and return its inverse without using any additional packages.
# (b) Before calculating the inverse, include a check to ensure that the input matrix is invertible. If the matrix is singular, return an appropriate error message.
def matrix_inverse(input_matrix):
    matrix_size = len(input_matrix) # Get size of matrix

    # Check if matrix is square
    if not all(len(row) == matrix_size for row in input_matrix):
        return "Error: Matrix must be square"

    # Create identity matrix of the same size as the input matrix
    identity_matrix = [[0]*matrix_size for _ in range(matrix_size)]

    # Fill identity matrix with 1s on the diagonal
    for index in range(matrix_size):
        identity_matrix[index][index] = 1

    # Perform Gauss-Jordan elimination
    for pivot_row in range(matrix_size):
        # If the pivot element is zero, swap with a row below that has a non-zero in the same column
        if input_matrix[pivot_row][pivot_row] == 0:
            # If no such row exists, the matrix is singular
            for swap_row in range(pivot_row+1, matrix_size):
                # If a row is found, swap it with the pivot row
                if input_matrix[swap_row][pivot_row] != 0:
                    # Swap rows in both matrices
                    input_matrix[pivot_row], input_matrix[swap_row] = input_matrix[swap_row], input_matrix[pivot_row]
                    identity_matrix[pivot_row], identity_matrix[swap_row] = identity_matrix[swap_row], identity_matrix[pivot_row]
                    break
            # If no row is found, the matrix is singular
            else:
                return "Error: Matrix is singular"
            
        # For each row below the pivot, subtract a multiple of the pivot row such that the pivot column in that row becomes zero - bringing the matrix to row-echelon form
        for target_row in range(pivot_row+1, matrix_size):
            ratio = input_matrix[target_row][pivot_row]/input_matrix[pivot_row][pivot_row]

            # Subtract ratio * pivot row from target row
            for column in range(matrix_size):
                input_matrix[target_row][column] -= ratio * input_matrix[pivot_row][column]
                identity_matrix[target_row][column] -= ratio * identity_matrix[pivot_row][column]

    # Now that the matrix is in row echelon form, we start from the bottom and subtract multiples of each row from the rows above it to get it into reduced row echelon form
    for pivot_row in range(matrix_size-1, -1, -1):
        for target_row in range(pivot_row-1, -1, -1):
            # Subtract ratio * pivot row from target row
            ratio = input_matrix[target_row][pivot_row]/input_matrix[pivot_row][pivot_row]
            for column in range(matrix_size):
                input_matrix[target_row][column] -= ratio * input_matrix[pivot_row][column]
                identity_matrix[target_row][column] -= ratio * identity_matrix[pivot_row][column]

        # Divide each row by its pivot element to make the pivot element 1
        divisor = input_matrix[pivot_row][pivot_row]
        for column in range(matrix_size):
            input_matrix[pivot_row][column] /= divisor
            identity_matrix[pivot_row][column] /= divisor

    # Return the identity matrix which is now the inverse of the original input matrix
    return identity_matrix

# (c) Provide 10 sample square matrices for testing, including invertible and singular matrices. Use your function to calculate the inverses or print out the corresponding messages if those do not exist.
sample_matrices = [
    [[1, 2],
     [3, 4]],  # Invertible

    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 10]],  # Invertible

    [[1, 0],
     [0, 1]],  # Invertible (identity matrix)

    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],  # Singular

    [[2, 3],
     [2, 3]],  # Singular

    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],  # Invertible (identity matrix)

    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 17]],  # Invertible

    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],  # Invertible (identity matrix)

    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]],  # Singular

    [[4, 7], 
     [2, 6]]  # Invertible
]

print("Task 1 (c):")

# Test matrix_inverse function for all sample matrices
for number, matrix in enumerate(sample_matrices):
    print("\nSample matrix " + str(number+1) + ": " + str(matrix))
    print(matrix_inverse(matrix))

# (d) Take a coefficient matrix A and a right-hand side vector b as inputs, calculate the matrix inverse, and use it to solve the system of linear equations Ax = b which is provided by some user. Print out the solution(s).
def solve_linear_system(A, b):
    # Calculate the inverse of matrix A
    A_inversed = matrix_inverse(A)

    # Check if A_inversed is a string, which means it has encountered an error (either matrix is not square or matrix is singular)
    if isinstance(A_inversed, str):
        return A_inversed

    # Create solution vector with zeros
    solution_vector = [0] * len(A_inversed)

    # Multiply A_inversed with vector b to get solution vector (based on if Ax = b, then x = A^-1 * b)
    for row_index in range(len(A_inversed)):
        for col_index in range(len(b)):
            solution_vector[row_index] += A_inversed[row_index][col_index] * b[col_index]

    print(solution_vector)

# Task 2
# (b) Solve this system using your code in Question 1.
A = [[1, -3, -7],
     [-1, 5, 6],
     [-1, 3, 10]]
b = [10, -21, -7]

print("\nTask 2 (b):")
solve_linear_system(A, b)

# Task 3
# Establish whether this matrix is invertible. If M is non-singular, compute its inverse manually or conclude otherwise. Use your code in Question 1 to support your conclusion.
matrix_M = [[1,2,3],
            [0,1,4],
            [5,6,0]]

print("\nTask 3:")
print(matrix_inverse(matrix_M))