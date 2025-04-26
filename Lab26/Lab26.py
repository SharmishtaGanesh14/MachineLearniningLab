import math


# Solve quadratic equation ax^2 + bx + c = 0
def solve_quadratic(a, b, c):
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_d = math.sqrt(discriminant)
    lambda1 = (-b + sqrt_d) / (2 * a)
    lambda2 = (-b - sqrt_d) / (2 * a)
    return lambda1, lambda2


# Function to multiply a matrix with a vector manually
def matrix_vector_multiply(matrix, vector):
    result = []
    for i in range(len(matrix)):
        row_sum = 0
        for j in range(len(matrix[i])):
            row_sum += matrix[i][j] * vector[j]
        result.append(row_sum)
    return result


# Function to compute the quadratic form XᵀAX manually
def quadratic_form(matrix, vector):
    # Step 1: Calculate A * X manually (Matrix-vector multiplication)
    A_X = matrix_vector_multiply(matrix, vector)

    # Step 2: Calculate Xᵀ * (A * X) manually (Dot product)
    result = 0
    for i in range(len(vector)):
        result += vector[i] * A_X[i]
    return result


# Function to check positive definiteness of a matrix using XᵀAX
def check_positive_definiteness(matrix):
    # Define a random non-zero vector X (just an example, could be any non-zero vector)
    X = [1, 1]  # Example non-zero vector

    # Step 3: Compute XᵀAX
    result = quadratic_form(matrix, X)

    # Step 4: Check the value of XᵀAX
    if result > 0:
        return "The matrix is Positive Definite"
    elif result == 0:
        return "The matrix is Positive Semi-Definite"
    else:
        return "The matrix is Negative Definite"


# Check positive definiteness
def is_positive_definite(matrix):
    a11 = matrix[0][0]
    a22 = matrix[1][1]
    det = a11 * a22 - matrix[0][1] * matrix[1][0]
    print(f"First minor: {a11}, Determinant: {det}")
    if a11 > 0 and det > 0:
        return True
    else:
        return False


# Find eigenvalues of 2x2 matrix
def find_eigenvalues(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    # characteristic equation: (a-λ)(d-λ) - bc = 0
    A = 1
    B = -(a + d)
    C = (a * d - b * c)
    eigenvalues = solve_quadratic(A, B, C)
    return eigenvalues


# Evaluate Hessian for f(x,y) = x^3 + 2y^3 - xy
def hessian_f1(x, y):
    H = [
        [6 * x, -1],
        [-1, 12 * y]
    ]
    return H


# Evaluate Hessian for f(x,y) = 4x + 2y - x^2 - 3y^2
def hessian_f2():
    H = [
        [-2, 0],
        [0, -6]
    ]
    return H


if __name__ == "__main__":

    # 1. Check if matrix A is positive definite
    A = [[9, -15], [-15, 21]]
    if is_positive_definite(A):
        print("Matrix A is positive definite.")
    else:
        print("Matrix A is NOT positive definite (indefinite).")

    # 2. Check positive definiteness using quadratic form XᵀAX
    result = check_positive_definiteness(A)
    print(result)

    # 3. Eigenvalues of Hessian at (3,1)
    H = [[108, -1], [-1, 2]]
    eigenvalues = find_eigenvalues(H)
    print(f"Eigenvalues: {eigenvalues}")

    # 4. Concavity of f(x,y) = x^3 + 2y^3 - xy at different points
    print("\n3. Concavity of f(x,y) = x^3 + 2y^3 - xy")

    # (i) At (0,0)
    print("\nAt (0,0):")
    H0 = hessian_f1(0, 0)
    eig0 = find_eigenvalues(H0)
    print(f"Eigenvalues: {eig0}")
    if eig0[0] * eig0[1] < 0:
        print("Saddle point.")
    elif eig0[0] > 0:
        print("Local minimum.")
    else:
        print("Local maximum.")

    # (ii) At (3,3)
    print("\nAt (3,3):")
    H1 = hessian_f1(3, 3)
    eig1 = find_eigenvalues(H1)
    print(f"Eigenvalues: {eig1}")
    if eig1[0] > 0 and eig1[1] > 0:
        print("Local minimum.")
    elif eig1[0] < 0 and eig1[1] < 0:
        print("Local maximum.")
    else:
        print("Saddle point.")

    # (iii) At (3,-3)
    print("\nAt (3,-3):")
    H2 = hessian_f1(3, -3)
    eig2 = find_eigenvalues(H2)
    print(f"Eigenvalues: {eig2}")
    if eig2[0] > 0 and eig2[1] > 0:
        print("Local minimum.")
    elif eig2[0] < 0 and eig2[1] < 0:
        print("Local maximum.")
    else:
        print("Saddle point.")

    # 5. Find critical points and classify for f(x,y) = 4x + 2y - x^2 - 3y^2

    # Find critical points by solving gradient = 0
    # fx = 4 - 2x = 0 -> x = 2
    # fy = 2 - 6y = 0 -> y = 1/3
    x_crit = 2
    y_crit = 1 / 3
    print(f"\nCritical point at: ({x_crit}, {y_crit})")

    # Find Hessian and its eigenvalues
    H3 = hessian_f2()
    eig3 = find_eigenvalues(H3)
    print(f"Eigenvalues: {eig3}")
    if eig3[0] > 0 and eig3[1] > 0:
        print("Local minimum.")
    elif eig3[0] < 0 and eig3[1] < 0:
        print("Local maximum.")
    else:
        print("Saddle point.")

    # If all eigenvalues are positive, the matrix is Positive Definite.
    # If all eigenvalues are non-negative (including zeros), the matrix is Positive Semi-Definite.
    # If all eigenvalues are negative, the matrix is Negative Definite. If all eigenvalues are non-positive (including zeros), the matrix is Negative Semi-Definite.
    # If the matrix has both positive and negative eigenvalues, it is Indefinite.

    # A matrix that is Positive Definite corresponds to a strictly convex function, while
    # a Positive Semi-Definite matrix represents a convex function (but not strictly).
    # A Negative Definite matrix corresponds to a strictly concave function, and
    # a Negative Semi-Definite matrix represents a concave function (but not strictly).
    # An Indefinite matrix indicates that the function is neither convex nor concave.