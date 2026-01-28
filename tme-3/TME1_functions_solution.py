from ndlists import ndlist
from typing import List

from math import cos, sin, sqrt, pi

"""
Replace with your name and the one of your working partner, otherwise you won't be evaluated
Respect the syntax "firstname_lastname". If you have several names it is simply "pierre_paul_jacques_dupont"
"""
student_name = ["name1", "name2"]  # Replace with your names


# Paste here all the function you implemented in the TME1 notebook
# Exercise 1
def _ket(lst: list) -> 'ndlist':
    """
    Create a ket from a list.
    :param lst: list of elements
    :return: ndlist representing the ket
    """
    return ndlist([[x] for x in lst])


def _norm(array: ndlist) -> float:
    """
    Compute the norm of a ndlist object.
    :param array: ndlist object
    :return: norm (float)
    """
    return sqrt(sum(x[0]**2 for x in array))


def _scalar_mult(scalar: complex, array: ndlist) -> 'ndlist':
    """
    Multiply a ndlist object by a scalar.
    :param scalar: scalar
    :param array: ndlist
    :return: ndlist object after multiplication
    """
    return ndlist([[scalar * x[0]] for x in array])


def _normalize(array: ndlist) -> 'ndlist':
    """
    Normalize a ndlist object.
    :param array: ndlist object
    :return: normalized ndlist object
    """
    norm = _norm(array)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return _scalar_mult(1 / norm, array)


# Exercise 2
def _zeros(n: int, m: int) -> 'ndlist':
    """
    Create a zero matrix of shape (n, m).
    :param n: number of rows
    :param m: number of columns
    :return: ndlist representing the zero matrix
    """
    return ndlist([[0 for _ in range(m)] for _ in range(n)])


def _identity(n: int) -> 'ndlist':
    """
    Create an identity matrix of shape (n, n).
    :param n: size of the identity matrix
    :return: ndlist representing the identity matrix
    """
    return ndlist([[1 if i == j else 0 for j in range(n)] for i in range(n)])


def _matrix(lvec: List[ndlist]) -> 'ndlist':
    """
    Create a matrix from a list of lists using the zero matrix function.
    :param lvec: list of lists representing the matrix
    :return: ndlist representing the matrix
    """
    if not lvec:
        raise ValueError("The list of vectors is empty.")
    n = len(lvec)
    m = len(lvec[0])
    for vec in lvec:
        if len(vec) != m:
            raise ValueError("All vectors must have the same length.")
    mat = _zeros(n, m)
    for i in range(n):
        for j in range(m):
            mat[i, j] = lvec[i][j, 0]
    return mat


def _matmul(A: ndlist, B: ndlist) -> 'ndlist':
    """
    Perform matrix multiplication A @ B.
    :param A: ndlist representing matrix A
    :param B: ndlist representing matrix B
    :return: ndlist representing the result of A @ B
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for matrix multiplication")

    result = _zeros(A.shape[0], B.shape[1])

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            sum_product = 0
            for k in range(A.shape[1]):
                sum_product += A[i, k] * B[k, j]
            result[i, j] = sum_product

    return result


def _det(matrix: ndlist) -> float:
    """
    Compute the determinant of a square matrix.
    :param matrix: ndlist representing a square matrix
    :return: determinant (float)
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square to compute determinant.")

    # Base case for 2x2 matrix
    if matrix.shape == (2, 2):
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    # Recursive case
    determinant = 0
    for col in range(matrix.shape[1]):
        # Create the minor matrix by excluding the first row and current column
        minor = ndlist([
            [matrix[i, j] for j in range(matrix.shape[1]) if j != col]
            for i in range(1, matrix.shape[0])
        ])
        # Add or subtract the cofactor
        determinant += ((-1) ** col) * matrix[0, col] * _det(minor)

    return determinant


def _transpose(matrix: ndlist) -> 'ndlist':
    """
    Compute the transpose of a matrix.
    :param matrix: ndlist representing a matrix
    :return: ndlist representing the transposed matrix
    """
    n, m = matrix.shape
    transposed = _zeros(m, n)
    for i in range(n):
        for j in range(m):
            transposed[j, i] = matrix[i, j]
    return transposed


def _hermitian(matrix: ndlist) -> 'ndlist':
    """
    Compute the Hermitian conjugate (conjugate transpose) of a matrix.
    :param matrix: ndlist representing a matrix
    :return: ndlist representing the Hermitian conjugate
    """
    n, m = matrix.shape
    hermitian = _zeros(m, n)
    for i in range(n):
        for j in range(m):
            hermitian[j, i] = complex(matrix[i, j]).conjugate()
    return hermitian


# Exercise 3
def is_unitary(matrix: ndlist, tol: float = 1e-9) -> bool:
    """
    Check if a matrix is unitary.
    :param matrix: ndlist object representing a matrix
    :param tol: tolerance level for floating point comparison
    :return: True if unitary, False otherwise
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    identity = _identity(matrix.shape[0])
    product = _matmul(matrix, _hermitian(matrix))
    for i in range(identity.shape[0]):
        for j in range(identity.shape[1]):
            if abs(product[i, j] - identity[i, j]) > tol:
                return False
    return True


def apply_unitary(ket: ndlist, U: ndlist) -> 'ndlist':
    """
    Apply a unitary matrix U to a ket.
    :param ket: ndlist representing the ket
    :param U: ndlist representing the unitary matrix
    :return: ndlist representing the new ket
    """
    if not is_unitary(U):
        raise ValueError("The provided matrix is not unitary.")
    if U.shape[1] != ket.shape[0]:
        raise ValueError("Incompatible shapes for matrix multiplication.")
    return _matmul(U, ket)


# Exercise 4
def bra(ket: ndlist) -> 'ndlist':
    """
    Compute the bra associated to a ket.
    :param ket: ndlist representing the ket
    :return: ndlist representing the bra
    """
    return _hermitian(ket)


# Exercise 5
def _inner(ket1: ndlist, ket2: ndlist) -> complex:
    """
    Compute the inner product between two kets.
    :param ket1: ndlist representing the first ket
    :param ket2: ndlist representing the second ket
    :return: inner product (complex)
    """
    if ket1.shape != ket2.shape:
        raise ValueError("Kets must have the same shape for inner product.")
    bra1 = bra(ket1)
    product = _matmul(bra1, ket2)
    return product[0, 0]


# Exercise 6
def measure_in_basis(ket: ndlist, basis: ndlist) -> List[float]:
    """
    Compute the probabilities of measuring the ket in the given basis.
    :param ket: ndlist representing the ket
    :param basis: ndlist representing the basis as a matrix
    :return: list of probabilities
    """
    if basis.shape[0] != ket.shape[0]:
        raise ValueError("Basis and ket must have compatible shapes.")
    probabilities = []
    for i in range(basis.shape[1]):
        basis_vector = ndlist([[basis[j, i]] for j in range(basis.shape[0])])
        prob = abs(_inner(basis_vector, ket))**2
        probabilities.append(prob)
    total_prob = sum(probabilities)
    if total_prob == 0:
        raise ValueError("Total probability is zero, cannot normalize.")
    return [p / total_prob for p in probabilities]


