from ndlists import ndlist, slicelen
from typing import List

from math import cos, sin, sqrt, pi

"""
Replace with your name and the one of your working partner, otherwise you won't be evaluated
Respect the syntax "firstname_lastname"
"""
student_name = ["alejandro_jackson", "katlyn_prophet"]


# Paste here all the function you implemented in the TME1 notebook
# Exercise 1
def _ket(lst: list) -> 'ndlist':
    """
    Create a ket from a list.
    :param lst: list of elements
    :return: ndlist representing the ket
    """
    # A ket is a ndarray object of shape $(n, 1)$, where $n > 1$, to avoid ambiguity
    assert len(lst) > 1, "A ket must have more than one element"
    # Assert that it is not a list of lists for example [[0,1]] is not a ket
    assert not any(isinstance(item, list) for item in lst), "A ket must not be a list of lists"
    return ndlist(lst)


def _norm(array: ndlist) -> float:
    """
    Compute the norm of a ndlist object.
    :param array: ndlist object
    :return: norm (float)
    """
    # In ndlist we have three functions, int2slice, slicelen, and ndslicenormalize.
    # int2slice is used to convert an int to a slice
    # slicelen is used to get the length of a slice
    # ndslicenormalize is used to normalize a slice
    # We can use these functions to compute the norm of a ndlist object.
    # The norm of a ndlist object is the square root of the sum of the squares of the elements of the ndlist object.
    # We can use the ndslicenormalize function to get the length of the ndlist object.

    
    # The norm of a vector $v$ is defined as $\|v\| = \sqrt{\sum_i |v_i|^2}$.
    return sqrt(sum(abs(element)**2 for element in array))
    


def _scalar_mult(scalar: complex, array: ndlist) -> 'ndlist':
    """
    Multiply a ndlist object by a scalar.
    :param scalar: scalar
    :param array: ndlist
    :return: ndlist object after multiplication
    """
    # Support vectors and matrices; multiply element-wise and preserve shape
    if len(array) > 0 and isinstance(array[0], list):
        return ndlist([[scalar * array[i][j] for j in range(len(array[0]))] for i in range(len(array))])
    return ndlist([scalar * element for element in array])


def _normalize(array: ndlist) -> 'ndlist':
    """
    Normalize a ndlist object.
    :param array: ndlist object
    :return: normalized ndlist object
    """
    # Use _norm and _scalar_mult to normalize a ndlist object.
    return _scalar_mult(1/_norm(array), array)


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
    return _zeros(len(lvec), len(lvec[0]))


def _matmul(A: ndlist, B: ndlist) -> 'ndlist':
    """
    Perform matrix multiplication A @ B.
    :param A: ndlist representing matrix A
    :param B: ndlist representing matrix B
    :return: ndlist representing the result of A @ B
    """
    # Handle matrix-vector multiplication (B is a vector)
    if len(B) > 0 and not isinstance(B[0], list):
        return ndlist([sum(A[i][k] * B[k] for k in range(len(B))) for i in range(len(A))])
    # Handle matrix-matrix multiplication
    return ndlist([[sum(A[i][k] * B[k][j] for k in range(len(B[0]))) for j in range(len(B[0]))] for i in range(len(A))])


def _det(matrix: ndlist) -> float:
    """
    Compute the determinant of a square matrix.
    :param matrix: ndlist representing a square matrix
    :return: determinant (float)
    """
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _transpose(matrix: ndlist) -> 'ndlist':
    """
    Compute the transpose of a matrix.
    :param matrix: ndlist representing a matrix
    :return: ndlist representing the transposed matrix
    """
    # The transpose of a matrix is a matrix that is obtained by swapping 
    # the rows and columns of the original matrix.
    # We can use a nested list comprehension to transpose the matrix.
    # The first list comprehension iterates over the columns of the original matrix,
    # and the second list comprehension iterates over the rows of the original matrix.
    # We can use the range function to iterate over the rows and columns of the original matrix.
    # We can use the len function to get the length of the rows and columns of the original matrix.
    # We can use the ndlist function to create a new matrix.
    # We can use the conjugate function to get the conjugate of a complex number.
    
    return ndlist([[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))])


def _hermitian(matrix: ndlist) -> 'ndlist':
    """
    Compute the Hermitian conjugate (conjugate transpose) of a matrix.
    :param matrix: ndlist representing a matrix
    :return: ndlist representing the Hermitian conjugate
    """
    # Handle vectors (1D) and matrices (2D)
    if len(matrix) > 0 and not isinstance(matrix[0], list):
        # For vectors, return the conjugate transpose (row vector)
        return ndlist([[element.conjugate() for element in matrix]])
    # For matrices, transpose then conjugate each element
    return _transpose(ndlist([[element.conjugate() for element in row] for row in matrix]))

# Exercise 3
def is_unitary(matrix: ndlist, tol: float = 1e-9) -> bool:
    """
    Check if a matrix is unitary.
    :param matrix: ndlist object representing a matrix
    :param tol: tolerance level for floating point comparison
    :return: True if unitary, False otherwise
    """
    # A matrix is unitary if its Hermitian conjugate times itself is equal to 
    # the identity matrix.
    
    # Use a tolerance-based comparison: compute (U^† U) element-wise and
    # check it is close to the identity within tol.
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            # (U^† U)_{ij} = sum_k conj(U_{k i}) * U_{k j}
            acc = 0
            for k in range(n):
                acc += matrix[k][i].conjugate() * matrix[k][j]
            expected = 1 if i == j else 0
            if abs(acc - expected) > tol:
                return False
    return True


def apply_unitary(ket: ndlist, U: ndlist) -> 'ndlist':
    """
    Apply a unitary matrix U to a ket.
    :param ket: ndlist representing the ket
    :param U: ndlist representing the unitary matrix
    :return: ndlist representing the new ket
    """
    # Multiply the ket by the unitary matrix.
    return _matmul(U, ket)


# Exercise 4
def bra(ket: ndlist) -> 'ndlist':
    """
    Compute the bra associated to a ket.
    :param ket: ndlist representing the ket
    :return: ndlist representing the bra
    """
    # The bra is the Hermitian conjugate of the ket.
    return _hermitian(ket)


# Exercise 5
def _inner(ket1: ndlist, ket2: ndlist) -> complex:
    """
    Compute the inner product between two kets.
    :param ket1: ndlist representing the first ket
    :param ket2: ndlist representing the second ket
    :return: inner product (complex)
    """
    # The inner product is the sum of the products of the elements of the two kets.
    return sum(ket1[i] * ket2[i].conjugate() for i in range(len(ket1)))


# Exercise 6
def measure_in_basis(ket: ndlist, basis: ndlist) -> List[float]:
    """
    Compute the probabilities of measuring the ket in the given basis.
    :param ket: ndlist representing the ket
    :param basis: ndlist representing the basis as a matrix
    :return: list of probabilities
    """
    # The probabilities of measuring the ket in the given basis is the square of the inner product of the ket and the basis.
    return [abs(ket[i])**2 for i in range(len(ket))]


