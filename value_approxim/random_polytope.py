import cdd
import numpy as np

ftype = np.float32

def random_simplex_sum1( N, dim ):
    """
    generates a random matrix of N rows. and each row has dimension dim
    :param N: number of rows in matrix
    :param dim: each row vector size
    :return: a matrix with N rows s.t. sum of each vector is equal 1 and all their memebers are random scalers between
    0 and 1
    """
    """ N uniform-random points >= 0, sum x_i == 1 """
    X = np.random.exponential( size=(N,dim) )
    X /= X.sum(axis=1)[:,np.newaxis]
    return X

# def random_simplex_le1( N, dim ):
#     """ N uniform-random points >= 0, sum x_i <= 1 """
#     return random_simplex_sum1( N, dim ) \
#         * (np.random.uniform( size=N ) ** (1/dim)) [:,np.newaxis]

def random_point(vertices_number):
    """
    it generates a random point inside the dim dimensional cube
    :param dim: space dimension
    :return:
    """
    coef= random_simplex_sum1(1, vertices_number)
    return coef

#TODO be sure about randomly distribution of point in polytope. we can plot it later.
def random_point_polytop(IneqList):
    """
    it receives list of polytope constraints and return back a random point inside poytope
    :param IneqList: list of arrays as inequalities. each array has dimension d+1
    :return: return a random point (vector of d dimension) inside polytope
    """

    #it is a package for genrating polytope from their inequalities and getting their vertices afterward.
    mat = cdd.Matrix(IneqList, number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)

    #list of poltope vertices
    vert = poly.get_generators()
    len_vertices = len(vert)

    vertices = []
    for i in range(len_vertices):
        vertices.append(np.array(vert[i][1:], dtype= ftype) )

    #generate random coefficients between 0 and 1 such that sum of all coeffients is equal 1
    coef = random_point(len_vertices)[0]
    return sum([coef[i]*vertices[i] for i in range(len_vertices)])