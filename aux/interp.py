# -*- coding: utf-8 -*-
""" Definition of auxiliary yates' interpolation functions

Description
-----------
This module implements nodal interpolating functions based on Lagrange interpolants.


References
----------

....

:First added:  2016-04-25
:Last updated: 2016-04-25
:Copyright: Copyright(C) 2014 apalha
:License: GNU GPL version 3 or any later version
"""

__all__ = [
           'lobattoQuad', 'lobattoPoly'
          ]


import numpy

def lobattoQuad(p):
    r"""
    lobattoQuad Returns the p+1 Lobatto points and weights of Gauss-Lobatto
    quadrature.


    For the computation of the nodes it uses a Newton method
    up to machine precision. As initial guess it uses the Chebychev roots.
    for the roots.
    USAGE
    -----
        x, w = lobattoQuad(p)

        Gives the p+1 nodes and weights of the Gauss-Lobatto quadrature of
        order p.

    INPUTS
    ------
        p : (type: int32, size: single value)
            quadrature order


    OUTPUTS
    -------
        x : (type: numpy.array, size: [1, p+1])
            nodes of Gauss-Lobatto quadrature of order p. x \in [-1,1].
        w : (type: numpy.array, size: [1, p+1])
            weights of Gauss-Lobatto quadrature of order p.

    References
    ----------


    TODO
    ----



    :First Added:   2016-04-25
    :Last Modified: 2016-04-25
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version
    """

    n = p+1
    x = numpy.cos(numpy.pi*numpy.arange(0,n,dtype='float64')/p)
    P = numpy.zeros([n,n])
    xold = 2.

    eps = numpy.finfo(float).eps # define machine precision

    # while numpy.abs(x-xold).max() > eps:
    #     xold = x
    #     P[:,0] = 1.0
    #     P[:,1] = x
    #     for k in range(3,n+1):
    #         P[:,k-1] = ((2*(k-1)-1)*x*P[:,k-2] - (k-2)*P[:,k-3])/(k-1.0)
    #
    #     x = xold - (x*P[:,n-1] - P[:,n-2])/(n*P[:,n-1])
    #
    # w = 2.0/(p*n*(P[:,n-1])**2)

    while numpy.abs(x-xold).max() > eps:
        xold = x
        P[0,:] = 1.0
        P[1,:] = x
        for k in range(3,n+1):
            P[k-1,:] = ((2*(k-1)-1)*x*P[k-2,:] - (k-2)*P[k-3,:])/(k-1.0)

        x = xold - (x*P[n-1,:] - P[n-2,:])/(n*P[n-1,:])

    x = numpy.flipud(x)
    w = 2.0/(p*n*(P[n-1,:])**2)


    return x,w


def lobattoPoly(x,p):
    r"""
    lobattoPoly Returns the p+1 Lobatto Lagrange interpolant basis functions,
    evaluated at x.

    It returns a 2-dimensional matrix with the values of the Lobatto basis
    interpolants evaluated at x.

    If x is a vector of length N it returns a 2d matrix whose rows are the
    values of the evaluated polynomial, P(x), in x:
                  -                                               -
        result = | P_{1}(x(1))   P_{1}(x(2))   ...   P_{1}(x(N))   |
                 | P_{2}(x(1))   P_{2}(x(2))   ...   P_{2}(x(N))   |
                 |                   ...                           |
                 | P_{p+1}(x(1)) P_{p+1}(x(2)) ...   P_{p+1}(x(N)) |
                  -                                               -

        If x=[] then it computes the Lobatto polynomial basis at the Lobatto nodes,
        that is, the result is a sparse identity matrix.

    USAGE
    -----
        result = lobattoPoly(x,p)

        Gives the p+1 Lobatto basis interpolants evaluated at x.

    INPUTS
    ------
    x : numpy.array, size: [N,1] or [1,N])
        Locations where to evaluate the polynomial basis functions.
        x \in [-1,1].
    p : int32, size: single value
        The degree of the polynomial basis functions.

    OUTPUTS
    -------
    result : numpy.array, size: [p+1, N])
             The (p+1) polynomials evaluated at the x points.



    References
    ----------


    TODO
    ----


    :First Added:   2016-04-25
    :Last Modified: 2016-04-25
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version
    """

    # allocate memory space for the result
    result = numpy.ones([x.size,p+1])

    # compute lobatto roots
    roots,weights = lobattoQuad(p)

    # Compute each polynomial n using the formula:
    # \frac{\prod_{i=1\\i\neq n}^{p+1}(x-r_{i})}{\prod_{i=1\\i\neq
    # j}^{p+1}(r_{n}-r_{i})}
    #
    # For the top product one uses the built in function poly, based upon
    # the roots. For the bottom part, one just computed the product.

    # fast implementation using Carlo Castoldi's code:
    # http://www.mathworks.com/matlabcentral/fileexchange/899-lagrange-polynomial-interpolation
    for i in range(p+1):
        for j in range(p+1):
            if (i != j):
                result[:,i] = result[:,i]*(x.ravel()-roots[j])/(roots[i]-roots[j])

    return result.reshape(numpy.hstack((x.shape,p+1)))