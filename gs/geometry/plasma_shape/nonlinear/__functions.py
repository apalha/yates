# -*- coding: utf-8 -*-
""" Boundary definition functions for known Soloviev analytical solutions to the Grad-Shafranov equation

Description
-----------
This module implements several functions to compute the boundary of Soloviev solutions to the
linear Grad-Shafranov equation.


References
----------
.. [pataki] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., O'Neil, M. (2013).
                    A fast, high-order solver for the Grad-Shafranov equation.
                    Journal of Computational Physics, 243, 28-45. doi:10.1016/j.jcp.2013.02.045


....

:First added:  2016-04-21
:Last updated: 2016-04-21
:Copyright: Copyright(C) 2015 apalha
:License: GNU GPL version 3 or any later version
"""



"""
Reviews
-------
1. First implementation. (apalha, 2016-04-26)

"""

__all__ = [
           'pataki', 'palha_x_iter'
          ]


import numpy


# -- Plasma boundary ---------------------------------------------------------------------------------------------------
#                                                                                                  <editor-fold desc="">


def pataki(n,epsilon,kappa,delta):
    r"""
    Function that returns a set of n points that defines the boundary of
    the pataki plasma shape test case [1].

    Usage
    -----
    .. code-block :: python

        pataki(n,epsilon,kappa,delta)


    Parameters
    ----------
    n : int
        The number of points on the boundary to generate.
    epsilon : float64
               As given in [1], related to the size of the horizontal axis of the plasma shape.
    kappa : float64
             As given in [1], related to the size of the vertical axis of the plasma shape (kappa*epsilon).
    delta : float64
            As given in [1], related to the misalignment between the top of the plasma shape and the  axis.
            of the plasma shape.


    Returns
    -------
    r : numpy.array(float64), [1,n]
        The r coordinates of the points in the boundary of the plasma.

    z : numpy.array(float64), [1,n]
        The z coordinates of the points in the boundary of the plasma.


    REFERENCES
    ----------
    .. [pataki2013] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., and O'Neil, M. (2013).
                    A fast, high-order solver for the Grad-Shafranov equation.
                    Journal of Computational Physics, 243, 28-45 doi:10.1016/j.jcp.2013.02.045


    :First Added:   2015-12-01
    :Last Modified: 2015-12-01
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2015-12-01)
    """

    alpha = numpy.arcsin(delta) # the alpha term appearing in [pataki2013]

    xi = numpy.linspace(0,2.0*numpy.pi,n) # the points along the angle-like variable, which will parametrise the points
                                          # on the boundary

    r = 1.0 + epsilon*numpy.cos(xi + alpha*numpy.sin(xi))
    z = epsilon*kappa*numpy.sin(xi)

    return r, z


def palha_x_iter(n):
    r"""
    Function that returns a set of n points that defines the boundary of
    the x-point plasma shape test case [palha2015].

    Usage
    -----
    .. code-block :: python

        palha_x_iter(n)


    Parameters
    ----------
    n : int
        The number of points to generate on the boundary.

    Returns
    -------
    r : numpy.array(float64), [1,n]
        The r coordinates of the points in the boundary of the plasma.

    z : numpy.array(float64), [1,n]
        The z coordinates of the points in the boundary of the plasma.


    REFERENCES
    ----------
    .. [palha2015] A. Palha, B. Koren, and F. Felici, “A mimetic spectral element solver for the Grad-Shafranov
                   equation on curved meshes,” http://arxiv.org/abs/1512.05989,
                   submitted to Journal of Computational Physics, 2015.


    :First Added:   2016-01-06
    :Last Modified: 2016-01-06
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-01-06)
    """

    # The boundary of the x point plasma is made up of fours segments parametrized from [0,1]. For this reason
    # we take the interval [0,4] and distribute the n points inside this interval. The points in [0,1] are generated
    # using the first segment, the ones from ]1,2] are generated using the second line segment, the ones from ]2,3]
    # are generated using the third line segment, and the points from ]3,4] are generated using the fourth line segment

    # generate the points from [0,4]
    s = numpy.linspace(0,4,n)

    # allocate memory space for the point's coordinates
    r = numpy.zeros(n)
    z = numpy.zeros(n)

    # segment 1
    s_segment_1 = s[s <= 1] # no rescalling needed because s_segment is already in [0,1]
    n_segment_1 = s_segment_1.size
    cx = numpy.fliplr(numpy.array([[0.88, 0.5521148541571086, -0.20640336053451946, 2.4834931938984552, -5.03596728562579, 2.825936559606835, 0.32138363979983614, -0.5005576013019241]]))[0]
    cy = numpy.fliplr(numpy.array([[-0.6, 0.5999999999999996, 0.0, 0.0, 0.0, 0.0, 0.0]]))[0]

    r[0:n_segment_1] = numpy.polyval(cx,s_segment_1)
    z[0:n_segment_1] = numpy.polyval(cy,s_segment_1)

    # segment 2
    s_segment_2 = s[numpy.logical_and((s > 1), (s <= 2))] - 1.0 # -1.0 so that s_segment_2 lies in [0,1]
    n_segment_2 = s_segment_2.size
    cx = numpy.fliplr(numpy.array([[1.3200000000000014, 0., -0.42243831504242935, 1.2501879617549474, -5.368764301947667, 11.182100647055545, -11.580478496134559, 4.499392504314161]]))[0]
    cy = numpy.fliplr(numpy.array([[0., 0.8891881373291781, -4.194429431647997, 22.70616098798567, -59.94304165245803, 82.42892663380348, -56.412807237548165, 15.069366375444783]]))[0]

    r[n_segment_1:(n_segment_2+n_segment_1)] = numpy.polyval(cx,s_segment_2)
    z[n_segment_1:(n_segment_2+n_segment_1)] = numpy.polyval(cy,s_segment_2)

    # segment 3
    s_segment_3 = s[numpy.logical_and((s > 2), (s <= 3))] - 2.0 # -2.0 so that s_segment_3 lies in [0,1]
    s_segment_3 = 1.0 - s_segment_3 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_3 = s_segment_3.size
    cx = numpy.fliplr(numpy.array([[0.6799999999999871, 0., 0.16248409671487218, -1.14462836550234, 5.240947487862182, -11.313789484633782, 11.659292015452298,-4.404305749893218]]))[0]
    cy = numpy.fliplr(numpy.array([[0.0, 0.8891881373291746, -4.194429431648004, 22.706160987985644, -59.943041652457914, 82.42892663380331, -56.412807237548044, 15.06936637544475]]))[0]

    r[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.polyval(cx,s_segment_3)
    z[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.polyval(cy,s_segment_3)

    # segment 4
    s_segment_4 = s[s > 3] - 3.0 # -3.0 so that s_segment_4 lies in [0,1]
    s_segment_4 = 1.0 - s_segment_4 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_4 = s_segment_4.size
    cx = numpy.fliplr(numpy.array([[0.88, -0.36475148713106564, 0.15669462964640002, -0.9082699105455433, 2.8740467630422044, -3.379335261756935, 1.7746456511237334, -0.35303038437880685]]))[0]
    cy = numpy.fliplr(numpy.array([[-0.6, 0.5999999999999993, 0., 0., 0., 0., 0.]]))[0]

    r[(n_segment_3+n_segment_2+n_segment_1):] = numpy.polyval(cx,s_segment_4)
    z[(n_segment_3+n_segment_2+n_segment_1):] = numpy.polyval(cy,s_segment_4)


    return r, z


#    ---------------------------------------------------------------------------------------------------- </editor-fold>


