# -*- coding: utf-8 -*-
""" Analytical Soloviev solutions to the linear Grad-Shafranov equation

Description
-----------
This module implements several known analytical Soloviev solutions to the linear Grad-Shafranov equation.


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

from traits.trait_types import self

"""
Reviews
-------
1. First implementation. (apalha, 2016-04-21)

"""

__all__ = [
           'pataki','pataki_eval_params',\
           'cerfon','cerfon_eval_params'
          ]



import numpy

from .... import aux


def pataki(r,z,epsilon=None,kappa=None,delta=None,d=None):
    r"""
    Returns the values of the magnetic flux at the points
    (r,z) for the soloviev solution in [pataki]_. The models for :math:`p(\psi)` and :math:`f(\psi)` are:

    .. math::

        p = C\psi + P_{0} \quad \mathrm{and} \quad f = f_{0}\,.

    With :math:`C = 1`.

    Usage
    -----
    .. code-block :: python

        pataki(r,z,epsilon=epsilon,kappa=kappa,delta=delta)
        or
        pataki(r,z,d=d)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].
    epsilon : float64, size: single value
              as given in [pataki]_, related to the size of the horizontal
              axis of the plasma shape.
    kappa : float64, size: single value
            as given in [pataki]_, related to the size of the vertical
            axis of the plasma shape (kappa*epsilon).
    delta : float64, size: single value
            as given in [pataki]_, related to the misalignment between the
            top of the plasma shape and the axis of the plasma shape.
    d : numpy.array, size: 3
        The three coefficients associated to the weights of each of the particular
        solutions of the linear Grad-Shafranov equation, as given in [pataki]_.

    Returns
    -------
    psi : numpy.array, size: [N,M]
          The computation of the magnetic flux at the points (r,z).


    References
    ----------
    .. [pataki] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., O'Neil, M. (2013).
                A fast, high-order solver for the Grad-Shafranov equation.
                Journal of Computational Physics, 243, 28-45. doi:10.1016/j.jcp.2013.02.045

    :First Added:   2016-04-19
    :Last Modified: 2015-04-21
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-19)
    2. Changed input parameters to allow calling the function with epsilon,kappa and delta
       or simply with pre-computed d coefficients. (apalha, 2016-04-21)
    """

    if ((epsilon is None) or (kappa is None) or (delta is None)) and (d is None):
        raise aux.errors.InputError('pataki','Missing inputs. Either epsilon,kappa,delta must be given as input or d must be given as input.')

    if d is None:
        # compute parameters
        d = pataki_eval_params(epsilon,kappa,delta)


    # compute psi
    psi = ((r**4)/8.0) + d[0] + d[1]*(r**2) + d[2]*((r**4) - 4.0*(r**2)*(z**2))

    return psi


def pataki_eval_params(epsilon,kappa,delta):
    r"""
    Returns the parameters that define the parameters of the soloviev solution in [pataki]_.
    The models for :math:`p(\psi)` and :math:`f(\psi)` are:

    .. math::

        p = C\psi + P_{0} \quad \mathrm{and} \quad f = f_{0}\,.

    With :math:`C = 1`.

    Usage
    -----
    .. code-block :: python

        pataki_eval_params(epsilon,kappa,delta)


    Parameters
    ----------
    epsilon : float64, size: single value
              as given in [pataki]_, related to the size of the horizontal
              axis of the plasma shape.
    kappa : float64, size: single value
            as given in [pataki]_, related to the size of the vertical
            axis of the plasma shape (kappa*epsilon).
    delta : float64, size: single value
            as given in [pataki]_, related to the misalignment between the
            top of the plasma shape and the axis of the plasma shape.

    Returns
    -------
    d : numpy.array, size: 3
        The three coefficients associated to the weights of each of the particular
        solutions of the linear Grad-Shafranov equation, as given in [pataki_].


    References
    ----------
    .. [pataki] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., O'Neil, M. (2013).
                A fast, high-order solver for the Grad-Shafranov equation.
                Journal of Computational Physics, 243, 28-45. doi:10.1016/j.jcp.2013.02.045

    :First Added:   2016-04-21
    :Last Modified: 2015-04-21
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-21)
    """

    # compute parameters

    # convert the epsilon,kappa,delta parameters into d1,d2,d3 parameters
    # see [pataki] for a full explanation
    M = numpy.array([[1.0, (1.0 + epsilon)**2, (1.0 + epsilon)**4],
                     [1.0, (1.0 - epsilon)**2, (1.0 - epsilon)**4],
                     [1.0, (1.0 - delta*epsilon)**2, ((1.0 - (delta*epsilon))**4)
                          - 4.0*((1.0 - delta*epsilon)**2) * (kappa**2) * (epsilon**2)]])

    b = -(1.0/8.0)*numpy.array([(1.0 + epsilon)**4, (1.0 - epsilon)**4, (1.0 - delta*epsilon)**4])

    # compute the parameters
    d = numpy.linalg.solve(M,b)

    return d



def cerfon(r,z,A,epsilon=None,kappa=None,delta=None,rsep=None,zsep=None,d=None):
    r"""
    Returns the values of the magnetic flux at the points
    (r,z) for the Soloviev solution in [cerfon]_. The models for :math:`p(\psi)` and :math:`f(\psi)` are:

    .. math::

        p = (1-A)\psi \quad \mathrm{and} \quad f = \sqrt(2\mu_{0}A\psi)\,.

    This Soloviev solution contains an X-point at the point (rsep,zsep) when these coordinates are specified.

    Usage
    -----
    .. code-block :: python

        cerfon(r,z,A,epsilon=epsilon,kappa=kappa,delta=delta,rsep=rsep,zsep=zsep)
        or
        cerfon(r,z,A,d=d)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].
    A : float64, size: single value
        Related to the profile, as given in [cerfon]_.
    epsilon : float64, size: single value
              as given in [pataki]_, related to the size of the horizontal
              axis of the plasma shape.
    kappa : float64, size: single value
            as given in [pataki]_, related to the size of the vertical
            axis of the plasma shape (kappa*epsilon).
    delta : float64, size: single value
            as given in [pataki]_, related to the misalignment between the
            top of the plasma shape and the axis of the plasma shape.
    rsep : float64, size: single value
           The r-coordinate of the X-point, as given in [cerfon]_.
    zsep : float64, size: single value
           The z-coordinate of the X-point, as given in [cerfon]_.
    d : numpy.array, size: [1,7] (no X-point) or [1,12] (with X-point)
        The coefficients associated to the weights of each of the particular
        solutions of the linear Grad-Shafranov equation, as given in [cerfon]_.

    Returns
    -------
    psi : numpy.array, size: [N,M]
          The computation of the magnetic flux at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    2. Changed input parameters to allow calling the function with epsilon,kappa and delta
       or simply with pre-computed d coefficients. (apalha, 2016-04-21)
    """

    if ((epsilon is None) or (kappa is None) or (delta is None) or (rsep is None) or (zsep is None)) and (d is None):
        raise aux.errors.InputError('cerfon','Missing inputs. Either epsilon,kappa,delta,rsep,zsep must be given as input or d must be given as input.')

    if d is None:
        # compute parameters
        d = cerfon_eval_params(A,epsilon,kappa,delta,rsep,zsep)


    # compute psi
    # psi is the linear combination of the particular solution and the homogeneous solutions
    # \psi = \psi_{p} + \sum_{i=1} d_{i} \psi_{h}^{i}
    try:
        d_shape = numpy.hstack((12,numpy.ones(len(r.shape))))
    except AttributeError:
        d_shape = numpy.array([12,1])

    psi = cerfon_psi_p(r,z,A) + (d.reshape(d_shape)*cerfon_psi_h(r,z)).sum(0)

    return psi


def cerfon_eval_params(A,epsilon,kappa,delta,rsep,zsep):
    r"""
    Returns the parameters that define the magnetic flux at the points
    (r,z) for the Soloviev solution in [cerfon]_. The models for :math:`p(\psi)` and :math:`f(\psi)` are:

    .. math::

        p = (1-A)\psi \quad \mathrm{and} \quad f = \sqrt(2\mu_{0}A\psi)\,.

    This Soloviev solution contains an X-point at the point (rsep,zsep) when these coordinates are specified.

    Usage
    -----
    .. code-block :: python

        cerfon_eval_params(A,epsilon,kappa,delta,rsep,zsep)


    Parameters
    ----------
    A : float64, size: single value
        Related to the profile, as given in [cerfon]_.
    epsilon : float64, size: single value
              as given in [pataki]_, related to the size of the horizontal
              axis of the plasma shape.
    kappa : float64, size: single value
            as given in [pataki]_, related to the size of the vertical
            axis of the plasma shape (kappa*epsilon).
    delta : float64, size: single value
            as given in [pataki]_, related to the misalignment between the
            top of the plasma shape and the axis of the plasma shape.
    rsep : float64, size: single value
           The r-coordinate of the X-point, as given in [cerfon]_.
    zsep : float64, size: single value
           The z-coordinate of the X-point, as given in [cerfon]_.

    Returns
    -------
    d : numpy.array, size: 7 (no X-point) or 12 (with an X-point)
        The coefficients associated to the weights of each of the particular
        solutions of the linear Grad-Shafranov equation, as given in [cerfon]_.


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    # convert the A,epsilon,kappa,delta parameters,rsep,zsep into the parameters d
    # see [cerfon]_ for a full explanation

    M1 = numpy.vstack((cerfon_psi_h(1.0+epsilon,0.0).flatten(),\
                      cerfon_psi_h(1.0-epsilon,0.0).flatten(),\
                      cerfon_psi_h(1.0-delta*epsilon,kappa*epsilon).flatten(),\
                      cerfon_psi_h(rsep,zsep).flatten(),\
                      cerfon_psi_h_ddz(1.0+epsilon,0.0).flatten(),\
                      cerfon_psi_h_ddz(1.0-epsilon,0.0).flatten(),\
                      cerfon_psi_h_ddr(1.0-delta*epsilon,kappa*epsilon).flatten(),\
                      cerfon_psi_h_ddr(rsep,zsep).flatten(),\
                      cerfon_psi_h_ddz(rsep,zsep).flatten(),\
                      cerfon_psi_h_ddz2(1.0+epsilon,0.0).flatten(),\
                      cerfon_psi_h_ddz2(1.0-epsilon,0.0).flatten(),\
                      cerfon_psi_h_ddr2(1.0-delta*epsilon,kappa*epsilon).flatten()))

    n1 = -((1.0+numpy.arcsin(delta))**2)/(epsilon*kappa*kappa)
    n2 = ((1.0-numpy.arcsin(delta))**2)/(epsilon*kappa*kappa)
    n3 = -kappa/(epsilon*numpy.cos(numpy.arcsin(delta))*numpy.cos(numpy.arcsin(delta)))

    M2 = numpy.zeros([12,12])
    M2[9,:] = -n1*cerfon_psi_h_ddr(1.0+epsilon,0.0).flatten()
    M2[10,:] = -n2*cerfon_psi_h_ddr(1.0-epsilon,0.0).flatten()
    M2[11,:] = -n3*cerfon_psi_h_ddz(1.0-delta*epsilon,kappa*epsilon).flatten()

    b1 = -numpy.array([cerfon_psi_p(1.0+epsilon,0.0,A),\
                       cerfon_psi_p(1.0-epsilon,0.0,A),\
                       cerfon_psi_p(1.0-delta*epsilon,kappa*epsilon,A),\
                       cerfon_psi_p(rsep,zsep,A),\
                       cerfon_psi_p_ddz(1.0+epsilon,0.0,A),\
                       cerfon_psi_p_ddz(1.0-epsilon,0.0,A),\
                       cerfon_psi_p_ddr(1.0-delta*epsilon,kappa*epsilon,A),\
                       cerfon_psi_p_ddr(rsep,zsep,A),\
                       cerfon_psi_p_ddz(rsep,zsep,A),\
                       cerfon_psi_p_ddz2(1.0+epsilon,0.0,A),\
                       cerfon_psi_p_ddz2(1.0-epsilon,0.0,A),\
                       cerfon_psi_p_ddr2(1.0-delta*epsilon,kappa*epsilon,A)]);

    b2 = numpy.zeros(12)
    b2[9] = -n1*cerfon_psi_p_ddr(1.0+epsilon,0.0,A)
    b2[10] = -n2*cerfon_psi_p_ddr(1.0-epsilon,0.0,A)
    b2[11] = -n3*cerfon_psi_p_ddz(1.0-delta*epsilon,kappa*epsilon,A)

    d = numpy.linalg.solve(M1 - M2, b1 + b2)

    return d


def cerfon_psi_p(r,z,A):
    r"""
    Particular Soloviev solution of Grad-Shafranov equation, refering to :math:`\psi_{p}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_p(r,z,A)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].
    A : float64, size: single value
        Related to the profile, as given in [cerfon]_.

    Returns
    -------
    psi_p : numpy.array, size: [N,M]
            The computation of the particular solution of the magnetic flux at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    return ((r**4)/8.0) + A*(0.5*(r**2)*numpy.log(r) - ((r**4)/8.0))


def cerfon_psi_p_ddr(r,z,A):
    r"""
    Derivative with respect to r of particular Soloviev solution of Grad-Shafranov equation, refering to
    :math:`\frac{\partial\psi_{p}}{\partial r}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_p_ddr(r,z,A)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].
    A : float64, size: single value
        Related to the profile, as given in [cerfon]_.

    Returns
    -------
    psi_p_ddr : numpy.array, size: [N,M]
                The computation of the particular solution of the magnetic flux at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    return 0.5*(r**3) + A*(0.5*r - 0.5*(r**3) + r*numpy.log(r))


def cerfon_psi_p_ddr2(r,z,A):
    r"""
    Second derivative with respect to r of particular Soloviev solution of Grad-Shafranov equation, refering to
    :math:`\frac{\partial^{2}\psi_{p}}{\partial r^{2}}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_p_ddr2(r,z,A)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].
    A : float64, size: single value
        Related to the profile, as given in [cerfon]_.

    Returns
    -------
    psi_p_ddr2 : numpy.array, size: [N,M]
                The computation of the particular solution of the magnetic flux at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    return 1.5*(r**2) + A*(1.5 - 1.5*(r**2) + numpy.log(r))


def cerfon_psi_p_ddz(r,z,A):
    r"""
    Derivative with respect to z of particular Soloviev solution of Grad-Shafranov equation, refering to
    :math:`\frac{\partial\psi_{p}}{\partial z}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_p_ddz(r,z,A)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].
    A : float64, size: single value
        Related to the profile, as given in [cerfon]_.

    Returns
    -------
    psi_p_ddz : numpy.array, size: [N,M]
                The computation of the particular solution of the magnetic flux at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    try:
        data_shape = r.shape
    except AttributeError:
        data_shape = 1

    return numpy.zeros(data_shape)


def cerfon_psi_p_ddz2(r,z,A):
    r"""
    Second derivative with respect to z of particular Soloviev solution of Grad-Shafranov equation, refering to
    :math:`\frac{\partial\psi_{p}}{\partial z}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_p_ddz2(r,z,A)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].
    A : float64, size: single value
        Related to the profile, as given in [cerfon]_.

    Returns
    -------
    psi_p_ddz2 : numpy.array, size: [N,M]
                The computation of the particular solution of the magnetic flux at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    try:
        data_shape = r.shape
    except AttributeError:
        data_shape = 1

    return numpy.zeros(data_shape)


def cerfon_psi_h(r,z):
    r"""
    Homogeneous Soloviev solutions of Grad-Shafranov equation, refering to
    :math:`\psi_{h}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_h(r,z)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].

    Returns
    -------
    psi_h : numpy.array, size: [N,M,12]
            The computation of the 12 Soloviev homogeneous solutions of the Grad-Shafranov equation at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    # allocate memory space for the solution
    # there are 12 homogeneous solutions, and the data is [M,N], therefore we allocate a matrix of
    # dimensions [12,M,N]
    try:
        data_shape = numpy.hstack((12,r.shape))
    except AttributeError:
        data_shape = numpy.array([12,1])

    psi_h = numpy.zeros(data_shape)

    # compute psi_h_1
    psi_h[0] = numpy.ones(data_shape[1:])

    # compute psi_h_2
    psi_h[1] = r**2

    # compute psi_h_3
    psi_h[2] = (z**2) - (r**2)*numpy.log(r)

    # compute psi_h_4
    psi_h[3] = (r**4) - 4.0*(r**2)*(z**2)

    # compute psi_h_5
    psi_h[4] = 2.0*(z**4) - 9.0*(z**2)*(r**2) + 3.0*(r**4)*numpy.log(r) - 12.0*(r**2)*(z**2)*numpy.log(r)

    # compute psi_h_6
    psi_h[5] = (r**6) - 12.0*(r**4)*(z**2) + 8.0*(r**2)*(z**4)

    # compute psi_h_7
    psi_h[6] = 8.0*(z**6) - 140.0*(z**4)*(r**2) + 75.0*(z**2)*(r**4) - 15.0*(r**6)*numpy.log(r) +\
               180.0*(r**4)*(z**2)*numpy.log(r) - 120.0*(r**2)*(z**4)*numpy.log(r)

    # compute psi_h_8
    psi_h[7] = z

    # compute psi_h_9
    psi_h[8] = z*(r**2)

    # compute psi_h_10
    psi_h[9] = (z**3) - 3.0*z*(r**2)*numpy.log(r)

    # compute psi_h_11
    psi_h[10] = 3.0*z*(r**4) - 4.0*(z**3)*(r**2)

    # compute psi_h_12
    psi_h[11] = 8.0*(z**5) - 45.0*z*(r**4) - 80.0*(z**3)*(r**2)*numpy.log(r) + 60.0*z*(r**4)*numpy.log(r)


    return psi_h


def cerfon_psi_h_ddr(r,z):
    r"""
    First derivative with respect to r of homogeneous Soloviev solutions of Grad-Shafranov equation, refering to
    :math:`\frac{\partial\psi_{h}}{\partial r}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_h_ddr(r,z)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].

    Returns
    -------
    psi_h_ddr : numpy.array, size: [N,M,12]
                The computation of first derivative with respect to r of the 12 Soloviev homogeneous solutions
                of the Grad-Shafranov equation at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    # allocate memory space for the solution
    # there are 12 homogeneous solutions, and the data is [M,N], therefore we allocate a matrix of
    # dimensions [12,M,N]
    try:
        data_shape = numpy.hstack((12,r.shape))
    except AttributeError:
        data_shape = numpy.array([12,1])

    psi_h_ddr = numpy.zeros(data_shape)

    # compute psi_h_ddr_1
    #psi_h_ddr[0] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddr_2
    psi_h_ddr[1] = 2.0*r

    # compute psi_h_ddr_3
    psi_h_ddr[2] = -r -2.0*r*numpy.log(r)

    # compute psi_h_ddr_4
    psi_h_ddr[3] = 4.0*(r**3) - 8.0*r*(z**2)

    # compute psi_h_ddr_5
    psi_h_ddr[4] = 3.0*(r**3) - 30.0*r*(z**2) + 12.0*(r**3)*numpy.log(r) - 24.0*r*(z**2)*numpy.log(r)

    # compute psi_h_ddr_6
    psi_h_ddr[5] = 6.0*(r**5) - 48.0*(r**3)*(z**2) + 16.0*r*(z**4)

    # compute psi_h_ddr_7
    psi_h_ddr[6] = -15.0*(r**5) + 480.0*(r**3)*(z**2) - 400.0*r*(z**4) - 90.0*(r**5)*numpy.log(r) +\
                   720.0*(r**3)*(z**2)*numpy.log(r) - 240.0*r*(z**4)*numpy.log(r)

    # compute psi_h_ddr_8
    #psi_h_ddr[7] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddr_9
    psi_h_ddr[8] = 2.0*z*r

    # compute psi_h_ddr_10
    psi_h_ddr[9] = -3.0*r*z - 6.0*r*z*numpy.log(r)

    # compute psi_h_ddr_11
    psi_h_ddr[10] = 12.0*(r**3)*z - 8.0*r*(z**3)

    # compute psi_h_ddr_12
    psi_h_ddr[11] = -120.0*(r**3)*z - 80.0*r*(z**3) + 240.0*(r**3)*z*numpy.log(r) - 160.0*r*(z**3)*numpy.log(r)


    return psi_h_ddr


def cerfon_psi_h_ddr2(r,z):
    r"""
    Second derivative with respect to r of homogeneous Soloviev solutions of Grad-Shafranov equation, refering to
    :math:`\frac{\partial^{2}\psi_{h}}{\partial r^{2}}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_h_ddr2(r,z)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].

    Returns
    -------
    psi_h_ddr2 : numpy.array, size: [N,M,12]
                The computation of first derivative with respect to r of the 12 Soloviev homogeneous solutions
                of the Grad-Shafranov equation at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    # allocate memory space for the solution
    # there are 12 homogeneous solutions, and the data is [M,N], therefore we allocate a matrix of
    # dimensions [12,M,N]
    try:
        data_shape = numpy.hstack((12,r.shape))
    except AttributeError:
        data_shape = numpy.array([12,1])
    psi_h_ddr2 = numpy.zeros(data_shape)

    # compute psi_h_ddr2_1
    #psi_h_ddr2[0] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddr2_2
    psi_h_ddr2[1] = 2.0*numpy.ones(data_shape[1:])

    # compute psi_h_ddr2_3
    psi_h_ddr2[2] = -3.0 - 2.0*numpy.log(r)

    # compute psi_h_ddr2_4
    psi_h_ddr2[3] = 12.0*(r**2) - 8.0*(z**2)

    # compute psi_h_ddr2_5
    psi_h_ddr2[4] = 21.0*(r**2) - 54.0*(z**2) + 36.0*(r**2)*numpy.log(r) - 24.0*(z**2)*numpy.log(r)

    # compute psi_h_ddr2_6
    psi_h_ddr2[5] = 30.0*(r**4) - 144.0*(r**2)*(z**2) + 16.0*(z**4)

    # compute psi_h_ddr2_7
    psi_h_ddr2[6] = -165.0*(r**4) + 2160.0*(r**2)*(z**2) - 640.0*(z**4) - 450.0*(r**4)*numpy.log(r) + \
                    2160.0*(r**2)*(z**2)*numpy.log(r) - 240.0*(z**4)*numpy.log(r)

    # compute psi_h_ddr2_8
    #psi_h_ddr2[7] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddr2_9
    psi_h_ddr2[8] = 2.0*z

    # compute psi_h_ddr2_10
    psi_h_ddr2[9] = -9.0*z - 6.0*z*numpy.log(r)

    # compute psi_h_ddr2_11
    psi_h_ddr2[10] = 36.0*(r**2)*z - 8.0*z**3

    # compute psi_h_ddr2_12
    psi_h_ddr2[11] = -120.0*(r**2)*z - 240.0*(z**3) + 720.0*(r**2)*z*numpy.log(r) - 160.0*(z**3)*numpy.log(r)


    return psi_h_ddr2


def cerfon_psi_h_ddz(r,z):
    r"""
    First derivative with respect to z of homogeneous Soloviev solutions of Grad-Shafranov equation, refering to
    :math:`\frac{\partial\psi_{h}}{\partial z}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_h_ddz(r,z)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].

    Returns
    -------
    psi_h_ddz : numpy.array, size: [N,M,12]
                The computation of first derivative with respect to r of the 12 Soloviev homogeneous solutions
                of the Grad-Shafranov equation at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    # allocate memory space for the solution
    # there are 12 homogeneous solutions, and the data is [M,N], therefore we allocate a matrix of
    # dimensions [12,M,N]
    try:
        data_shape = numpy.hstack((12,r.shape))
    except AttributeError:
        data_shape = numpy.array([12,1])
    psi_h_ddz = numpy.zeros(data_shape)

    # compute psi_h_ddz_1
    #psi_h_ddz[0] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddz_2
    #psi_h_ddz[1] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddz_3
    psi_h_ddz[2] = 2.0*z

    # compute psi_h_ddz_4
    psi_h_ddz[3] = -8.0*(r**2)*z

    # compute psi_h_ddz_5
    psi_h_ddz[4] = -18.0*(r**2)*z + 8.0*(z**3) - 24.0*(r**2)*z*numpy.log(r)

    # compute psi_h_ddz_6
    psi_h_ddz[5] = -24.0*(r**4)*z + 32.0*(r**2)*(z**3)

    # compute psi_h_ddz_7
    psi_h_ddz[6] = 150.0*(r**4)*z - 560.0*(r**2)*(z**3) + 48.0*(z**5) + 360.0*(r**4)*z*numpy.log(r) - 480.0*(r**2)*(z**3)*numpy.log(r)

    # compute psi_h_ddz_8
    psi_h_ddz[7] = numpy.ones(data_shape[1:])

    # compute psi_h_ddz_9
    psi_h_ddz[8] = (r**2)

    # compute psi_h_ddz_10
    psi_h_ddz[9] = 3.0*(z**2) - 3.0*(r**2)*numpy.log(r)

    # compute psi_h_ddz_11
    psi_h_ddz[10] = 3.0*(r**4) - 12.0*(r**2)*(z**2)

    # compute psi_h_ddz_12
    psi_h_ddz[11] = -45.0*(r**4) + 40.0*(z**4) + 60.0*(r**4)*numpy.log(r) - 240.0*(r**2)*(z**2)*numpy.log(r)


    return psi_h_ddz


def cerfon_psi_h_ddz2(r,z):
    r"""
    Second derivative with respect to z of homogeneous Soloviev solutions of Grad-Shafranov equation, refering to
    :math:`\frac{\partial^{2}\psi_{h}}{\partial z^{2}}` in [cerfon]_.

    Usage
    -----
    .. code-block :: python

        cerfon_psi_h_ddz2(r,z)


    Parameters
    ----------
    r : numpy.array, size: (N,M)
        The r coordinates of the points where to compute the magnetic
        flux. r \in [0,:math:`\infty`].
    z : numpy.array, size: (N,M)
        The z coordinates of the points where to compute the magnetic
        flux. z \in [-:math:`\infty`,:math:`\infty`].

    Returns
    -------
    psi_h_ddz2 : numpy.array, size: [N,M,12]
                The computation of first derivative with respect to r of the 12 Soloviev homogeneous solutions
                of the Grad-Shafranov equation at the points (r,z).


    References
    ----------
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    :First Added:   2016-04-22
    :Last Modified: 2015-04-22
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-22)
    """

    # allocate memory space for the solution
    # there are 12 homogeneous solutions, and the data is [M,N], therefore we allocate a matrix of
    # dimensions [12,M,N]
    try:
        data_shape = numpy.hstack((12,r.shape))
    except AttributeError:
        data_shape = numpy.array([12,1])
    psi_h_ddz2 = numpy.zeros(data_shape)

    # compute psi_h_ddr2_1
    #psi_h_ddr2[0] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddr2_2
    #psi_h_ddz2[1] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddr2_3
    psi_h_ddz2[2] = 2.0*numpy.ones(data_shape[1:])

    # compute psi_h_ddr2_4
    psi_h_ddz2[3] = -8.0*(r**2)

    # compute psi_h_ddr2_5
    psi_h_ddz2[4] = -18.0*(r**2) + 24.0*(z**2) - 24.0*(r**2)*numpy.log(r)

    # compute psi_h_ddr2_6
    psi_h_ddz2[5] = -24.0*(r**4) + 96.0*(r**2)*(z**2)

    # compute psi_h_ddr2_7
    psi_h_ddz2[6] = 150.0*(r**4) - 1680.0*(r**2)*(z**2) + 240.0*(z**4) + 360.0*(r**4)*numpy.log(r) - 1440.0*(r**2)*(z**2)*numpy.log(r)

    # compute psi_h_ddr2_8
    #psi_h_ddz2[7] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddr2_9
    #psi_h_ddz2[8] = numpy.zeros(data_shape[1:]) # skipped because it is already initialized as zero

    # compute psi_h_ddr2_10
    psi_h_ddz2[9] = 6.0*z

    # compute psi_h_ddr2_11
    psi_h_ddz2[10] = -24.0*(r**2)*z

    # compute psi_h_ddr2_12
    psi_h_ddz2[11] = 160.0*(z**3) - 480.0*(r**2)*z*numpy.log(r)


    return psi_h_ddz2
