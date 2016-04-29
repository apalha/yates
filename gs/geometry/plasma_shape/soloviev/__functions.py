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
1. First implementation. (apalha, 2016-04-21)

"""

__all__ = [
           'pataki_iter','pataki_nstx', 'cerfon_x_iter', 'cerfon_x_nstx'
          ]


import numpy

from yates import aux


def pataki_iter(n):
    r"""
    Function that returns a set of n points that defines the boundary of
    the Soloviev Pataki ITER-like plasma shape test case [pataki2013]_.

    Usage
    -----
    .. code-block :: python

        pataki_iter(n)


    Parameters
    ----------
    n : int
        The number of points on the boundary to generate.


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


    :First Added:   2016-04-26
    :Last Modified: 2016-04-26
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-26)
    """

    # The boundary is made up of fours segments parametrized from [-1,1]. For this reason
    # we take the interval [0,4] and distribute the n points inside this interval. The points in [0,1] are generated
    # using the first segment (after remapping [0,1]-->[-1,1]), the ones from ]1,2] are generated using the second line
    # segment (after remapping [1,2]-->[-1,1]), the ones from ]2,3]  are generated using the third line segment
    # (after remapping [2,3]-->[-1,1]) and the points from ]3,4] are generated using the fourth line segment
    # (after remapping [3,4]-->[-1,1])

    # generate the points from [0,4]
    s = numpy.linspace(0,4,n)

    # allocate memory space for the point's coordinates
    r = numpy.zeros(n)
    z = numpy.zeros(n)

    # segment 1 --------------------------------------------------------------------------------------------------------
    s_segment_1 = s[s <= 1] # select the points for the first segment
    n_segment_1 = s_segment_1.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([ 1.0000000000000002, 1.0067521193843756, 1.0223800153205331,\
                            1.0461589937358393, 1.0768930106147496, 1.1127861719534229,\
                            1.1514728734182662, 1.1902173863978538, 1.2262741699796955,\
                            1.2573273127982527, 1.2818793511742466, 1.299464988631548,\
                            1.3106243147575545, 1.316653355101908,  1.3192164389787169,\
                            1.3199287559501636, 1.3200000000000003])

    zInterp = numpy.array([-0.544979950070218,   -0.5430793996083392,  -0.5378507062525623,\
                           -0.52769219861124,    -0.510612902875075,   -0.48482216218281277,\
                           -0.4492826362209447,  -0.4041040654103523,  -0.35069096493666246,\
                           -0.2916218509957791,  -0.2303028702908299,  -0.17049176168376695,\
                           -0.11581495024858685, -0.06938964248356089, -0.03361581531768714,\
                           -0.010139695215151097, 0.])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_1 = s_segment_1*2.0 -1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_1,p)


    r[0:n_segment_1] = numpy.dot(evaluatedPoly,rInterp)
    z[0:n_segment_1] = numpy.dot(evaluatedPoly,zInterp)


    # segment 2 --------------------------------------------------------------------------------------------------------
    s_segment_2 = s[numpy.logical_and((s > 1), (s <= 2))] - 1.0 # -1.0 so that s_segment_2 lies in [0,1]
    n_segment_2 = s_segment_2.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([1.3200000000000003, 1.319942249605395, 1.3193646099897596,\
                           1.3172825702549036, 1.3123629527350429, 1.3031738118441916,\
                           1.288466804650494, 1.2674487327143866, 1.2400000000000002,\
                           1.2068049011885247, 1.169370985909541, 1.1299304840585587,\
                           1.0912336789341839, 1.056259769863937, 1.0278830112967758,\
                           1.0085399530833412, 1.0000000000000002])

    zInterp = numpy.array([0., 0.009129159471835747, 0.03027312578855965,\
                           0.06254552449259597, 0.10461188157640205, 0.15460408339312767,\
                           0.21014439915569302, 0.2684398500856314, 0.3264523746064577,\
                           0.38113858375574194, 0.4297440338488312, 0.4701289112713609,\
                           0.5010858219546498, 0.5225737210394233, 0.5357356275419278,\
                           0.5425397851089682, 0.544979950070218])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_2 = s_segment_2*2.0 -1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_2,p)


    r[n_segment_1:(n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,rInterp)
    z[n_segment_1:(n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,zInterp)


    # segment 3 --------------------------------------------------------------------------------------------------------
    s_segment_3 = s[numpy.logical_and((s > 2), (s <= 3))] - 2.0 # -2.0 so that s_segment_3 lies in [0,1]
    s_segment_3 = 1.0 - s_segment_3 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_3 = s_segment_3.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.68, 0.6800577503946053, 0.6806353900102406,\
                           0.6827174297450967, 0.6876370472649574,\
                           0.6968261881558089, 0.7115331953495063,\
                           0.7325512672856138, 0.7600000000000001,\
                           0.7931950988114755, 0.8306290140904593,\
                           0.8700695159414417, 0.9087663210658165,\
                           0.9437402301360633, 0.9721169887032246,\
                           0.9914600469166592, 1.0000000000000002])

    zInterp = numpy.array([0., 0.012718442511455489, 0.042147006024027836,\
                           0.08686580517415964, 0.1444596718038423, 0.2112412417733608,\
                           0.28235659088950577, 0.35230563357005407, 0.4157863631952753,\
                           0.4685873239445894, 0.5082295382185879, 0.5342092969696193,\
                           0.5478760951242375, 0.552055953270008, 0.5504871512348618,\
                           0.5470703258745934, 0.544979950070218])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_3 = s_segment_3*2.0 - 1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_3,p)


    r[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,rInterp)
    z[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,zInterp)


    # segment 4 --------------------------------------------------------------------------------------------------------
    s_segment_4 = s[s > 3] - 3.0 # -3.0 so that s_segment_4 lies in [0,1]
    s_segment_4 = 1.0 - s_segment_4 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_4 = s_segment_4.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([1.0000000000000002, 0.9932478806156247, 0.9776199846794674,\
                           0.9538410062641611, 0.9231069893852508, 0.8872138280465776,\
                           0.8485271265817342, 0.8097826136021467, 0.7737258300203049,\
                           0.7426726872017476, 0.7181206488257539, 0.7005350113684523,\
                           0.689375685242446, 0.6833466448980925, 0.6807835610212835,\
                           0.6800712440498368, 0.68])

    zInterp = numpy.array([-0.544979950070218, -0.5466618561433154, -0.5497037223391971,\
                           -0.5519820575913459, -0.5504546819293027, -0.5415468046900095,\
                           -0.5217331721461321, -0.48829068495033134, -0.4401299850837467,\
                           -0.3784881848168107, -0.3071543228929294, -0.23196092241852154,\
                           -0.1596080142699597, -0.09630046356729889, -0.046792668693458384,\
                           -0.014126064213235588, 0.])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_4 = s_segment_4*2.0 - 1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_4,p)


    r[(n_segment_3+n_segment_2+n_segment_1):] = numpy.dot(evaluatedPoly,rInterp)
    z[(n_segment_3+n_segment_2+n_segment_1):] = numpy.dot(evaluatedPoly,zInterp)


    return r, z


def pataki_nstx(n):
    r"""
    Function that returns a set of n points that defines the boundary of
    the Soloviev Pataki NSTX-like plasma shape test case [pataki2013]_.

    Usage
    -----
    .. code-block :: python

        pataki_nstx(n)


    Parameters
    ----------
    n : int
        The number of points on the boundary to generate.


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


    :First Added:   2016-04-26
    :Last Modified: 2016-04-26
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-26)
    """

    # The boundary is made up of fours segments parametrized from [-1,1]. For this reason
    # we take the interval [0,4] and distribute the n points inside this interval. The points in [0,1] are generated
    # using the first segment (after remapping [0,1]-->[-1,1]), the ones from ]1,2] are generated using the second line
    # segment (after remapping [1,2]-->[-1,1]), the ones from ]2,3]  are generated using the third line segment
    # (after remapping [2,3]-->[-1,1]) and the points from ]3,4] are generated using the fourth line segment
    # (after remapping [3,4]-->[-1,1])

    # generate the points from [0,4]
    s = numpy.linspace(0,4,n)

    # allocate memory space for the point's coordinates
    r = numpy.zeros(n)
    z = numpy.zeros(n)

    # segment 1 --------------------------------------------------------------------------------------------------------
    s_segment_1 = s[s <= 1] # select the points for the first segment
    n_segment_1 = s_segment_1.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.9999999999999997, 1.0164582909994149, 1.0545512873437985,\
                           1.1125125472311075, 1.1874267133734513, 1.2749162941364673,\
                           1.3692151289570231, 1.463654879344768, 1.5515432893255072,\
                           1.6272353249457405, 1.6870809184872253, 1.7299459097893977,\
                           1.7571467672215384, 1.7718425530609, 1.778090070010622,\
                           1.7798263426285232, 1.7800000000000002])

    zInterp = numpy.array([-1.447056883438098, -1.4371130565621713, -1.4127706756195084,\
                           -1.3721000095322844, -1.3126980520752947, -1.2325535316223384,\
                           -1.1308822714797482, -1.008844819578136, -0.8699674627005893,\
                           -0.7201041864416098, -0.5668900204452094, -0.4188110564443869,\
                           -0.2841568527606102, -0.1701448753707013, -0.08240544981209831,\
                           -0.024854575073684135, 0.])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_1 = s_segment_1*2.0 -1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_1,p)


    r[0:n_segment_1] = numpy.dot(evaluatedPoly,rInterp)
    z[0:n_segment_1] = numpy.dot(evaluatedPoly,zInterp)


    # segment 2 --------------------------------------------------------------------------------------------------------
    s_segment_2 = s[numpy.logical_and((s > 1), (s <= 2))] - 1.0 # -1.0 so that s_segment_2 lies in [0,1]
    n_segment_2 = s_segment_2.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([1.7800000000000002, 1.77985923341315, 1.7784512368500387,\
                           1.773376264996327, 1.7613846972916665, 1.7389861663702162,\
                           1.7031378363355787, 1.6519062859913167, 1.5850000000000002,\
                           1.5040869466470286, 1.4128417781545055, 1.3167055548927362,\
                           1.2223820924020725, 1.1371331890433458, 1.0679648400358905,\
                           1.0208161356406433, 0.9999999999999997])

    zInterp = numpy.array([0., 0.022377504447081708, 0.07421011854760332, 0.15335320009922312,\
                           0.25662324769058376, 0.3796287690110551, 0.5168642411911135,\
                           0.6619638145523354, 0.8081169009464911, 0.9486062673718955,\
                           1.0773925277334848, 1.1896490852631756, 1.282153008883561,\
                           1.3534537652512026, 1.4037529767064005, 1.4344224361671618,\
                           1.447056883438098])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_2 = s_segment_2*2.0 -1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_2,p)


    r[n_segment_1:(n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,rInterp)
    z[n_segment_1:(n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,zInterp)


    # segment 3 --------------------------------------------------------------------------------------------------------
    s_segment_3 = s[numpy.logical_and((s > 2), (s <= 3))] - 2.0 # -2.0 so that s_segment_3 lies in [0,1]
    s_segment_3 = 1.0 - s_segment_3 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_3 = s_segment_3.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.219999999999999, 0.22014076658684925, 0.22154876314996058,\
                           0.22662373500367225, 0.2386153027083328, 0.261013833629783,\
                           0.29686216366442064, 0.3480937140086826, 0.41499999999999915,\
                           0.49591305335297076, 0.5871582218454937, 0.683294445107263,\
                           0.7776179075979267, 0.8628668109566534, 0.9320351599641089,\
                           0.979183864359356, 0.9999999999999997])

    zInterp = numpy.array([0., 0.06362192214029187, 0.2100056087792247, 0.42685165904090167,\
                           0.6880690444748594, 0.9544847886720907, 1.1870584581709251,\
                           1.3627590658971411, 1.4785238927700444, 1.5433856597050069,\
                           1.5694394741798414, 1.567570922547975, 1.5470608295347092,\
                           1.5165152796187953, 1.484487546280828, 1.4591419390017881,\
                           1.447056883438098])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_3 = s_segment_3*2.0 - 1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_3,p)


    r[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,rInterp)
    z[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,zInterp)


    # segment 4 --------------------------------------------------------------------------------------------------------
    s_segment_4 = s[s > 3] - 3.0 # -3.0 so that s_segment_4 lies in [0,1]
    s_segment_4 = 1.0 - s_segment_4 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_4 = s_segment_4.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.9999999999999997, 0.9835417090005845, 0.9454487126562009,\
                           0.8874874527688918, 0.8125732866265479, 0.7250837058635318,\
                           0.6307848710429762, 0.5363451206552313, 0.4484567106744921,\
                           0.3727646750542587, 0.31291908151277403, 0.27005409021060156,\
                           0.24285323277846083, 0.22815744693909923, 0.2219099299893772,\
                           0.22017365737147598, 0.21999999999999897])

    zInterp = numpy.array([-1.447056883438098, -1.4566573074616085, -1.4775635729989254,\
                           -1.5058294412215822, -1.5357912279459789, -1.5604055271248114,\
                           -1.5714639205740395, -1.5593244939754174, -1.5121208786599685,\
                           -1.4154293887103497, -1.2552868610758876, -1.0278815676309934,\
                           -0.7522810563877296, -0.47126933843227137, -0.2329196356577878,\
                           -0.07065679653773042, 0.])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_4 = s_segment_4*2.0 - 1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_4,p)


    r[(n_segment_3+n_segment_2+n_segment_1):] = numpy.dot(evaluatedPoly,rInterp)
    z[(n_segment_3+n_segment_2+n_segment_1):] = numpy.dot(evaluatedPoly,zInterp)


    return r, z


def cerfon_x_iter(n):
    r"""
    Function that returns a set of n points that defines the boundary of
    the x-point ITER plasma shape test case in [cerfon].

    Usage
    -----
    .. code-block :: python

        cerfon_x_iter(n)


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
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.


    :First Added:   2016-04-26
    :Last Modified: 2016-04-26
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-26)
    """

    # The boundary is made up of fours segments parametrized from [-1,1]. For this reason
    # we take the interval [0,4] and distribute the n points inside this interval. The points in [0,1] are generated
    # using the first segment (after remapping [0,1]-->[-1,1]), the ones from ]1,2] are generated using the second line
    # segment (after remapping [1,2]-->[-1,1]), the ones from ]2,3]  are generated using the third line segment
    # (after remapping [2,3]-->[-1,1]) and the points from ]3,4] are generated using the fourth line segment
    # (after remapping [3,4]-->[-1,1])

    # generate the points from [0,4]
    s = numpy.linspace(0,4,n)

    # allocate memory space for the point's coordinates
    r = numpy.zeros(n)
    z = numpy.zeros(n)

    # segment 1 --------------------------------------------------------------------------------------------------------
    s_segment_1 = s[s <= 1] # select the points for the first segment
    n_segment_1 = s_segment_1.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.88, 0.8885882004288301, 0.9072121685199243, 0.9346357195881851,\
                           0.969160761679216, 1.009318213054816, 1.0539799056590036,\
                           1.1021662187112302, 1.1525194632298479, 1.2024225972590756,\
                           1.247276495488194, 1.2817292401695104, 1.3033952627728922,\
                           1.314395400819731, 1.3187456879894344, 1.3198891862679323,\
                           1.3200000000000018])

    zInterp = numpy.array([-0.6, -0.5900654704045275, -0.569172718595628, -0.5398559016247365,\
                           -0.5049922030024573, -0.46663747417152185, -0.42580726872439684,\
                           -0.38260076579785296, -0.3364383778978876, -0.2864393940656467,\
                           -0.2322740958749147, -0.17562391926435733, -0.12076312569333342,\
                           -0.07272986484790007, -0.03529165013846899, -0.010648870526125464, 0.])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_1 = s_segment_1*2.0 -1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_1,p)


    r[0:n_segment_1] = numpy.dot(evaluatedPoly,rInterp)
    z[0:n_segment_1] = numpy.dot(evaluatedPoly,zInterp)


    # segment 2 --------------------------------------------------------------------------------------------------------
    s_segment_2 = s[numpy.logical_and((s > 1), (s <= 2))] - 1.0 # -1.0 so that s_segment_2 lies in [0,1]
    n_segment_2 = s_segment_2.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([1.3200000000000012, 1.3198901796074014, 1.3188226627829676,\
                           1.3151396795533445, 1.306837309853852, 1.2918589538054168,\
                           1.2681925116432426, 1.234306746222676, 1.1902948077394662,\
                           1.1386388824646854, 1.0832407429098843, 1.0278553737709262,\
                           0.975617140586031, 0.9294708544522408, 0.8925141672323831,\
                           0.8676082462805893, 0.8567196138320295])

    zInterp = numpy.array([0., 0.010727396232104483, 0.03555677811326186, 0.07335976771048011,\
                           0.12237122451430688, 0.18007868118761727, 0.2431375694676744,\
                           0.3073269757879991, 0.36812375455092405, 0.4220083025188111,\
                           0.46711711291058394, 0.5024904164822502, 0.5271838682212899,\
                           0.5405782510901451, 0.5439893590352796, 0.5417418424170913,\
                           0.5394324490167269])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_2 = s_segment_2*2.0 -1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_2,p)


    r[n_segment_1:(n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,rInterp)
    z[n_segment_1:(n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,zInterp)


    # segment 3 --------------------------------------------------------------------------------------------------------
    s_segment_3 = s[numpy.logical_and((s > 2), (s <= 3))] - 2.0 # -2.0 so that s_segment_3 lies in [0,1]
    s_segment_3 = 1.0 - s_segment_3 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_3 = s_segment_3.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.6799999999999911, 0.6800098451818372, 0.6801054392983898,\
                           0.6804351956050655, 0.6811806395142455, 0.6825406908954548,\
                           0.6847713283193102, 0.688302950313424, 0.6939589125194354,\
                           0.7032436523632031, 0.718328297645575, 0.7408152608643869,\
                           0.7695543164304514, 0.8003421359664202, 0.8278802198060913,\
                           0.847592423467504, 0.8567196138320295])

    zInterp = numpy.array([0.0, 0.006470214670915746, 0.021473674832225856,\
                           0.044500676671458676, 0.07497795659304982, 0.1123694602408083,\
                           0.15627331004164127, 0.20638304653654357, 0.26217740919342475,\
                           0.3221531680516285, 0.3826846516754357, 0.43781604199180535,\
                           0.48170689793106014, 0.5116875179322604, 0.528861216412531,\
                           0.536816306620966, 0.5394324490167269])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_3 = s_segment_3*2.0 - 1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_3,p)


    r[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,rInterp)
    z[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,zInterp)


    # segment 4 --------------------------------------------------------------------------------------------------------
    s_segment_4 = s[s > 3] - 3.0 # -3.0 so that s_segment_4 lies in [0,1]
    s_segment_4 = 1.0 - s_segment_4 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_4 = s_segment_4.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.88, 0.8741330162303284, 0.8610150568773567, 0.8418451052043947,\
                           0.8182290891163575, 0.7919919940442296, 0.7651392641082765,\
                           0.739874969342749, 0.7184039550134975, 0.7022679324130782,\
                           0.6916340151815252, 0.6854314548782595, 0.6821984205839355,\
                           0.6807158900254275, 0.6801573570118585, 0.6800137806461887,\
                           0.6799999999999911])

    zInterp = numpy.array([-0.6, -0.5900191120462467, -0.567769984132001, -0.5352873976121912,\
                           -0.49492175785399356, -0.44872982669692874, -0.39823245735490154,\
                           -0.3445594404968846, -0.2889146862834369, -0.23307750988386616,\
                           -0.179397792706003, -0.1301935132586193, -0.08726116271054629,\
                           -0.051886769776421614, -0.025053090986514717, -0.00754976011730754,\
                           0.0])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_4 = s_segment_4*2.0 - 1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_4,p)


    r[(n_segment_3+n_segment_2+n_segment_1):] = numpy.dot(evaluatedPoly,rInterp)
    z[(n_segment_3+n_segment_2+n_segment_1):] = numpy.dot(evaluatedPoly,zInterp)


    return r, z


def cerfon_x_nstx(n):
    r"""
    Function that returns a set of n points that defines the boundary of
    the x-point NSTX plasma shape test case in [cerfon].

    Usage
    -----
    .. code-block :: python

        cerfon_x_nstx(n)


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
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.


    :First Added:   2016-04-28
    :Last Modified: 2016-04-28
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version

    """

    """
    Reviews:
    1. First implementation. (apalha, 2016-04-28)
    """

    # The boundary is made up of fours segments parametrized from [-1,1]. For this reason
    # we take the interval [0,4] and distribute the n points inside this interval. The points in [0,1] are generated
    # using the first segment (after remapping [0,1]-->[-1,1]), the ones from ]1,2] are generated using the second line
    # segment (after remapping [1,2]-->[-1,1]), the ones from ]2,3]  are generated using the third line segment
    # (after remapping [2,3]-->[-1,1]) and the points from ]3,4] are generated using the fourth line segment
    # (after remapping [3,4]-->[-1,1])

    # generate the points from [0,4]
    s = numpy.linspace(0,4,n)

    # allocate memory space for the point's coordinates
    r = numpy.zeros(n)
    z = numpy.zeros(n)

    # segment 1 --------------------------------------------------------------------------------------------------------
    s_segment_1 = s[s <= 1] # select the points for the first segment
    n_segment_1 = s_segment_1.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.7, 0.7234633706005325, 0.776699203615713, 0.8555753258264527,\
                           0.9551133889977392, 1.070028160843518, 1.1947543715601117,\
                           1.3229086061887905, 1.4466585346506253, 1.5569824375996102,\
                           1.6457888087669386, 1.709079106191855, 1.748332551802545,\
                           1.7689225365681753, 1.7774458285788837, 1.7797699837362337,\
                           1.7800000000000005])

    zInterp = numpy.array([-1.71, -1.686944338132322, -1.635348306549131, -1.560325603605155,\
                           -1.4669594429575197, -1.3588112588111587, -1.2374422664977696,\
                           -1.1029455071327114, -0.9552581408056923, -0.796083061911744,\
                           -0.630666542007622, -0.4678848465984374, -0.3181076886214276,\
                           -0.1906146014157926, -0.09233601015685, -0.027850425360290414, 0.])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_1 = s_segment_1*2.0 -1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_1,p)


    r[0:n_segment_1] = numpy.dot(evaluatedPoly,rInterp)
    z[0:n_segment_1] = numpy.dot(evaluatedPoly,zInterp)


    # segment 2 --------------------------------------------------------------------------------------------------------
    s_segment_2 = s[numpy.logical_and((s > 1), (s <= 2))] - 1.0 # -1.0 so that s_segment_2 lies in [0,1]
    n_segment_2 = s_segment_2.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([1.7800000000000005, 1.779751424071047, 1.777290213144307,\
                           1.7685635630356966, 1.7483487397155513, 1.7113005913762516,\
                           1.6528598266265673, 1.5701407920005492, 1.462985271595146,\
                           1.3347567473884572, 1.192003801093924, 1.0430083307177616,\
                           0.8964705027179901, 0.7614900394954633, 0.6486827808673042,\
                           0.5698610408150089, 0.5346528039080656])

    zInterp = numpy.array([0., 0.02906863133218886, 0.09637500293235754, 0.19898803387768582,\
                           0.332396262441717, 0.49025343559685036, 0.6644905665476979,\
                           0.8456077464532068, 1.0233582890476691, 1.1879421086912032,\
                           1.3310006106900836, 1.4453099956802362, 1.5234408123320697,\
                           1.5583117495142824, 1.5504188526416782, 1.5186805636341394,\
                           1.4960847640641393])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_2 = s_segment_2*2.0 -1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_2,p)


    r[n_segment_1:(n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,rInterp)
    z[n_segment_1:(n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,zInterp)


    # segment 3 --------------------------------------------------------------------------------------------------------
    s_segment_3 = s[numpy.logical_and((s > 2), (s <= 3))] - 2.0 # -2.0 so that s_segment_3 lies in [0,1]
    s_segment_3 = 1.0 - s_segment_3 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_3 = s_segment_3.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.21999999999999942, 0.22001508051977828, 0.22015869614198244,\
                           0.2206366879545397, 0.2216600262835425, 0.22339562814832764,\
                           0.22601099649075945, 0.2298537390975662, 0.23585103995752124,\
                           0.2462781380254665, 0.2656832991357172, 0.2999177463203487,\
                           0.35071560085304965, 0.41138506404658653, 0.4699097072086467,\
                           0.5140508610331874, 0.5346528039080656])

    zInterp = numpy.array([0.0, 0.015251345481609047, 0.0506211136887252,\
                           0.1049357248242771, 0.17693478057590664, 0.2655688789740946,\
                           0.3703289996824086, 0.4913456910377319, 0.6290000159765382,\
                           0.7825106083492844, 0.9468228991364073, 1.1093138621420369,\
                           1.2526210597830798, 1.3636928758053237, 1.4383557572398136,\
                           1.4801332492018406, 1.4960847640641393])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_3 = s_segment_3*2.0 - 1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_3,p)


    r[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,rInterp)
    z[(n_segment_2+n_segment_1):(n_segment_3+n_segment_2+n_segment_1)] = numpy.dot(evaluatedPoly,zInterp)


    # segment 4 --------------------------------------------------------------------------------------------------------
    s_segment_4 = s[s > 3] - 3.0 # -3.0 so that s_segment_4 lies in [0,1]
    s_segment_4 = 1.0 - s_segment_4 # flip the order of the points because the definition of this line segment goes the opposite way
    n_segment_4 = s_segment_4.size

    # the interpolation data at Lobatto points along the boundary
    # used to construct the polynomial that represents the boundary segment
    rInterp = numpy.array([0.7, 0.6841423621684852, 0.6510795182456547, 0.6034913974092164,\
                           0.5460241320463914, 0.4835570928463047, 0.42081829994129094,\
                           0.3623147505328137, 0.3122199458874615, 0.27371339701762787,\
                           0.2477646557604886, 0.23260081116796077, 0.22490108326424135,\
                           0.2215298892288634, 0.22032412432787368, 0.2200276623355889,\
                           0.21999999999999942])

    zInterp = numpy.array([-1.71, -1.676370201838863, -1.6061075211607174, -1.5042971026895338,\
                           -1.3793214513463101, -1.238869147498682, -1.0890170382588285,\
                           -0.9343150292400962, -0.7785523180725756, -0.6257550481412574,\
                           -0.4807206249055658, -0.34850252945637933, -0.23342018786382,\
                           -0.13873567011814072, -0.06697349019859555, -0.020181222748749728,\
                           0.0])

    # compute the degree of the polynomial interpolation
    p = rInterp.size - 1

    # evaluate the interpolating polynomials at the boundary points
    s_segment_4 = s_segment_4*2.0 - 1.0 # rescale [0,1] --> [-1,1] because the the segments are parametrized as [-1,1] --> (r,z)
    evaluatedPoly = aux.lobattoPoly(s_segment_4,p)


    r[(n_segment_3+n_segment_2+n_segment_1):] = numpy.dot(evaluatedPoly,rInterp)
    z[(n_segment_3+n_segment_2+n_segment_1):] = numpy.dot(evaluatedPoly,zInterp)


    return r, z