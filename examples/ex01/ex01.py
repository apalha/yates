# -*- coding: utf-8 -*-
""" ex01.py :: Test of Grad-Shafranov analytical solutions available in yates.

Description
-----------
This script tests the implementation of the available analytical solutions for the
Grad-Shafranov equations. One type of solutions is computed: (i) linear (Soloviev).
Below you find the list of solutions tested:

Linear (Soloviev)
    1. ITER- and NSTX-like solutions as presented in [pataki2013]_ .
    2. X-point ITER-like solution as presented in [cerfon2010]_ .
    

References
----------
.. [pataki2013] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., and O’Neil, M. (2013).
                "A fast, high-order solver for the Grad-Shafranov equation",
                Journal of Computational Physics, 243, 28–45. http://doi.org/10.1016/j.jcp.2013.02.045
.. [cerfon2010] Cerfon, A. J., & Freidberg, J. P. (2010).
                "'One size fits al' analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, 17(3), 032502. http://doi.org/10.1063/1.3328818
.. [palhaGS2016] Palha, A., Koren, B. , and Felici, F., 
                 "A mimetic spectral element solver for the Grad–Shafranov equation",
                 Journal of Computational Physics, vol. 316, pp. 63–93, Jul. 2016.

:First added:  2016-04-19
:Last updated: 2016-04-19
:Copyright: Copyright(C) 2016 apalha
:License: GNU GPL version 3 or any later version
"""

"""
Reviews
-------
1. First implementation. (apalha, 2016-04-19)

"""

import yates
import numpy
import pylab


# Input parameters ------------------------------------------------------------

# set the analytical solution to plot
solType = 'soloviev_xpoint_nstx' # availabe options:
                          # 'soloviev_iter'|'soloviev_nstx'
                          # 'soloviev_xpoint_iter'|'soloviev_xpoint_nstx'
nPlotPoints = numpy.array([100,100]) # the number of points to use to plot the
                                     # solution
# -----------------------------------------------------------------------------


# Internal parameters ---------------------------------------------------------

# specify plot domain and contour values, etc., for different solutions

if solType == 'soloviev_iter':
    # r and z bounds, used to plot the solution
    rBounds = numpy.array([0.6,1.4])
    zBounds = numpy.array([-0.6,0.6])
    # the bounds of the contours used to plot the solution
    cBounds = numpy.array([-0.04,0.0])
    # compute the contour values
    cValues = numpy.linspace(cBounds[0],cBounds[1],11)

elif solType == 'soloviev_nstx':
    # r and z bounds, used to plot the solution
    rBounds = numpy.array([0.,2.0])
    zBounds = numpy.array([-1.8,1.8])
    # the bounds of the contours used to plot the solution
    cBounds = numpy.array([-0.25,0.0])
    # compute the contour values
    cValues = numpy.linspace(cBounds[0],cBounds[1],11)
    
elif solType == 'soloviev_xpoint_iter':
    # r and z bounds, used to plot the solution
    rBounds = numpy.array([0.6,1.4])
    zBounds = numpy.array([-0.7,0.7])
    # the bounds of the contours used to plot the solution
    cBounds = numpy.array([-0.035,0.0])
    # compute the contour values
    cValues = numpy.linspace(cBounds[0],cBounds[1],11)
    
elif solType == 'soloviev_xpoint_nstx':
    # r and z bounds, used to plot the solution
    rBounds = numpy.array([0.1,2.0])
    zBounds = numpy.array([-1.8,1.8])
    # the bounds of the contours used to plot the solution
    cBounds = numpy.array([-0.25,0.0])
    # compute the contour values
    cValues = numpy.linspace(cBounds[0],cBounds[1],11)

else:
    raise ValueError(str(solType) + ' is invalid. solType must be: \
                                      soloviev_iter|soloviev_nstx|soloviev_xpoint_iter')
# -----------------------------------------------------------------------------


# Compute the analytical solution ---------------------------------------------

if solType == 'soloviev_iter':
    analytical_solution = yates.gs.SolovievSolution('pataki_iter')

elif solType == 'soloviev_nstx':
    analytical_solution = yates.gs.SolovievSolution('pataki_nstx')
    
elif solType == 'soloviev_xpoint_iter':
    analytical_solution = yates.gs.SolovievSolution('cerfon_x_iter')
    
elif solType == 'soloviev_xpoint_nstx':
    analytical_solution = yates.gs.SolovievSolution('cerfon_x_nstx')
# -----------------------------------------------------------------------------
    
    
# Plot the solution -----------------------------------------------------------

# generate the grid where to plot
r,z = numpy.meshgrid(numpy.linspace(rBounds[0],rBounds[1],nPlotPoints[0]),
                             numpy.linspace(zBounds[0],zBounds[1],nPlotPoints[1]))
                             
# evaluate the solution at the points
psi = analytical_solution(r,z)

# plot the contour plot of the solution
pylab.contour(r,z,psi,cValues)

pylab.colorbar()

pylab.gca().set_aspect('equal', adjustable='box')
pylab.xlim(rBounds)
pylab.ylim(zBounds)
pylab.show()
# -----------------------------------------------------------------------------    