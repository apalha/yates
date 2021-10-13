# -*- coding: utf-8 -*-
""" Test of the linear solver for the Grad-Shafranov equation.

Description
-----------
This script tests the implementation of the GS module for solving the Grad-Shafranov equation. This test solves a linear
Grad-Shafranov equation. For this reason a Soloviev solution is used, see [1]_ and [2]_. The boundary conditions used
are Dirichlet boundary conditions since the analytical solution for the Grad-Shafranov equation is known.

References
----------
.. [1] Imazawa, R., Kawano, Y., Itami, K., Kusama, Y., Linearly-independent method for a safety factor profile,
       Nuclear Fusion, 2014.

   [2] Shi, B., General Equilibrium property of spherical torus configurations with large triangularity,
       Plasma Physics and Controlled Fusion, 2004.

:First added:  2015-04-13
:Last updated: 2015-04-13
:Copyright: Copyright(C) 2015 apalha
:License: GNU GPL version 3 or any later version
"""

"""
Reviews
-------
1. First implementation. (apalha, 2015-04-13)

"""

import yates
import dolfin
import fenicstools
import numpy
import time
import inspect
import triangle
import triangle.plot
import pylab
import dolfinmplot
import matplotlib.tri as tri

yates.output_parameters['runtime_info'] = True

solver_parameters = yates.DEFAULT_SOLVER_PARAMETERS
solver_parameters['nonlinear_solver'] = 'Picard' # Picard|Newton

# define the mesh
plasma_shape = yates.PlasmaShape('xpoint_iter',n=128) # 128 gives 2049 elements
                                                      # xpoint_iter|pataki_nstx|pataki_iter_0|pataki_iter_1|pataki_iter_2|pataki_iter_3
mesh = plasma_shape.mesh


# define the Grad-Shafranov solver
plasma = yates.GS_Solver(mesh, inner_objects=None, outer_objects=None, solver_parameters=yates.DEFAULT_SOLVER_PARAMETERS)


N = 100
mesh = dolfin.UnitIntervalMesh(N)
x0 = 0.0
h = 1.0/N

V = dolfin.FunctionSpace(mesh,'CG',1)

# Non-eigenvalue solution -----------------------------------------------------

def J(r,z,psi):
    return r*r*0.95 + 0.05 #1.0 + psi + 0.1*psi*psi + 0.1*psi*psi*psi #numpy.sin(psi*psi)+1


psi_old = yates.DolfinFunction(plasma.psi)


for k in range(40):
    print '\nIteration k= %d' % k
    start_time = time.time()
    plasma.solve_step(J)#,bc_function=psi_python)
    print time.time() - start_time
    start_time = time.time()
    error = dolfin.norm(psi_old.vector()-plasma.psi.vector(),'linf')
    psi_old.update(plasma.psi)
    print time.time() - start_time
    print error


# Eigenvalue solution ---------------------------------------------------------
def J(r,z,psi):
    C1 = -1.0
    C2 = 2.0
    return ((C1/r) + C2*r)*psi


psi_old = yates.DolfinFunction(plasma.psi)
psi_dif = yates.DolfinFunction(plasma.psi)

sigma = 1.0

for k in range(10):
    print '\nIteration k= %d' % k
    start_time = time.time()
    plasma.solve_step(J,sigma=sigma)#,bc_function=psi_python)
    print '   Solving time    : ' + str(time.time() - start_time)
    if k > 0:
        psi_max = numpy.abs(plasma.psi_vector.array()).max()
        plasma.psi_vector[:] = plasma.psi_vector[:] / psi_max
        sigma = sigma/psi_max
        print '   Psi max         : ' + str(psi_max)

    start_time = time.time()
    error = dolfin.norm(psi_old.vector()-plasma.psi.vector(),'linf')

    psi_dif.update(psi_old - plasma.psi)

    psi_old.update(plasma.psi)

    print '   Updating time   : ' + str(time.time() - start_time)
    print '   Iteraton error  : ' + str(error)


myplot = dolfinmplot.plot(plasma.psi)
pylab.colorbar(myplot)
#pylab.clim(0.0,1.0)
dolfinmplot.contour(plasma.psi,levels=numpy.linspace(0,1,11))
#dolfinmplot.plot(plasma.mesh)
