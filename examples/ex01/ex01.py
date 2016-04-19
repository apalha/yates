# -*- coding: utf-8 -*-
""" ex01.py :: Test of the linear solver for the Grad-Shafranov equation.

Description
-----------
This script tests the implementation of the GS module for solving the 
Grad-Shafranov equation. This test solves a linear Grad-Shafranov equation. 
For this reason a Soloviev solution is used, see [1]_ and [2]_. 
The boundary conditions used are Dirichlet boundary conditions since the
analytical solution for the Grad-Shafranov equation is known.

References
----------
.. [1] Imazawa, R., Kawano, Y., Itami, K., Kusama, Y., Linearly-independent method for a safety factor profile,
       Nuclear Fusion, 2014.

   [2] Shi, B., General Equilibrium property of spherical torus configurations with large triangularity,
       Plasma Physics and Controlled Fusion, 2004.

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
import dolfin
import numpy
import time
import pylab
import dolfinmplot

# Set yates parameters --------------------------------------------------------

yates.gs.output_parameters['runtime_info'] = True # yates computes time spent in each function

# Define yates solver parameters. 
solver_parameters = yates.gs.DEFAULT_SOLVER_PARAMETERS # Start by using the default parameters.
solver_parameters['nonlinear_solver'] = 'Picard'    # Select the type of nonlinear solver options:
                                                    # Picard|Newton
# -----------------------------------------------------------------------------


# Input parameters ------------------------------------------------------------

# Define the current density. YATES requires that J is a function of (r,z,psi)
# therefore the three inputs need to be present in the definition of the function,
# nevertheless, the function itself does not need to depend on all variables. In
# this case, since it is a linear case, J only depends on (r,z).
def J(r,z,psi):
    return r
# -----------------------------------------------------------------------------

# Non-eigenvalue solution -----------------------------------------------------



# Initialize the solver -------------------------------------------------------

# define the mesh
plasma_shape = yates.gs.PlasmaShape('xpoint_iter',n=128) # Use a predefined plasma shape. Available shapes:
                                                      # xpoint_iter|pataki_nstx|pataki_iter_0|pataki_iter_1|pataki_iter_2|pataki_iter_3
                                                      # 128 gives 2049 elements
mesh = plasma_shape.mesh


# define the Grad-Shafranov solver
# here we consider a fixed boundary therefore the inner_objects are set to None
# since there is only the plasma in the computational domain, outer_objects is
# also set to None because the boundary of the plasma is also fixed and the
# value of psi at the boundary is set to 0, the solver_parameters have been defined
# before and it will essentially be Picard iterations for the nonlinearity (in
# this case it is linear, therefore Picard and Newton will result in the same
# solution)
plasma = yates.gs.GS_Solver(mesh, inner_objects=None, outer_objects=None, solver_parameters=solver_parameters)
# -----------------------------------------------------------------------------


  
psi_old = yates.DolfinFunction(plasma.psi)


for k in range(10):
    print '\nIteration k= %d' % k
    start_time = time.time()
    plasma.solve_step(J)#,bc_function=psi_python)
    print time.time() - start_time
    start_time = time.time()
    error = dolfin.norm(psi_old.vector()-plasma.psi.vector(),'linf')
    psi_old.update(plasma.psi)
    print time.time() - start_time
    print error
    

myplot = dolfinmplot.plot(plasma.psi)
pylab.colorbar(myplot)
#pylab.clim(0.0,1.0)
dolfinmplot.contour(plasma.psi,levels=numpy.linspace(0,1,11))
#dolfinmplot.plot(plasma.mesh) 

## Eigenvalue solution ---------------------------------------------------------
#def J(r,z,psi):
#    C1 = -1.0
#    C2 = 2.0
#    return ((C1/r) + C2*r)*psi
#    
# 
#psi_old = yates.DolfinFunction(plasma.psi)
#psi_dif = yates.DolfinFunction(plasma.psi)
#
#sigma = 1.0
#
#for k in range(10):
#    print '\nIteration k= %d' % k
#    start_time = time.time()
#    plasma.solve_step(J,sigma=sigma)#,bc_function=psi_python)
#    print '   Solving time    : ' + str(time.time() - start_time)
#    if k > 0:
#        psi_max = numpy.abs(plasma.psi_vector.array()).max()
#        plasma.psi_vector[:] = plasma.psi_vector[:] / psi_max
#        sigma = sigma/psi_max
#        print '   Psi max         : ' + str(psi_max)
#        
#    start_time = time.time()
#    error = dolfin.norm(psi_old.vector()-plasma.psi.vector(),'linf')
#    
#    psi_dif.update(psi_old - plasma.psi)
#    
#    psi_old.update(plasma.psi)
#    
#    print '   Updating time   : ' + str(time.time() - start_time)
#    print '   Iteraton error  : ' + str(error)
#
#
#myplot = dolfinmplot.plot(plasma.psi)
#pylab.colorbar(myplot)
##pylab.clim(0.0,1.0)
#dolfinmplot.contour(plasma.psi,levels=numpy.linspace(0,1,11))
##dolfinmplot.plot(plasma.mesh) 


r = numpy.linspace(0.6,1.6,200)
z = numpy.linspace(-0.7,0.7,200)

r = numpy.linspace(0.,2.0,200)
z = numpy.linspace(-1.6,1.6,200)

rGrid,zGrid = numpy.meshgrid(r,z)

epsilon = 0.32
kappa = 1.7
delta = 0.33

epsilon = 0.78
kappa = 2.0
delta = 0.35

psiGrid = yates.psi_soloviev_pataki(rGrid,zGrid,epsilon,kappa,delta)

pylab.pcolormesh(rGrid,zGrid,psiGrid,shading='gouraud')
pylab.clim(-0.04,0.0)
pylab.clim(-0.25,0.0)
