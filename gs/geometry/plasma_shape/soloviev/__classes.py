# -*- coding: utf-8 -*-
""" Plasma geometry

Description
-----------
This module implements the mesh of the plasma (for fixed boundary simulations) and user defined plasma shapes.


References
----------


....

:First added:  2016-04-20
:Last updated: 2016-06-20
:Copyright: Copyright(C) 2016 apalha
:License: GNU GPL version 3 or any later version
"""


__all__ = [
           'SolovievShape'                             # classes
          ]


import dolfin
import numpy
import time
import triangle

from .__functions import *
from ..... import config
from ..... import aux


# Classes --------------------------------------------------------------------------------------------------------------
#                                                                                                  <editor-fold desc="">


class SolovievShape():
    r"""
    This class sets up known plasma shapes for the fixed boundary Grad-Shafranov solver.

    Implements the generation of the plasma boundary (self.boundary) and plasma mesh (self.mesh).

    Usage
    -----
    .. code-block:: python

        PlasmaShape(shape,n=20)


    Parameters
    ----------
    shape : string
        The identification of the plasma shape to generate.
        The codes identifying the different shapes are: ::
            "pataki_iter" : ITER like smooth plasma shape as presented in [pataki2013]_ (full elongation).
            "pataki_nstx" : NSTX like smooth plasma shape as presented in [pataki2013]_.
            "cerfon_x_iter" : ITER like x-point plasma shape as presented in [cerfon2010]_.

    n : int
        The number of points on the boundary to generate.

    Attributes
    ----------
    shape : string
        The identification of the plasma shape to generate.
        The codes identifying the different shapes are: ::
            "pataki_iter" : ITER like smooth plasma shape as presented in [pataki2013]_ (full elongation).
            "pataki_nstx" : NSTX like smooth plasma shape as presented in [pataki2013]_.
            "cerfon_x_iter" : ITER like x-point plasma shape as presented in [cerfon2010]_.

    n : int
        The number of points on the boundary to generate.

    r : numpy.array(float64), [1,n]
        The r coordinates of the points on the boundary of the plasma.

    z : numpy.array(float64), [1,n]
        The z coordinates of the points on the boundary of the plasma.


    References
    ----------
    .. [pataki2013] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., and O’Neil, M. (2013).
                    A fast, high-order solver for the Grad-Shafranov equation.
                    Journal of Computational Physics, 243, 28–45. http://doi.org/10.1016/j.jcp.2013.02.045
    .. [cerfon2010] Cerfon, A. J., & Freidberg, J. P. (2010).
                    ``One size fits al’' analytic solutions to the Grad-Shafranov equation.
                    Physics of Plasmas, 17(3), 032502. http://doi.org/10.1063/1.3328818



    TODO
    ----

    ....

    :First Added:   2015-12-01
    :Last Modified: 2015-12-01
    :Copyright:     Copyright (C) 2015 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2015-12-01)
    """

    def __init__(self, shape, n=20):
        """
        """

        """
        :First Added:   2015-12-01
        :Last Modified: 2015-12-01
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-01)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing = {} # initialize timing dictionary, this is only done in __init__
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        # store inputs
        self.shape = shape
        self.n = n

        # generate the n points at the boundary
        self.generate_boundary(self.n)

        # generate the mesh of the plasma domain based on the boundary
        self.generate_mesh()


        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


    def generate_boundary(self,n):
        r"""
        Compute the piecewise definition of the boundary of the plasma as an array of n points.

        Usage
        -----
        .. code-block :: python

            self.boundary(n)


        Parameters
        ----------
        n : int
            The number of points on the boundary to generate.


        Returns
        -------


        Attributes
        ----------
        n
        r
        z


        :First Added:   2015-12-01
        :Last Modified: 2015-12-01
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-01)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        self.n = n # update the number of points

        # compute the n boundary points
        if self.shape == 'pataki_iter':
            self.r, self.z = pataki_iter(self.n)
        elif self.shape == 'pataki_nstx':
            self.r, self.z = pataki_nstx(self.n)
        elif self.shape == 'cerfon_x_iter':
            self.r, self.z = cerfon_x_iter(self.n)


        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


    def generate_mesh(self,n=None,show=False):
        r"""
        Compute mesh of of the plasma domain, using n vertices on the boundary.

        Usage
        -----
        .. code-block :: python

            self.generate_mesh(n=None,show=False)


        Parameters
        ----------
        n : int
            The number of points on the boundary to generate.

        show : bool
               A flag that determines of the mesh is plotted of not. Note that if gs.output_parameter['plotting'] = False
               no plots will be shown even if show = True.


        Returns
        -------


        Attributes
        ----------
        n
        r
        z
        mesh


        :First Added:   2015-12-02
        :Last Modified: 2015-12-02
        :Copyright:     Copyright (C) 2015 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2015-12-02)
        """

        # timing start -------------------------------------------------------------------------------------------------
        if config.output['timing']:
            start_time = time.time() # start the timer for this function
        #---------------------------------------------------------------------------------------------------------------


        if n != None:
            self.generate_boundary(n)

        # compute the maximum area size of the triangulation
        h = numpy.hypot(numpy.diff(self.r), numpy.diff(self.z)) # start by computing the length of the edges along the boundary
        hmin = h.min()
        hmax = h.max()
        tri_area = 0.25*(hmin+hmax)**2 # the maximum area is the area of an equilateral triangle of side equal to the
                                       # average between the maximum and minimum edge length along the boundary

        # compute the triangulation using triangle
        vertices = numpy.stack((self.r[0:-1],self.z[0:-1]),axis=1)                       # Start by generating the vertices
                                                                                         # at the boundary of the plasma
                                                                                         # in the format used by triangle.
                                                                                         # The last vertex is not used
                                                                                         # in order not to have repeated
                                                                                         # vertices.

        segment_markers = numpy.ones([self.r.size-1,1],dtype=numpy.int32) # there is only one segment marker, this value is arbitrary, so we choose 1

        vertex_markers = 2*numpy.ones([self.r.size-1,1],dtype=numpy.int32) # there is only one vertex marker, this value is arbitrary, so we choose 1

        segments = numpy.stack((numpy.arange(0,self.r.size-1,dtype=numpy.int32),numpy.arange(1,self.r.size,dtype=numpy.int32)),axis=1)
        segments[-1,-1] = 0

        boundary_definition = {'segment_markers':segment_markers,'segments':segments,'vertex_markers':vertex_markers,'vertices':vertices}
        triangle_mesh = triangle.triangulate(boundary_definition,'pq30a%.20f' %(tri_area)) # Triangulate with quality parameters:
                                                                                      # q30, meaning that the minimum angle
                                                                                      # of a triangle of the mesh is 30,
                                                                                      # and a%f %(tri_area), meaning that
                                                                                      # the maximum area of the triangles
                                                                                      # will be equal to tri_area.
        nvertices = triangle_mesh['vertices'].shape[0] # the number of vertices in the mesh
        ncells = triangle_mesh['triangles'].shape[0] # the number of cells in the mesh


        # create the dolfin mesh and open it for editing
        self.mesh = dolfin.Mesh()
        mesh_editor = dolfin.MeshEditor()
        mesh_editor.open(self.mesh,2,2) # the mesh has dimension 2 (surface) and topological dimension 2 also (surfaces)

        # add the vertices
        mesh_editor.init_vertices(nvertices)
        for k,vertex in enumerate(triangle_mesh['vertices']):
            mesh_editor.add_vertex(k, vertex[0], vertex[1])

        # Add the cells (triangular elements)
        mesh_editor.init_cells(ncells)
        for k,cell in enumerate(triangle_mesh['triangles']):
            mesh_editor.add_cell(k, cell[0], cell[1], cell[2])

        mesh_editor.close()

        if config.output['runtime_info']:
            print 'Created mesh with %d vertices and %d cells' % (nvertices, ncells)

        if config.output['plotting'] and show:
            dolfin.plot(self.mesh)
            dolfin.interactive()


        # timing end ---------------------------------------------------------------------------------------------------
        if config.output['timing']:
            self.timing[aux.currentFuncName()] = time.time() - start_time # compute the execution time of the current function
        #---------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------- </editor-fold>


