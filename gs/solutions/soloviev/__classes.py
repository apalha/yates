# -*- coding: utf-8 -*-
""" Class interface for analytical Soloviev solutions to the linear Grad-Shafranov equation

Description
-----------
This module implements a class to interface known analytical Soloviev solutions to the linear Grad-Shafranov equation.


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
           'SolovievSolution'
          ]

from .__functions import pataki, pataki_eval_params, cerfon, cerfon_eval_params


class SolovievSolution():
    r"""
    This class sets up an interface for specific analytical Soloviev solutions to the linear Grad Shafranov equation.

    The linear Grad-Shafranov equation solved is:

    .. math::

        -\nabla\cdot\left(\frac{1}{\mu_{0} r}\nabla\psi\right) = j_{\phi}(r,z)



    Usage
    -----
    .. code-block:: python

        SolovievSolution(solType)


    Parameters
    ----------
    solType : string
              Specifies the analytical Soloviev solution.
              The available solutions are: ::
                    'pataki_iter'      : ITER-like solution from [pataki]_.
                    'pataki_nstx'      : NSTX-like solution from [pataki]_.
                    'cerfon_x_iter'    : ITER-like solution with x-point from [cerfon]_.
                    'cerfon_x_nstx'    : NSTX-like solution with x-point from [cerfon]_.


    Attributes
    ----------
    attribute_1 : type
                  Description of attribute 1aaa.

    attribute_2 : type
                  Description of attribute 2.


    References
    ----------
    .. [pataki] Pataki, A., Cerfon, A. J., Freidberg, J. P., Greengard, L., O'Neil, M. (2013).
                A fast, high-order solver for the Grad-Shafranov equation.
                Journal of Computational Physics, 243, 28-45. doi:10.1016/j.jcp.2013.02.045
    .. [cerfon] A. J. Cerfon and J. P. Freidberg,
                "One size fits al" analytic solutions to the Grad-Shafranov equation",
                Physics of Plasmas, vol. 17, no. 3, p. 032502, Mar. 2010.

    TODO
    ----


    ....

    :First Added:   2016-04-21
    :Last Modified: 2016-04-21
    :Copyright:     Copyright (C) 2016 apalha
    :License:       GNU GPL version 3 or any later version
    """

    """
    Reviews:
        1. First implementation. (apalha, 2016-04-21)
    """


    def __init__(self,solType):
        """
        """

        """
        :First Added:   2016-04-21
        :Last Modified: 2016-04-21
        :Copyright:     Copyright (C) 2016 apalha
        :License:       GNU GPL version 3 or any later version
        """

        """
        Reviews:
        1. First implementation. (apalha, 2016-04-21)
        """

        self.solType = solType

        # pre-compute all parameters associated to the solution, to optimize
        if self.solType == 'pataki_iter':
            self.parameters = {'epsilon':0.32,'kappa':1.7,'delta':0.33}
            self.parameters['d'] = pataki_eval_params(self.parameters['epsilon'],
                                                      self.parameters['kappa'],
                                                      self.parameters['delta'])

        elif self.solType == 'pataki_nstx':
            self.parameters = {'epsilon':0.78,'kappa':2.0,'delta':0.35}
            self.parameters['d'] = pataki_eval_params(self.parameters['epsilon'],
                                                      self.parameters['kappa'],
                                                      self.parameters['delta'])

        elif self.solType == 'cerfon_x_iter':
            self.parameters = {'epsilon':0.32,'kappa':1.7,'delta':0.33,'A':-0.155,'rsep':0.88,'zsep':-0.6}
            self.parameters['d'] = cerfon_eval_params(self.parameters['A'],
                                                      self.parameters['epsilon'],
                                                      self.parameters['kappa'],
                                                      self.parameters['delta'],
                                                      self.parameters['rsep'],
                                                      self.parameters['zsep'])

        elif self.solType == 'cerfon_x_nstx':
            self.parameters = {'epsilon':0.78,'kappa':2.0,'delta':0.35,'A':-0.05,'rsep':0.7,'zsep':-1.71}
            self.parameters['d'] = cerfon_eval_params(self.parameters['A'],
                                                      self.parameters['epsilon'],
                                                      self.parameters['kappa'],
                                                      self.parameters['delta'],
                                                      self.parameters['rsep'],
                                                      self.parameters['zsep'])

        else:
            raise ValueError(str(self.solType) + ' is invalid. solType must be: pataki_iter|pataki_nstx|cerfon_x_iter')


    def __call__(self,r,z):
        r"""
        Evaluates the Soloviev solution at the points (r,z).

        Usage
        -----
        .. code-block :: python

            self(r,z)


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
        psi : numpy.array, size: [N,M]
              The computation of the Soloviev solution of the magnetic flux at the points (r,z).


        References
        ----------


        :First Added:   2016-04-21
        :Last Modified: 2015-04-21
        :Copyright:     Copyright (C) 2016 apalha
        :License:       GNU GPL version 3 or any later version

        """

        """
        Reviews:
        1. First implementation. (apalha, 2016-04-21)
        """

        if self.solType == 'pataki_iter':
            return pataki(r,z,d=self.parameters['d'])

        elif self.solType == 'pataki_nstx':
            return pataki(r,z,d=self.parameters['d'])

        elif self.solType == 'cerfon_x_iter':
            return cerfon(r,z,self.parameters['A'],d=self.parameters['d'])

        elif self.solType == 'cerfon_x_nstx':
            return cerfon(r,z,self.parameters['A'],d=self.parameters['d'])


