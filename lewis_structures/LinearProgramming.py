#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
solve linear programs

This module solves linear optimization problems of the form

   maximize      c^T.x               (objective function)
   subject to    A_eq.x  = b_eq      (equality constraints)
                 A_ub.x <= b_ub      (inequality constraints)
                 x >= 0

using the simplex algorithm.

References
----------
[1] Jiri Matousek, Bernd Gaertner; "Understanding and Using Linear Programming",
    Springer, 2007

"""
from __future__ import print_function

import itertools
import numpy as np
import numpy.linalg as la
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(module)-12s] %(message)s", level='INFO') #level='DEBUG')

# Reasons why a linear program cannot be solved.
class UnboundedError(Exception):
    pass

class InfeasibleError(Exception):
    pass

class NoSuitableBasisError(Exception):
    pass

def _equational_form(c_orig, A_eq, b_eq, A_ub, b_ub):
    """
    brings a linear program into equational form

        maximize    c^T.x
        subject to  A.x <= b
                    x >= 0

    by introducing slack variables
    """
    # equality constraints
    m_eq,n_eq = A_eq.shape
    # inequality constraints
    m_ub,n_ub = A_ub.shape
    # m: number of constraints
    m = m_eq + m_ub
    # n: number of original variables + slack variables
    assert n_eq == n_ub == len(c_orig)
    n = n_eq + m_ub

    A = np.zeros((m,n), dtype=int)
    A[:m_eq,:n_eq] = A_eq
    A[m_eq:,:n_eq] = A_ub
    # Slack variables
    A[m_eq:,n_eq:] = np.eye(m_ub)

    b = np.zeros(m, dtype=int)
    b[:m_eq] = b_eq
    b[m_eq:] = b_ub

    c = np.zeros(n, dtype=int)
    c[:n_eq] = c_orig
    
    return c, A,b


class SimplexTableau:
    def __init__(self, basic, non_basic, p, Q, r, z0):
        """

        A simplex tableau is a set of m+1 linear equations with the same solutions
        as A.x = b, z=c^T.x

        .. code-block:: none

          x  = p + Q.x
           B          N
          -----------------
          z  = z0 + r^T.x
                         N

        B is the set of basic variables, N the set of non-basic variables.

        see 65 in Ref. [1]

        """
        self.basic = basic
        self.non_basic = non_basic
        self.nvar = len(self.basic) + len(self.non_basic)
        self.p = p
        self.Q = Q
        self.r = r
        self.z0 = z0
        # current solution vector
        self.x = np.zeros(self.nvar)
        self.x[self.basic] = self.p

    def __repr__(self):
        txt = "\nSimplex-Tableau: \n"
        txt += " basis= %s  x= %s  z= %s\n\n" % ([b+1 for b in self.basic], self.x, self.z0)
        for bb,b in enumerate(self.basic):
            txt += "  x%d \t = %d \t " % (b+1, self.p[bb])
            for nn,n in enumerate(self.non_basic):
                if abs(self.Q[bb,nn]) > 1.0e-10:
                    if (self.Q[bb,nn] == +1.0):
                        f = "+"
                    elif (self.Q[bb,nn] == -1.0):
                        f = "-"
                    else:
                        f = "%+2.1f*" % self.Q[bb,nn]
                    txt += " %sx%d \t" % (f, n+1)
                else:
                    txt += " \t"
            txt += "\n"
        txt += "  ----------------------------------------\n"
        txt += "  z \t = %+2.1f \t" % self.z0
            
        for nn,n in enumerate(self.non_basic):
            if abs(self.r[nn]) > 1.0e-10:
                if (self.r[nn] == +1.0):
                    f = "+"
                elif (self.r[nn] == -1.0):
                    f = "-"
                else:
                    f = "%+2.1f*" % self.r[nn]

                txt += " %sx%d \t" % (f, n+1)
            else:
                txt += " \t"
        txt += "\n"
                    
        return txt
    
    def optimal(self):
        """
        True if the simplex tableau corresponds to an optimal solution.
        """
        return np.all(self.r <= 0)

    def change_basis(self, enter, leave):
        """

        A basic variable (`leave`) leaves the basis, while a non-basic variable
        (`enter`) enters it. The simplex tableau is updated accordingly.
        
        Parameters
        ----------
        enter      :  int
          index of non-basic variable that enters the basis
        leave      :  int
          index of basic variable that leaves the basis

        Returns
        -------
        tableau    :  instance of SimplexTableau
          updated tableau with new basis

        """
        assert leave in self.basic, "leaving variable must be basic"
        assert enter in self.non_basic, "entering variable must be non-basic"
        # Find the index of the basic variable that leaves the basis
        # and update the set of basic variables.
        basic_new = []
        for i,b in enumerate(self.basic):
            if b == leave:
                l = i
                basic_new.append(enter)
            else:
                basic_new.append(b)

        # Find index of the non-basic variable that enters the basis
        # and update the set of non-basic variables.
        non_basic_new = []
        for j,n in enumerate(self.non_basic):
            if n == enter:
                e = j
                non_basic_new.append(leave)
            else:
                non_basic_new.append(n)

        logger.debug("variable %s enters basis, variable %s leaves" % (enter, leave))
        logger.debug("old basic variables : %s" % self.basic)
        logger.debug("new basic variables : %s" % basic_new)
        logger.debug("old non-basic variables : %s" % self.non_basic)
        logger.debug("new non-basic variables : %s" % non_basic_new)
                
        p,Q,r,z0 = self.p, self.Q, self.r, self.z0

        pnew, Qnew, rnew = np.zeros(p.shape), np.zeros(Q.shape), np.zeros(r.shape)

        z0new = z0 - (r[e]*p[l])/Q[l,e]

        """
        # slow implementation using explicit for loops
        for j in range(0, len(self.non_basic)):
            if j == e:
                rnew[j] = r[e]/Q[l,e]
            else:
                rnew[j] = r[j] - (r[e]*Q[l,j])/Q[l,e]

        for i in range(0, len(self.basic)):
            if i == l:
                pnew[i] = -p[l]/Q[l,e]
            else:
                pnew[i] = p[i] - (Q[i,e]*p[l])/Q[l,e]

        for i in range(0, len(self.basic)):
            for j in range(0, len(self.non_basic)):
                if (i == l) and (j == e):
                    Qnew[i,j] = 1.0/Q[l,e]
                else:
                    if (i == l):
                        Qnew[i,j] = -Q[l,j]/Q[l,e]
                    elif (j == e):
                        Qnew[i,j] = Q[i,e]/Q[l,e]
                    else:
                        Qnew[i,j] = Q[i,j] - (Q[i,e]*Q[l,j])/Q[l,e]
        """
        # faster implementation using numpy arrays
        rnew    = r - (r[e]*Q[l,:])/Q[l,e]
        rnew[e] = r[e]/Q[l,e]

        pnew    = p - (Q[:,e]*p[l])/Q[l,e]
        pnew[l] = -p[l]/Q[l,e]

        Qnew = Q - np.outer(Q[:,e],Q[l,:])/Q[l,e]
        Qnew[l,:] = -Q[l,:]/Q[l,e]
        Qnew[:,e] =  Q[:,e]/Q[l,e]
        #Qnew[l,e] = 1.0/Q[l,e]
        Qnew[l,e] = Q[l,e]**(-1)
        
        new_tableau = SimplexTableau(basic_new, non_basic_new, pnew, Qnew, rnew, z0new)
        logger.debug(str(new_tableau))
        return new_tableau
    
def linear_independent_rows(A, tol=1.0e-10):
    """
    find set of linear independent rows of A

    Parameters
    ----------
    A          :  ndarray (shape (m,n))
       m x n matrix

    Optional
    --------
    tol        :  float
       threshold for considering small numbers as zero when determining
       the rank of A

    Returns
    -------
    idx        :   list of int
       indices into rows of A that constitute the linear independent set
    """
    A = A.astype(float)
    # number of rows
    nrows = A.shape[0]
    # set of orthogonalized vectors that span the same vector space
    # as the rows of A
    basis = []
    # indices into rows of A which form a linearly independent basis
    idx = []

    ## Gram-Schmidt orthogonalization
    #
    # loop over row of A
    for j in range(0, A.shape[0]):
        # project the j-th row vector on the basis vectors generated by the
        # Gram-Schmidt process
        u = A[j,:]
        for i in range(0, len(basis)):
            u -= np.dot(basis[i],u)*basis[i]
        norm_u = la.norm(u)
        # If the remainder u - P.u is not zero, the vector
        # A[j,:] is retained.
        if norm_u > tol:
            u /= la.norm(u)
            basis.append(u)
            idx.append(j)
            
    # row rank of matrix A
    rank = len(basis)
    logger.info("selected %d out of %d linearly independent rows" % (rank, nrows))

    return idx
    
class LinearProgram(object):
    def __init__(self, c_orig, A_eq, b_eq, A_ub, b_ub):
        self.c_orig = c_orig
        logger.info("  Presolver steps:")
        logger.info("1) remove linearly dependent equality constraints")
        idx = linear_independent_rows(A_eq)
        self.A_eq = A_eq[idx,:]
        self.b_eq = b_eq[idx]
        # What about linearly dependent inequality constraints? Selecting linearly independent
        # rows from A_ub does not work. Constraining inequalities may be added, but only if
        # all coefficients are non-negative. For example, subtracting the two inequalities
        #  (1) x1 + x2 <= b1
        #  (2) x1 + x2 <= b2
        # does not (!!) imply that
        # (2)-(1)   0 <= b2-b1
        self.A_ub = A_ub
        self.b_ub = b_ub
        # number of equality constraints in original problem
        self.n_eq = len(self.b_eq)
        # number of inequality constraints in original problem = number of slack variables
        self.n_ub = len(self.b_ub)
        
        logger.info("2) bring linear program into equational form")
        self.c, self.A, self.b = _equational_form(self.c_orig,
                                                  self.A_eq, self.b_eq,
                                                  self.A_ub, self.b_ub)
        
        logger.info("3) remove linearly dependent rows from equational form")
        idx = linear_independent_rows(self.A)
        self.A = self.A[idx,:]
        self.b = self.b[idx]

        self.m,self.n = self.A.shape
        assert self.m <= self.n

        # indices of all variables
        self.indices = range(self.n)
                
    def _set_initial_basis(self, basic):
        self.tableau = self._set_basis(basic)
        
    def _set_basis(self, basic):
        """

        set new basic variables

        Parameters
        ----------
        basic    :  list
          indices of basic variables

        Returns
        -------
        tableau  :   instance of SimplexTableau
          simplex tableau belonging to the new basis

        """
        non_basic = list(set(self.indices) - set(basic))
        ABinv = la.inv(self.A[:,basic])
        p = np.dot(ABinv, self.b)
        Q = -np.dot(ABinv, self.A[:,non_basic])
        z0 = np.dot(self.c[basic].T, np.dot(ABinv, self.b))
        r = self.c[non_basic] + np.dot(Q.T, self.c[basic])

        ### DEBUG
        """
        # Maybe this helps avoiding numerical errors?
        from fractions import Fraction
        f = np.vectorize(lambda x: Fraction(x))
        p, Q, r, z0 = f(p), f(Q), f(r), f(z0)
        """
        ###
        
        tableau = SimplexTableau(basic, non_basic, p, Q, r, z0)
        
        logger.debug(str(tableau))
        return tableau

    def _feasible_basis(self):
        """

        find an initial feasible basis, if the linear problem is infeasible an
        InfeasibleError is raised.


        Returns
        -------
        basic   :  list of basic variables corresponding to a feasible basic solution

        """
        # If there are only inequality constraints, the slack variables
        # serve as a feasible basis
        if self.n_eq == 0:
            logger.info("Linear program does not contain equality constraints,")
            logger.info("slack variable constitute an initial feasible basis.")
            # The number of slack variables equals the number of inequality
            # constraints in the original problem. The slack variables
            # correspond to the last n_ub variables.
            basic = self.indices[-self.n_ub:]
        # Otherwise we have to construct an auxiliary linear program the optimal
        # solution of which is a feasible solution of the original problem. For
        # each equality constraint a new variable is introduced that measures the
        # deviation of the constraint:
        #
        #  1) Arrange A and b such that b >= 0
        #
        #    sum_j  A[i,j] x_j = b[i]
        #
        #  2) add new variables x_(n+1),...,x_(n+n_eq) and
        #  replace equality constraints by
        #
        #    sum_j A[i,j] x_j  +  x_(n+j)  = b[i]
        #
        #  These constraints can always be satisfied by choosing x_(n+j) large enough.
        #
        #  3) maximize
        #
        #       z =  - x(n+1) - x(n+2) - ... - x(n+m)
        #
        #    subject to   A_aux . x_aux = b
        #
        #  If the optimal solution has z=0, all equality constraints are satisfied and
        #  the problem is feasible.
        # 
        else:
            logger.info("solving auxiliary linear program to obtain initial feasible solution")
            m, n = self.A.shape
            # A_aux = (A|I)
            Aaux = np.zeros((m,n+m), dtype=int)
            Aaux[:,:n] = self.A
            Aaux[:,n:] = np.eye(m, dtype=int)
            baux = np.copy(self.b)
            caux = np.zeros(n+m, dtype=int)
            # 1) ensure b >= 0            
            for i in range(0, m):
                if baux[i] < 0:
                    baux[i] *= -1
                    Aaux[i,:] *= -1
                caux[n+i] = -1

            lp_aux = LinearProgram(caux, Aaux, baux, np.zeros((0,m+n)), np.zeros(0))
            # Use the auxiliary variables as initial basic variables
            lp_aux._set_initial_basis(list(range(n,n+m)))

            xaux, z0aux = lp_aux.solve()
            if z0aux < 0:
                raise InfeasibleError("Linear program has no feasible solution")
            # Since the objective function is the negative of non-negative variables
            # its maximum is 0.
            assert z0aux == 0
            # All auxiliary variables should be zero, meaning the equality constraints
            # are fulfilled exactly.
            assert np.all(xaux[n:] == 0)

            x = xaux[:n]

            # All non-zero variables must belong to the basic set.
            non_zero = list(np.where(x > 0)[0])
            # Some basic variables might be zero as well. So we add
            # as many zero variables as needed to get a complete set of basis
            # variables.
            zero = list(np.where(x == 0)[0])
            # reordered column indices, non-zero (basic) variables come first, followed
            # by zero (basic and non-basic) variables.
            columns = np.array(non_zero + zero)
            # The columns of A are reordered such that those belonging to non-zero
            # variables in the feasible solution come first followed by the zero variables.
            # Then we select `m` linear independent columns going from left to right,
            # so that all columns of non-zero variables are added before adding zero variables.
            idx = linear_independent_rows(self.A[:,columns].T)
            basic = columns[idx]
            #
            assert abs(la.det(self.A[:,basic])) > 1.0e-10, "Columns of A are not linearly independent"
        basic.sort()
        
        logger.debug("initial basis: %s" % [b+1 for b in basic])
        return list(basic)
        
    def _next_basis(self):
        t = self.tableau
        # Which non-basic variable should enter the basis?
        for nn,n in enumerate(t.non_basic):
            if t.r[nn] > 0:
                # What is the maximum amount by which x_n can be increased,
                # so that all basic variables are still non-negative
                inc = None
                # index of most restrictive constraint
                br = None
                bbr = None
                for bb,b in enumerate(t.basic):
                    if (t.Q[bb,nn] < 0):
                        # If the non-basic variable x_n is increased by -p[b]/Q[b,n] >= 0
                        # the basic variable b becomes 0.
                        inc_b = -t.p[bb]/t.Q[bb,nn]
                        if inc is None or inc > inc_b:
                            inc = inc_b
                            br = b
                            bbr = bb
                if (br is None):
                    # None of the constraints prevents from increasing x_n.
                    raise UnboundedError("Unbound problem")
                logger.debug("non-basic variable x%d can be increased at most by %s because basic variable x%d has to stay non-negative" % (n+1, inc, br+1))
                
                #if not int(round(t.Q[bbr,nn])) == -1:
                #    continue
                
                enter = n
                leave = br
                        
                basic = t.basic[:]
                basic.remove(leave)
                basic.append(enter)
                
                basic.sort()

                logger.debug("  x%d enters basis and x%d leaves " % (enter+1,leave+1))

                # We have sucessfully found a new basis, update simple tableau and exit loop.
                self.tableau = self.tableau.change_basis(enter, leave)
                break
            else:
                logger.debug("Increasing variable x%d does not increase objective function!" % (n+1))
        else:
            raise NoSuitableBasisError("I could not find suitable new basis!")

    def _enumerate_optimal_bases(self, tableau, solutions=[], visited_bases={}, depth=0, max_depth=0):
        """

        exhaustively enumerate all bases up to certain depth that do not change the objective function

        Parameters
        ----------
        tableau       :  instance of SimplexTableau
          parent simplex tableau, all child tableaus that are obtained by exchanging a basic variable
          for a non-basic one such that the objective function does not change, are traversed.
        solutions     :  list
          list of solution vectors that have been found so far at higher levels
        visited_bases :  set
          bases that have been visited so far
        depth         :  int
          current depth 
        max_depth     :  int
          maximum recursion depth

        """
        # Which non-basic variable should enter the basis?
        for nn,n in enumerate(tableau.non_basic):
            # objective function should not depend on x_n
            if tableau.r[nn] == 0:
                # What is the maximum amount by which x_n can be increased,
                # so that all basic variables are still non-negative
                inc = None
                # index of most restrictive constraint
                br = None
                bbr = None
                for bb,b in enumerate(tableau.basic):
                    if (tableau.Q[bb,nn] < 0):
                        # If the non-basic variable x_n is increased by -p[b]/Q[b,n] >= 0
                        # the basic variable b becomes 0.
                        inc_b = -tableau.p[bb]/tableau.Q[bb,nn]
                        if inc is None or inc > inc_b:
                            inc = inc_b
                            br = b
                            bbr = bb
                if (br is None):
                    # None of the constraints prevents from increasing x_n.
                    raise UnboundedError("Unbound problem")
                logger.debug("non-basic variable x%d can be increased at most by %s because basic variable x%d has to stay non-negative" % (n+1, inc, br+1))
                
                #if not int(round(t.Q[bbr,nn])) == -1:
                #    continue
                
                enter = n
                leave = br
                        
                new_basic = tableau.basic[:]
                new_basic.remove(leave)
                new_basic.append(enter)
                
                new_basic.sort()

                if tuple(new_basic) in visited_bases:
                    logger.debug("  basis has been visited already")
                    continue
                
                logger.debug("  x%d enters basis and x%d leaves " % (enter+1,leave+1))

                new_tableau = tableau.change_basis(enter, leave)
                #new_tableau = self._set_basis(new_basic)
                logger.debug(str(new_tableau))
                # The value of objective function should not have changed. This test will fail
                # if small numerical errors cause z to be non-integer.
                assert new_tableau.z0 == tableau.z0,  "objective function changed from z= %s to z= %s" % (tableau.z0, new_tableau.z0)

                visited_bases.add(tuple(new_tableau.basic))
                
                new_x = list(new_tableau.x)
                if not new_x in solutions:
                    solutions.append(new_x)
                    logger.info("found %d solutions, current depth = %d (max. depth = %d )" % (len(solutions), depth, max_depth))
                    
                if depth < max_depth:
                    # recursively visit all bases that can be reached from new_tableau
                    # and add them
                    solutions, visited_bases = self._enumerate_optimal_bases(new_tableau,
                                                                             solutions, visited_bases,
                                                                             depth=depth+1,
                                                                             max_depth=max_depth)

        return solutions, visited_bases

    def all_optimal_solutions(self, max_depth=6):
        """

        enumerate all feasible solutions xi that are equivalent to the optimal 
        solution x* found by the simplex method in the sense that the value of the 
        objective function does not change.

           z(x*) = z(x1) = z(x2) .... = z(xk)

        Since the number of equivalent solutions may be exponentially large, only
        solutions up to a certain depth (`max_depth`) are retrieved.

        Optional
        --------
        max_depth : int
           maximum recursion depth while searching for equivalent optimal solutions

        Returns
        -------
        xs      :   list of ndarrays
           list of all optimal solutions
        z       :   float
           value of objective function of optimal solutions, only one number is returned
           since it is the same for all solutions

        """
        logger.info("Looking for equivalent optimal solutions ...")
        # Current tableau should contain optimal solution
        assert hasattr(self, "tableau") and self.tableau.optimal(), \
            "You have to find one optimal solution first using the simplex method before you can enumerate other optimal solutions."
        z0 = self.tableau.z0

        # List of unique solutions
        solutions = [list(self.tableau.x)]
        # We keep a set of bases that have been visited already to avoid cycling. It is important
        # so use a `set` instead of a `list` since looking since the complexity of testing membership
        # (basis in visited_bases) scales on average like O(1) for sets but like O(n) for lists.
        visited_bases = {tuple(self.tableau.basic)}

        # All basic solutions that do not change the objective function are traveresed recursively.
        solutions, visited_bases = self._enumerate_optimal_bases(self.tableau, solutions, visited_bases,
                                                                 max_depth=max_depth)

        logger.info("Found %d equivalent optimal solutions after visiting %d bases." % (len(solutions), len(visited_bases)))
        return solutions, z0
        
    def solve(self):
        if not hasattr(self, "tableau"):
            # If no feasible tableau exists so far,
            # find initial feasible solution.
            self._set_initial_basis(self._feasible_basis())

            
        logger.info("running simplex algorithm")
        # counts number of iterations
        niter = 0
        while not self.tableau.optimal():
            self._next_basis()

            niter += 1
            if (niter % 100 == 0):
                logger.info("%d iterations" % niter)
                
        logger.info("SOLUTION FOUND")
        return self.tableau.x, self.tableau.z0

######### TESTS ################################
    
def test_example_5p1():
    # example 5.1. from [1]
    c = np.array([1,1])
    A_eq = np.zeros((0,2))
    b_eq = np.zeros(0)
    A_ub = np.array([[-1,1],
                     [ 1,0],
                     [ 0,1]])
    b_ub = np.array([1,
                     3,
                     2])

    lp = LinearProgram(c, A_eq, b_eq, A_ub, b_ub)
    #t = lp._set_initial_basis([2,3,4])
    #t = lp._set_initial_basis(lp._feasible_basis())
    x_opt, z_opt = lp.solve()

    print("x*= ", x_opt)
    print("z*= ", z_opt)

def test_unbounded():
    """
    This function should fail with an UnboundedError exception.
    """
    # example for unbounded problem from section 5.2 in [1]
    c = np.array([1,0])
    A_eq = np.zeros((0,2))
    b_eq = np.zeros(0)
    A_ub = np.array([[ 1,-1],
                     [-1,+1]])
    b_ub = np.array([1,
                     2])

    lp = LinearProgram(c, A_eq, b_eq, A_ub, b_ub)
#    t = lp._set_initial_basis([2,3])
    x_opt, z_opt = lp.solve()

    print("x*= ", x_opt)
    print("z*= ", z_opt)

def test_degeneracy():
    # example (5.2) from [1]
    c = np.array([0,1])
    A_eq = np.zeros((0,2))
    b_eq = np.zeros(0)
    A_ub = np.array([[-1,+1],
                     [ 1, 0]])
    b_ub = np.array([0,
                     2])

    lp = LinearProgram(c, A_eq, b_eq, A_ub, b_ub)
#    t = lp._set_initial_basis([2,3])
    x_opt, z_opt = lp.solve()

    print("x*= ", x_opt)
    print("z*= ", z_opt)


__all__ = ["LinearProgram", "UnboundError", "InfeasibleError"]
    
if __name__ == "__main__":
    test_example_5p1()
    #test_unbounded()
    #test_degeneracy()
    
    
    
    
