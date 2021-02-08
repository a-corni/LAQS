import inspect
import numpy as np
from .MC_RK45 import RK45
from .MC_common import EPS, OdeSolution
from .MC_OdeSolver import OdeSolver
from scipy.optimize import OptimizeResult

#methods that can be used
METHODS = {'RK45': RK45}


MESSAGES = {0: "The solver successfully reached the end of the integration interval.", 1: "A termination event occurred."}


class OdeResult(OptimizeResult):
    pass


def MC_solve_ivp(fun, fun_scattering, proba_scattering, t_span, y0, method='RK45', t_eval=None, atol=1e-6, rtol=1e-10):
    """Solve an initial value problem for a system of ODEs.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system, with no stochastic force. The calling signature is ``fun(t, y)``.
        Here `t` is a scalar, and there are two options for the ndarray `y`:
        It can either have shape (n,); then `fun` must return array_like with
        shape (n,). Alternatively, it can have shape (n, k); then `fun`
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in `y`. 
    fun_scattering : callable
        Stocastic part of the right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here `t` is a scalar, and there are two options for the ndarray `y`:
        It can either have shape (n,); then `fun` must return array_like with
        shape (n,). Alternatively, it can have shape (n, k); then `fun`
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in `y`.
    proba_scattering : callable
        Decides whether to call the stocastic function or not. The calling signature is ``fun(t, y)``.
        Here `t` is a scalar, and there are two options for the ndarray `y`:
        It can either have shape (n,) or it can have shape (n, k). In both case `fun`
        must return a positive number.
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    method : string or `OdeSolver`, optional
        Integration method to use:
            * 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
              The error is controlled assuming accuracy of the fourth-order
              method, but steps are taken using the fifth-order accurate
              formula (local extrapolation is done). A quartic interpolation
              polynomial is used for the dense output [2]_. Can be applied in
              the complex domain.
            * 'BDF': Implicit multi-step variable-order (1 to 5) method based
              on a backward differentiation formula for the derivative
              approximation [5]_. The implementation follows the one described
              in [6]_. A quasi-constant step scheme is used and accuracy is
              enhanced using the NDF modification. Can be applied in the
              complex domain.
            * 'LSODA': Adams/BDF method with automatic stiffness detection and
              switching [7]_, [8]_. This is a wrapper of the Fortran solver
              from ODEPACK.
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.
    sol : `OdeSolution` or None
        Found solution as `OdeSolution` instance; None if `dense_output` was
        set to False.
    status : int
        Reason for algorithm termination:
            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
    message : string
        Human-readable description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).
    """
    
    #if the method can't be used
    if method not in METHODS and not (inspect.isclass(method) and issubclass(method, OdeSolver)):
        raise ValueError("`method` must be one of {} or OdeSolver class.".format(METHODS))

    #get the starting and ending point
    t0, tf = float(t_span[0]), float(t_span[1])

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if tf > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]
    
    #We now use MC_OdeSolver instance
    if method in METHODS:
        method = METHODS[method]

    solver = method(fun, fun_scattering, proba_scattering, t0, y0, tf, vectorized=False)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    else:
        ts = []
        ys = []

    status = None
    
    while status is None:
        
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y

        sol = None

        if t_eval is None:
            
            ts.append(t)
            ys.append(y)
            
        else:
            
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                
                t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
                
            else:
                
                t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

    message = MESSAGES.get(status, message)

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    else:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    return OdeResult(t=ts, y=ys, sol=sol,nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu, status=status, message=message, success=status >= 0)
