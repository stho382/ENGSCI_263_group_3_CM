from numpy.core.shape_base import atleast_1d
from analytic_solution import *
from ODE_Model_Function import *
import math


def plot_benchmark():
    """Compare analytical and numerical solutions.

    Parameters:
    -----------
    none

    Returns:
    --------
    none

    Notes:
    ------
    This function compares the numerical solution to the analytic solution for constant production and negligible conduction.
    ie: dqdt = 0, bt = 0

    It contains commands to obtain analytical and numerical solutions,
    plot these, and either display the plot to the screen or save it to the disk.

    """
    # Pressure ODE
    t0 = 0
    t1 = 10
    dt = 0.1
    x0 = 0
    P0 = 1
    ap = 1
    bp = 1
    cp = 0
    q = 100
    dqdt = 0

    # solve numerically
    pars = [q, dqdt, P0, ap, bp, cp]

    t, P = solve_pressure_ode(ode_pressure_model, t0, t1, dt, x0, pars)
    t2 = np.zeros(len(t))
    P2 = np.zeros(len(t))
    t2[0] = t0
    P2[0] = x0

    error = np.zeros(len(t))

    # analytic solution
    for i in range(len(t2) - 1):
        t2[i + 1] = t[i] + dt
        P2[i + 1] = pressure_analytic(t[i + 1], 0, *pars)

        # measure relative error
        error[i + 1] = (P2[i + 1] - P[i + 1]) / P2[i + 1]

    # convert to log scale currently broken
    # error = [math.log10(i) for i in error]

    # initial and analytic start at the same value so useless to compare them
    error = np.delete(error, 0)

    # Convergence analysis

    # set range of step sizes
    inv_dt = list(range(10, 30, 1))
    dt_conv = [10 / i for i in inv_dt]
    inv_dt = [i / 10 for i in inv_dt]

    t = np.delete(t, 0)

    conv = []
    for j in dt_conv:
        t3, x3 = solve_pressure_ode(ode_pressure_model, t0, t1, j, x0, pars)
        conv.append(x3[-1])

    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    ax1.plot(t2, P2, "r-", label="analytic")
    ax1.plot(t2, P, "bx-", label="numerical")

    ax2.plot(t, error, "g-")

    ax3.plot(inv_dt, conv, "k*")

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig("lab2_plot.png", dpi=300)

    t0 = 0
    t1 = 10
    dt = 0.1
    x0 = 0

    T0 = 0
    ap = 1
    bp = 1
    cp = 0
    at = 1
    bt = 1
    Tc = 1
    q = 100
    dqdt = 0
    Tt = 1
    T = 0

    # solve numerically
    pars = [Tt, Tc, T0, at, bt, ap, bp, P, P0]

    t, T = solve_temperature_ode(ode_temperature_model, t0, t1, dt, x0, pars)
    t2 = np.zeros(len(t))
    T2 = np.zeros(len(t))
    t2[0] = t0
    T2[0] = x0

    error = np.zeros(len(t))

    # analytic solution
    for i in range(len(t2) - 1):
        t2[i + 1] = t[i] + dt
        T2[i + 1] = temperature_analytic(t[i + 1], 0, *pars, q)

        # measure relative error
        error[i + 1] = (P2[i + 1] - P[i + 1]) / P2[i + 1]

    # convert to log scale currently broken
    # error = [math.log10(i) for i in error]

    # initial and analytic start at the same value so useless to compare them
    error = np.delete(error, 0)

    # Convergence analysis

    # set range of step sizes
    inv_dt = list(range(10, 30, 1))
    dt_conv = [10 / i for i in inv_dt]
    inv_dt = [i / 10 for i in inv_dt]

    t = np.delete(t, 0)

    conv = []
    for j in dt_conv:

        t3, x3 = solve_temperature_ode(ode_temperature_model, t0, t1, j, x0, pars)
        conv.append(x3[-1])

    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    ax1.plot(t2, T2, "r-", label="analytic")
    ax1.plot(t2, T, "bx-", label="numerical")

    ax2.plot(t, error, "g-")

    ax3.plot(inv_dt, conv, "k*")

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig("lab2_plot.png", dpi=300)


plot_benchmark()
