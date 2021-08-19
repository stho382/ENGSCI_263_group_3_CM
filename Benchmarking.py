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
    x0 = 3
    P0 = 3
    ap = 1
    bp = 0.5
    cp = 0
    q = 1
    dqdt = 0

    # solve numerically
    pars = [q, dqdt, P0, ap, bp, cp]

    t, P = solve_pressure_ode(ode_pressure_model, t0, t1, dt, x0, pars, benchmark=True)
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

    # We need to store these for when we do the temperature ode convergence analysis
    Pressure_conv = []
    conv = []
    for j in dt_conv:
        t3, x3 = solve_pressure_ode(
            ode_pressure_model, t0, t1, j, x0, pars, benchmark=True
        )
        conv.append(x3[-1])
        Pressure_conv.append(x3)

    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    ax1.plot(t2, P2, "r-", label="analytic")
    ax1.plot(t2, P, "bx-", label="numerical")

    ax2.plot(t, error, "g-")

    ax3.plot(inv_dt, conv, "k*")

    ax1.legend()

    ax1.set_title("Analytic vs Numerical solution")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Pressure")

    ax2.set_title("Relative Error, (analytic - numerical) / analytic")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error")

    ax3.set_title("Convergence testing")
    ax3.set_xlabel("1/step size")
    ax3.set_ylabel("Final value")
    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig("Pressure benchmark.png", dpi=300)

    # Temperature ODE
    # The analytic solution takes q as an input but the numerical solution takes Pressure
    # This means we need pressure values for given q from the pressure ODE first

    t0 = 0
    t1 = 10
    x0 = 5

    T0 = 5
    ap = 1
    bp = 0.5
    cp = 0
    at = 1
    bt = 0
    Tc = 3
    q = 1
    dqdt = 0
    Tt = 3
    T = 0

    # solve numerically
    pars = [Tt, Tc, T0, at, bt, ap, bp, P, P0]

    t, T = solve_temperature_ode(ode_temperature_model, t0, t1, dt, x0, pars)
    t2 = np.zeros(len(t))
    T2 = np.zeros(len(t))
    t2[0] = t0
    T2[0] = x0

    error = np.zeros(len(t))

    # analytic solution. note that P in pars is from the numerical solution for pressure
    for i in range(len(t2) - 1):
        t2[i + 1] = t[i] + dt
        T2[i + 1] = temperature_analytic(t[i + 1], 0, *pars, q)

        # measure relative error
        error[i + 1] = (T2[i + 1] - T[i + 1]) / T2[i + 1]

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
    for j in range(len(dt_conv)):

        pars[-2] = Pressure_conv[j]
        t3, x3 = solve_temperature_ode(
            ode_temperature_model, t0, t1, dt_conv[j], x0, pars
        )
        conv.append(x3[-1])

    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    ax1.plot(t2, T2, "r-", label="analytic")
    ax1.plot(t2, T, "bx-", label="numerical")

    ax2.plot(t, error, "g-")

    ax3.plot(inv_dt, conv, "k*")

    ax1.legend()

    ax1.set_title("Analytic vs Numerical solution")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature")

    ax2.set_title("Relative Error, (analytic - numerical) / analytic")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error")

    ax3.set_title("Convergence testing")
    ax3.set_xlabel("1/step size")
    ax3.set_ylabel("Final value")
    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig("Temperature benchmark.png", dpi=300)


plot_benchmark()
