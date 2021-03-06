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
    dt = 0.2
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

    fPT = plt.figure(constrained_layout=True, figsize=(11, 8))
    fPT.suptitle("Pressure and Temperature Benchmarking", fontsize=20)

    # Creating 2x1 subpfigure - one for pressue and one for temperature
    subfigsPT = fPT.subfigures(nrows=2, ncols=1)

    # ----------#
    # Pressure #
    # ----------#

    # Adding plots to subfig 1
    subfigsPT[0].suptitle("Pressure Benchmarks", fontsize=15, fontweight="bold")

    axP = subfigsPT[0].subplots(nrows=1, ncols=3)

    axP[0].plot(t2, P2, "b-", label="analytic")
    axP[0].plot(t2, P, "rx", label="numerical")

    axP[0].legend()

    # Analytical vs Numerical Plot
    axP[0].set_title("Analytic vs Numerical solution")
    axP[0].set_xlabel("Time")
    axP[0].set_ylabel("Pressure")
    axP[0].hlines(y=1.013475893998171, xmin=-1, xmax =11, colors='k', linestyles='--')
    axP[0].set_xlim(-0.5,10.5)
    # Relative Error Plot
    axP[1].plot(t, error, "r-")
    axP[1].hlines(y=0, xmin=-1, xmax =11, colors='k', linestyles='--')
    axP[1].set_title("Relative Error, (analytic - numerical) / analytic")
    axP[1].set_xlabel("Time")
    axP[1].set_ylabel("Error")
    axP[1].set_xlim(-0.5,10.5)
    # Convergence Testing Plot
    axP[2].plot(inv_dt, conv, "r*")
    axP[2].hlines(y=1.013475893998171, xmin=-1, xmax =11, colors='k', linestyles='--')
    axP[2].set_title("Convergence testing")
    axP[2].set_xlabel("1/step size")
    axP[2].set_ylabel("Final value")
    axP[2].set_xlim(0.9,3)
    
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
    T = 0

    # solve numerically
    pars = [Tc, T0, at, bt, ap, bp, P, P0]

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

    # -------------#
    # Temperature #
    # -------------#

    # Adding plots to subfig 2
    subfigsPT[1].suptitle("Temperature Benchmarks", fontsize=15, fontweight="bold")

    axT = subfigsPT[1].subplots(nrows=1, ncols=3)

    axT[0].plot(t2, T2, "b-", label="analytic")
    axT[0].plot(t2, T, "rx", label="numerical")
    axT[0].hlines(y=3.0006619445853655, xmin=-1, xmax =11, colors='k', linestyles='--')
    axT[0].legend()
    axT[0].set_xlim(-0.5,10.5)
    # Analytical v Numerical Plot
    axT[0].set_title("Analytic vs Numerical solution")
    axT[0].set_xlabel("Time")
    axT[0].set_ylabel("Temperature")
    
    # Relative Error Plot
    axT[1].plot(t, error, "r-")
    axT[1].hlines(y=0, xmin=-1, xmax =11, colors='k', linestyles='--')
    axT[1].set_title("Relative Error, (analytic - numerical) / analytic")
    axT[1].set_xlabel("Time")
    axT[1].set_ylabel("Error")
    axT[1].set_xlim(-0.5,10.5)

    # Convergence Testing Plot
    axT[2].plot(inv_dt, conv, "r*")
    axT[2].hlines(y=3.0006619445853655, xmin=-1, xmax =11, colors='k', linestyles='--')
    axT[2].set_xlim(0.9,3)
    axT[2].set_title("Convergence testing")
    axT[2].set_xlabel("1/step size")
    axT[2].set_ylabel("Final value")

    # -----------------------#
    # Show Plot / Save Plot  #
    # -----------------------#

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        fPT.savefig("Pressure_and_Temperature_Benchmarks.png")