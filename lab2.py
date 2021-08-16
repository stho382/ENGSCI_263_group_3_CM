# Use this as a guide

# imports
import numpy as np
from matplotlib import pyplot as plt
import math


def ode_model(t, x, q, a, b, x0):
    """Return the derivative dx/dt at time, t, for given parameters.

    Parameters:
    -----------
    t : float
        Independent variable.
    x : float
        Dependent variable.
    q : float
        Source/sink rate.
    a : float
        Source/sink strength parameter.
    b : float
        Recharge strength parameter.
    x0 : float
        Ambient value of dependent variable.

    Returns:
    --------
    dxdt : float
        Derivative of dependent variable with respect to independent variable.

    Notes:
    ------
    None

    Examples:
    ---------
    >>> ode_model(0, 1, 2, 3, 4, 5)
    22

    """
    return a * q - b * (x - x0)


def solve_ode(f, t0, t1, dt, x0, pars):
    """Solve an ODE numerically.

    Parameters:
    -----------
    f : callable
        Function that returns dxdt given variable and parameter inputs.
    t0 : float
        Initial time of solution.
    t1 : float
        Final time of solution.
    dt : float
        Time step length.
    x0 : float
        Initial value of solution.
    pars : array-like
        List of parameters passed to ODE function f.

    Returns:
    --------
    t : array-like
        Independent variable solution vector.
    x : array-like
        Dependent variable solution vector.

    Notes:
    ------
    ODE should be solved using the Improved Euler Method.

    Function q(t) should be hard coded within this method. Create duplicates of
    solve_ode for models with different q(t).

    Assume that ODE function f takes the following inputs, in order:
        1. independent variable
        2. dependent variable
        3. forcing term, q
        4. all other parameters
    """
    q = -1
    t = np.zeros(int((t1 - t0) / dt) + 1)
    x = np.zeros(len(t))
    t[0] = t0
    x[0] = x0
    for i in range(len(t) - 1):
        d1 = f(t[i], x[i], q, *pars)
        d2 = f(t[i] + dt, x[i] + dt * d1, q, *pars)
        x[i + 1] = x[i] + dt * (d1 + d2) / 2
        t[i + 1] = t[i] + dt

    return t, x


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
    This function called within if __name__ == "__main__":

    It should contain commands to obtain analytical and numerical solutions,
    plot these, and either display the plot to the screen or save it to the disk.

    """

    t0 = 0
    t1 = 10
    dt = 0.1
    x0 = 0
    q = -1
    pars = [1, 1, 0]
    t, x = solve_ode(ode_model, t0, t1, dt, x0, pars)
    t2 = np.zeros(len(t))
    x2 = np.zeros(len(t))
    t2[0] = t0
    x2[0] = x0

    error = np.zeros(len(t))

    for i in range(len(t2) - 1):
        t2[i + 1] = t[i] + dt
        x2[i + 1] = q * pars[0] / pars[1] * (1 - np.exp(-pars[1] * t2[i + 1]) + x0)
        error[i + 1] = (x2[i + 1] - x[i + 1]) / x2[i + 1]

    inv_dt = list(range(10, 30, 1))
    dt_conv = [10 / i for i in inv_dt]
    inv_dt = [i / 10 for i in inv_dt]
    error = np.delete(error, 0)
    t = np.delete(t, 0)
    error = [math.log10(i) for i in error]

    conv = []
    for j in dt_conv:
        t3, x3 = solve_ode(ode_model, t0, t1, j, x0, pars)
        conv.append(x3[-1])

    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    ax1.plot(t2, x2, "r-", label="analytic")
    ax1.plot(t2, x, "bx-", label="numerical")

    ax2.plot(t, error, "g-")

    ax3.plot(inv_dt, conv, "k*")

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig("lab2_plot.png", dpi=300)


def solve_ode_kettle(f, t0, t1, dt, x0, pars):
    """Solve an ODE numerically.

    Parameters:
    -----------
    f : callable
        Function that returns dxdt given variable and parameter inputs.
    t0 : float
        Initial time of solution.
    t1 : float
        Final time of solution.
    dt : float
        Time step length.
    x0 : float
        Initial value of solution.
    pars : array-like
        List of parameters passed to ODE function f.

    Returns:
    --------
    t : array-like
        Independent variable solution vector.
    x : array-like
        Dependent variable solution vector.

    Notes:
    ------
    ODE should be solved using the Improved Euler Method.

    Function q(t) should be hard coded within this method. Create duplicates of
    solve_ode for models with different q(t).

    Assume that ODE function f takes the following inputs, in order:
        1. independent variable
        2. dependent variable
        3. forcing term, q
        4. all other parameters
    """

    t = np.zeros(int((t1 - t0) / dt) + 1)
    t[0] = t0
    for i in range(len(t) - 1):
        t[i + 1] = t[i] + dt
    q = interpolate_kettle_heatsource(t)
    x = np.zeros(len(t))
    x[0] = x0
    for i in range(len(t) - 1):
        d1 = f(t[i], x[i], q[i], *pars)
        d2 = f(t[i] + dt, x[i] + dt * d1, q[i + 1], *pars)
        x[i + 1] = x[i] + dt * (d1 + d2) / 2

    return t, x


def load_kettle_temperatures():
    """Returns time and temperature measurements from kettle experiment.

    Parameters:
    -----------
    none

    Returns:
    --------
    t : array-like
        Vector of times (seconds) at which measurements were taken.
    T : array-like
        Vector of Temperature measurements during kettle experiment.

    Notes:
    ------
    It is fine to hard code the file name inside this function.

    Forgotten how to load data from a file? Review datalab under Files/cm/
    engsci233 on the ENGSCI263 Canvas page.
    """
    data = np.genfromtxt(
        "263_Kettle_Experiment_22-07-19.csv", delimiter=",", skip_header=7
    )
    time = data[:, 0]
    # voltage = data[:,1]
    # current = data[:,2]
    temperature = data[:, 3]

    return time, temperature


def interpolate_kettle_heatsource(t):
    """Return heat source parameter q for kettle experiment.

    Parameters:
    -----------
    t : array-like
        Vector of times at which to interpolate the heat source.

    Returns:
    --------
    q : array-like
        Heat source (Watts) interpolated at t.

    Notes:
    ------
    This doesn't *have* to be coded as an interpolation problem, although it
    will help when it comes time to do your project if you treat it that way.

    Linear interpolation is fine for this problem, no need to go overboard with
    splines.

    Forgotten how to interpolate in Python, review sdlab under Files/cm/
    engsci233 on the ENGSCI263 Canvas page.
    """
    # suggested approach
    # hard code vectors tv and qv which define a piecewise heat source for your kettle
    # experiment
    # use a built in Python interpolation function

    tv = [0, 30, 180, 210, 360, 390, 1200]
    qv = [0, 197, 197, 375.35, 375.35, 0, 0]
    return np.interp(t, tv, qv)


def plot_kettle_model():
    """Plot the kettle LPM over top of the data.

    Parameters:
    -----------
    none

    Returns:
    --------
    none

    Notes:
    ------
    This function called within if __name__ == "__main__":

    It should contain commands to read and plot the experimental data, run and
    plot the kettle LPM for hard coded parameters, and then either display the
    plot to the screen or save it to the disk.

    """
    data = np.genfromtxt(
        "263_Kettle_Experiment_22-07-19.csv", delimiter=",", skip_header=7
    )
    time = data[:, 0]
    # voltage = data[:,1]
    # current = data[:,2]
    temperature = data[:, 3]
    t0 = 0
    t1 = 1200
    dt = 10
    x0 = 22
    pars = [0.00053, 0.00067, 22]
    f = ode_model
    t, x = solve_ode_kettle(f, t0, t1, dt, x0, pars)

    f, ax1 = plt.subplots(nrows=1, ncols=1)

    ax1.plot(t, x, "b-", label="numerical")
    ax1.plot(time, temperature, "ro")

    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig("lab2_plot.png", dpi=300)
    pass


if __name__ == "__main__":
    plot_benchmark()
    plot_kettle_model()
