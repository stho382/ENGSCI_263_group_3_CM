from types import FrameType
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.npyio import load
from numpy.linalg import norm
from glob import glob
import os
from scipy.optimize import curve_fit


def load_ODE_Model_data():
    """Returns data from data file for Project 3.

            Parameters:
            -----------
            none

            Returns:
            --------

    WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT
            WaterLevel : array-like
                    Vector of water-level for the Whakarewarewa Geothermal Field.
            Yearp : array-like
                    Vector of times (Years) at which measurements were taken.
    Prodq1 : array-like
                    Vector of production rate for bore-hole 1.
    Yearq1 : array-like
                    Vector of times (Years) at which measurements were taken.
    Prodq2 : array-like
                    Vector of production rate for bore-hole 2.
    Yearq2 : array-like
                    Vector of times (Years) at which measurements were taken.
    Temp : array-like
                    Vector of Temperature measurements of the Whakarewarewa Geothermal Field.
    YearT : array-like
                    Vector of times (Years) at which measurements were taken.
    """
    # using glob to find the folder which contains the desired data file
    files = glob("data" + os.sep + "*")

    # loop to extract all the data from the files
    for file in files:
        if file == "data\\gr_p.txt":
            Data = np.genfromtxt(file, delimiter=",", skip_header=1)
            WaterLevel = Data[:, 1]
            Yearp = Data[:, 0]
        if file == "data\\gr_q1.txt":
            Data = np.genfromtxt(file, delimiter=",", skip_header=1)
            Prodq1 = Data[:, 1]
            Yearq1 = Data[:, 0]
        if file == "data\\gr_q2.txt":
            Data = np.genfromtxt(file, delimiter=",", skip_header=1)
            Prodq2 = Data[:, 1]
            Yearq2 = Data[:, 0]
        else:
            Data = np.genfromtxt(file, delimiter=",", skip_header=1)
            Temp = Data[:, 1]
            YearT = Data[:, 0]

    return WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT


WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT = load_ODE_Model_data()


def ode_pressure_model(t, P, P0, ap, bp, cp, q, dqdt):
    """
    Return the derivative dP/dt at time, t, for given parameters.

    Parameters:
    -----------
    t : float
            independent variable
    P : float
            dependent variable.
    P0 : float
            ambient value of dependent variable.
    ap : float
            Source/sink strength parameter.
    bp : float
            Recharge strength parameter.
    cp : float
            Slow strength parameter.
    q : float
            Source/sink rate.
    dqdt :
            rate of change of source/ sink rate

    Returns:
    --------
    dPdt : float
            Derivative of dependent variable with respect to independent variable.

    Notes:
    ------
    None
    """
    dPdt = -ap * q - bp * (P - P0) - cp * dqdt

    return dPdt


def ode_temperature_model(t, T, Tt, Tc, T0, at, bt, ap, bp, P, P0):
    """
    Return the derivative dT/dt at time, t, for given parameters.

    Parameters:
    -----------
    t : float
            independent variable
    T: float
            dependent variable
    Tt : float
            temperatue at given point in time
    Tc : float
            temperature of cold water
    T0 : float
            ambient value of dependent variable.
    at : float
            cold water inflow strength parameter
    bt : float
            conduction strength parameter
    ap : float
            Source/sink strength parameter
    bp : float
            Recharge strength parameter.
    P : float
            pressure value.
    P0 : float
            Ambient value of pressure.

    Returns:
    --------
    dxdt : float
            Derivative of dependent variable with respect to independent variable.

    Notes:
    ------
    None
    """

    if P > P0:
        dTdt = at * (bp / ap) * (P - P0) * (Tt - T0) - bt * (T - T0)
    else:
        dTdt = at * (bp / ap) * (P - P0) * (Tc - T0) - bt * (T - T0)
    return dTdt


def solve_pressure_ode(f, t0, t1, dt, x0, pars):
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
    """

    # initialise
    nt = int(np.ceil((t1 - t0) / dt))  # compute number of Euler steps to take
    ts = t0 + np.arange(nt + 1) * dt  # x array
    xs = 0.0 * ts  # array to store solution
    xs[0] = x0  # set initial value

    # loop that iterates improved euler's method
    for i in range(nt):
        xs[i + 1] = improved_euler_step(f, ts[i], xs[i], dt, pars)

    return ts, xs


def solve_temperature_ode(f, t0, t1, dt, x0, pars):
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
    """

    # initialise
    nt = int(np.ceil((t1 - t0) / dt))  # compute number of Euler steps to take
    ts = t0 + np.arange(nt + 1) * dt  # x array
    xs = 0.0 * ts  # array to store solution
    xs[0] = x0  # set initial value

    prod = interpolate_production_values(ts)

    # loop that iterates improved euler'smethod
    for i in range(nt):
        pars[0] = prod[i]
        xs[i + 1] = improved_euler_step(f, ts[i], xs[i], dt, pars)

    return ts, xs


def improved_euler_step(f, tk, yk, h, pars):
    """Compute a single Improved Euler step.

    Parameters
    ----------
    f : callable
            Derivative function.
    tk : float
            Independent variable at beginning of step.
    yk : float
            Solution at beginning of step.
    h : float
            Step size.
    pars : iterable
            Optional parameters to pass to derivative function.

    Returns
    -------
    yk1 : float
            Solution at end of the Euler step.
    """

    f0 = f(tk, yk, *pars)  # finding predictor
    f1 = f((h + tk), (yk + f0 * h), *pars)  # finding corrector
    yk2 = yk + h * (f0 / 2 + f1 / 2)  # finding solution

    return yk2


def fitting_pressure_model(f, x0, y0):
    a = 1
    return None


def fitting_temperature_model():
    a = 1
    return None


def interpolate_pressure_values(pv, tv, t):
    """Return heat source parameter p for geothermal field.

    Parameters:
    -----------
    pv : array-like
            vector of pressure values
    tv : array-like
            vector of time values
    t : array-like
            Vector of times at which to interpolate the pressure.

    Returns:
    --------
    p : array-like
            Pressure values interpolated at t.
    """
    p = np.interp(t, tv, pv)
    return p


def interpolate_production_values(t, prod1=Prodq1, t1=Yearq1, prod2=Prodq2, t2=Yearq2):
    """Return heat source parameter p for geothermal field.

    Parameters:
    -----------
    prod1 : array-like
            vector of pressure values, for bore hole 2
    prod2 : array-like
            vector of production values, for bore hole 2
    t1 : array-like
            vector of time values, for bore hole 1
    t2 : array-like
            vector of time values, for bore hole 2
    t : array-like
            Vector of times at which to interpolate the pressure.

    Returns:
    --------
    prod : array-like
            Production values interpolated at t.
    """

    p1 = np.interp(t, t1, prod1)
    p2 = np.interp(t, t2, prod2)
    prod = p1 + p2
    return prod


# WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT = load_ODE_Model_data()
t = np.linspace(1950, 2014, 262)


def plot_model():
    """Plot the LPM over top of the data.

    Parameters:
    -----------
    none

    Returns:
    --------
    none

    Notes:
    ------
    This function called within if __name__ == "__main__":


    """
    f, ax1 = plt.subplots(nrows=1, ncols=1)
    t, x = solve_pressure_ode(
        ode_pressure_model,
        1984.75,
        2010,
        0.25,
        -0.20629999999999882,
        [0.00052, 0.00065, 22],
    )
    t1 = 0
    x1 = 0
    ax1.plot(t, x, "b-", label="model")
    ax1.plot(t1, x1, "r-", label="model1")

    plt.show()
