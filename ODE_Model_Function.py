from types import FrameType
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.npyio import load
from numpy.linalg import norm
from glob import glob
import os
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns


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
        if file == "data" + os.sep + "gr_p.txt":
            Data = np.genfromtxt(file, delimiter=",", skip_header=1)
            WaterLevel = Data[:, 1]
            Yearp = Data[:, 0]
        elif file == "data" + os.sep + "gr_q1.txt":
            Data = np.genfromtxt(file, delimiter=",", skip_header=1)
            Prodq1 = Data[:, 1]
            Yearq1 = Data[:, 0]
        elif file == "data" + os.sep + "gr_q2.txt":
            Data = np.genfromtxt(file, delimiter=",", skip_header=1)
            Prodq2 = Data[:, 1]
            Yearq2 = Data[:, 0]
        else:
            Data = np.genfromtxt(file, delimiter=",", skip_header=1)
            Temp = Data[:, 1]
            YearT = Data[:, 0]

    return WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT


# Global Variables established
WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT = load_ODE_Model_data()
# Pressure should be in Pa rather than MPa
Pressure = (1 + ((WaterLevel - 296.85))) * 1000000
# Production should be per year rather than per day
Prodq2 = Prodq2 * 365
Prodq1 = Prodq1 * 365
# Given Parameters
P0 = 1.6e06
TCguess = 30


def ode_pressure_model(t, P, q, dqdt, P0, ap, bp, cp):
    """
    Return the derivative dP/dt at time, t, for given parameters.

    Parameters:
    -----------
    t : float
            independent variable
    P : float
            dependent variable.
    q : float
            Source/sink rate.
    dqdt :
            rate of change of source/ sink rate
    P0 : float
            ambient value of dependent variable.
    ap : float
            Source/sink strength parameter.
    bp : float
            Recharge strength parameter.
    cp : float
            Slow strength parameter.


    Returns:
    --------
    dPdt : float
            Derivative of dependent variable with respect to independent variable.

    Notes:
    ------
    None
    """
    # Formula
    dPdt = -ap * q - bp * (P - P0) - cp * dqdt

    return dPdt


def ode_temperature_model(t, T, Tc, T0, at, bt, ap, bp, P, P0):
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

    Tt = T

    # Formula
    if P > P0:
        dTdt = -at * (bp / ap) * (P - P0) * (Tt - T) - bt * (T - T0)
    else:
        dTdt = -at * (bp / ap) * (P - P0) * (Tc - T) - bt * (T - T0)
    return dTdt


def solve_pressure_ode(
    f,
    t0,
    t1,
    dt,
    x0,
    pars,
    future_prediction="False",
    benchmark=False,):

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
            List of parameters passed to ODE function f. [q, dqdt, P0, ap, bp, cp]
    future_prediction : False or value
            False if not being used for a future prediction, otherwise contains value of future predicted production rate
        benchmark : boolean
                        Tells if this is being used for a benchmark test


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

    if not benchmark:
        # if not doing future predictions
        if future_prediction == "False":
            # get interpolated production values
            prod = interpolate_production_values(ts)
            # calculate dqdt at each point
            dqdt = find_dqdt(prod, dt)

        # if doing future predictions, set production.
        # We know from previous data (with a 10000 tonne/day limit) that even with a constant overall production there
        # is some dqdt over time, and this has a significant impact on the equation for pressure
        # therefore we create a dqdt relative to the size of the q
        if future_prediction != "False":
            # production constant of future predicted valyes
            prod = [future_prediction] * len(ts)
            # dqdt = [0] * len(ts)

            # dqdt seems to be about 1/300th of q if constant q
            dqdt = [future_prediction / 300] * len(ts)

            # as overall q is not changing, dqdt should be negative as often as positive
            # generate random binary array
            arrays = np.random.randint(2, size=len(ts))
            # look through array and make dqdt negative if 0, (therefore positive if 1)
            for i in arrays:
                if i == 0:
                    dqdt[i] = -dqdt[i]
            # assuming change spread over gradual_change/4 years
            gradual_change = 40
            if future_prediction > 3650000:
                for i in range(gradual_change):
                    dqdt[i] = dqdt[i] + (future_prediction - 3650000) / gradual_change
            # if a limit is introduced, assume production will quickly be brought down over time
            # to be under limit - past evidence indicates this will take 4 years (ie 12 * 0.25 steps)
            if future_prediction < 3650000:
                for i in range(12):
                    dqdt[i] = dqdt[i] + (future_prediction - 3650000) / 12

    # loop that iterates improved euler'smethod
    for i in range(nt):
        if not benchmark:
            pars[0] = prod[i]
            pars[1] = dqdt[i]
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

    # Pressure is not constant over time so we need to pass in a list of values for it
    # The ode temperature model only accepts single inputs for pressure
    P = pars[-2]
    parsi = pars.copy()
    # make sure parsi has no elements that are array  (will be reset but sometimes causes issues for some reasons)
    parsi[-2] = P[0]

    # loop that iterates improved euler'smethod
    for i in range(nt):
        # this line pulls the correct value for pressure.
        parsi[-2] = P[i]
        xs[i + 1] = improved_euler_step(f, ts[i], xs[i], dt, parsi)

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


def find_dqdt(q, h):
    """Takes an array of q (production) over time and returns the dq/dt array

    Parameters:
    -----------
    q : array-like
            Array of q values
    h : int
            time-step size between each q value

    Returns:
    --------
    dqdt : array-like
            array of dqdt values
    """

    # make a 0 array length q
    dqdt = 0.0 * q

    # Loop through each position in dqdt and calculate dqdt
    for i in range(len(q)):
        # If first position use forward step
        if i == 0:
            dqdt[i] = (q[i + 1] - q[i]) / h
        # If last position use backward step 
        if i == (len(q) - 1):
            dqdt[i] = (q[i] - q[i - 1]) / h
        # if neither first nor last position, use double step
        if (i > 0) and (i < (len(q) - 1)):
            dqdt[i] = (q[i + 1] - q[i - 1]) / (2 * h)

    return dqdt


def fit_pressure_model(t, ap, bp, cp):
    """
    Takes the inputs ap, bp and cp along the time array t and uses them to solve the pressure ode
    This allows us to use the curvefit function to change only parameters ap, bp and cp

    Parameters:
    -----------
    t : array-like
            an array of time values
    ap : float
            a lumped paramter for pressure model
    bp : float
            a lumped paramter for pressure model
    cp : float
            a lumped paramter for pressure model

    Returns:
    --------
    p : array-like
            an array of pressure values at times t for the solved pressure ODE
    """

    # Takes inputs and passes them into solve_pressure_ode to solve
    (t, p) = solve_pressure_ode(
        ode_pressure_model,
        t[1] - 0.25,
        t[-1],
        0.25,
        Pressure[0],
        pars=[0, 0, P0, ap, bp, cp],
    )

    return p


def fit_pressure_model_P0(t, ap, bp, cp, P00):
    """
    Takes the inputs ap, bp, cp and P00 along the time array t and uses them to solve the pressure ode
    This allows us to use the curvefit function to change only parameters ap, bp and cp
    This one has P0 as a parameter whereas the earlier function does not

    Parameters:
    -----------
    t : array-like
            an array of time values
    ap : float
            a lumped paramter for pressure model
    bp : float
            a lumped paramter for pressure model
    cp : float
            a lumped paramter for pressure model
    P00 : float
            a number representing initial pressure

    Returns:
    --------
    p : array-like
            an array of pressure values at times t for the solved pressure ODE
    """
    # Like fit pressure model but also takes P0 as a variable rather than as a given

    # Takes inputs and passes them into solve_pressure_ode to solve
    (t, p) = solve_pressure_ode(
        ode_pressure_model,
        t[1] - 0.25,
        t[-1],
        0.25,
        Pressure[0],
        pars=[0, 0, P00, ap, bp, cp],
    )

    return p


def fit_temperature_model(tT, T0, at, bt):
    """
    Takes the inputs T0, at and bt along the time array tT and uses them to solve the temperature ode
    This allows us to use the curvefit function to change only parameters T0, at and bt

    Parameters:
    -----------
    tT : array-like
            an array of time values
    T0 : float
            initial temperature
    at : float
            a lumped paramter for temperature model
    bt : float
            a lumped paramter for temperature model
    

    Returns:
    --------
    pT : array-like
            an array of Temperature values at times tT for the solved temperature ODE
    """
    # [Tc, T0, at, bt, ap, bp, P, P0] need to be parsed into ode_temperature model

    # given P0
    P0 = 1.6e06

    # ap and bp from fit of pressure model
    ap = 0.0015
    bp = 0.035

    # get pressure values by solving pressure ODE
    (t, p) = solve_pressure_ode(
        ode_pressure_model,
        1960,
        tT[-1],
        0.25,
        1349003,
        pars=[0, 0, P0, 0.0015, 0.035, 0.6],
    )

    # create pressure array for next step
    pressure = [0] * int(np.ceil(len(t) / 4))

    # we need 1/4 as many pressure value as we generated because the step size for temp is 1,
    # compared to 0.25 for pressure
    # Loop through and take every 1 of 4 values
    for i in range(int(np.ceil(len(t) / 4))):
        time = t[i * 4]
        pressure[i] = p[i * 4]

    # Solve temperature ode
    (tT, pT) = solve_temperature_ode(
        ode_temperature_model,
        1960,
        tT[-1],
        1,
        Temp[0],
        pars=[20, T0, at, bt, ap, bp, pressure, P0],
    )

    return pT


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

    # p1 = np.interp(t, t1, prod1)
    # We decided as a group to use total q which is q2
    p2 = np.interp(t, t2, prod2)
    # prod = p1 + p2
    return p2


def plot_initial_attempt(back_date=False):
    """Plot's our first attempt at a pressure model

    Parameters:
    -----------
    back_date : boolean
            If backdate is true, we extrapolate pressure data to the past
    """

    # creates pressure array
    tP = np.arange(Yearp[0], (Yearp[-1] + 0.25), 0.25)
    # interp pressure values at time array points
    press = np.interp(tP, Yearp, Pressure)
    # create plot
    figP, axP = plt.subplots(1, 1)
    # sigma values created using ad hoc calibration
    sigma = [0.2] * len(press)

    # use curve_fit to fit data points
    p, cov = curve_fit(
        fit_pressure_model_P0, tP, press, sigma=sigma, p0=[0.0015, 0.035, 0.6, 1.6e06]
    )

    # Generate data points with calculated parameters
    tP0, xP0 = solve_pressure_ode(
        ode_pressure_model,
        tP[0],
        tP[-1],
        0.25,
        press[0],
        pars=[0, 0, p[3], p[0], p[1], p[2]],
    )

    # plot in red
    axP.plot(tP0, xP0, "r-")

    cov = cov / 5

    # if we want to also generate value for past data
    if back_date == True:
        # call pressure ode solve starting at earliest data point and stepping 
        # BACKWARDS with step-size 0.25 until 1950
        yall, xall = solve_pressure_ode(
            ode_pressure_model,
            tP[0],
            1950,
            -0.25,
            press[0],
            pars=[0, 0, p[3], p[0], p[1], p[2]],
        )
        axP.plot(yall, xall, "k-")

    # Plot know pressures as well
    axP.plot(Yearp, Pressure, "ko")
    axP.set_title("Initial Pressure Model")
    axP.set_xlabel("Year")
    axP.set_ylabel("Pressure (Pa)")

    # toggle to either save or show graph
    show_not_save = True
    if show_not_save == True:
        plt.show()
    else:
        figP.savefig("Initial_Pressure_Model.png")
        plt.close(figP)


def plot_second_attempt(back_date=False):
    """Plots our second attempt at a pressure model

    Parameters:
    -----------
    back_date : boolean
            If backdate is true, we extrapolate pressure data to the past
    """

    # creates pressure array
    tP = np.arange(Yearp[0], (Yearp[-1] + 0.25), 0.25)
    # interp pressure values at time array points
    press = np.interp(tP, Yearp, Pressure)
    # create plot
    figP, axP = plt.subplots(1, 1)
    # sigma values created using ad hoc calibration
    sigma = [0.2] * len(press)

    # use curve_fit to fit data points
    p, cov = curve_fit(
        fit_pressure_model, tP, press, sigma=sigma, p0=[0.0015, 0.035, 0.6]
    )

    # Generate model for past values
    tP0, xP0 = solve_pressure_ode(
        ode_pressure_model,
        tP[0],
        tP[-1],
        0.25,
        press[0],
        pars=[0, 0, P0, p[0], p[1], p[2]],
    )

    # plot in red
    axP.plot(tP0, xP0, "r-")

    cov = cov / 5

    # if we want to also generate value for past data
    if back_date == True:
        # call pressure ode solve starting at earliest data point and stepping 
        # BACKWARDS with step-size 0.25 until 1950
        yall, xall = solve_pressure_ode(
            ode_pressure_model,
            tP[0],
            1950,
            -0.25,
            press[0],
            pars=[0, 0, P0, p[0], p[1], p[2]],
        )
        axP.plot(yall, xall, "k-")

    # Plot know pressures as well
    axP.plot(Yearp, Pressure, "ko")
    axP.set_title("Second Pressure Model")
    axP.set_xlabel("Year")
    axP.set_ylabel("Pressure (Pa)")

    # toggle to either save or show graph
    show_not_save = True
    if show_not_save == True:
        plt.show()
    else:
        figP.savefig("Second_Pressure_Model.png")
        plt.close(figP)


def plot_final_model():
    """Plots our second attempt at a pressure model.
    Is called in main"""

    # creates pressure array
    tP = np.arange(Yearp[0], (Yearp[-1] + 0.25), 0.25)
    # create plot
    figP, axP = plt.subplots(1, 1)

    # curvefit doesn't give good values so we've generated our own using manual calibration
    ap = 0.0015
    bp = 0.035
    cp = 0.6
    p = [ap, bp, cp]

    # Solves ode and generates array of time and pressure values for given parameters
    tP0, xP0 = solve_pressure_ode(
        ode_pressure_model,
        1950,
        tP[-1],
        0.25,
        1.6e06,
        pars=[0, 0, P0, p[0], p[1], p[2]],
    )

    # plot in red
    axP.plot(tP0, xP0, "r-")
    axP.set_title("Final Pressure Model")
    axP.set_xlabel("Year")
    axP.set_ylabel("Pressure (Pa)")
    # Plot know pressures as well
    axP.plot(Yearp, Pressure, "ko")

    # NOW PLOTTING TEMPERATURE
    # create time array with step size 1
    tT = np.arange(YearT[0], (YearT[-1] + 1), 1)
    # interpolate temperature values at times tT
    temperature = np.interp(tT, YearT, Temp)
    # generate array of sigma values with length of temperature
    sigmaT = [0.3] * len(temperature)
    # use curvefit to fit parameters to data
    pT, covT = curve_fit(
        fit_temperature_model, tT, temperature, sigma=sigmaT, p0=[200, 5e-10, 0.025]
    )
    # create plot
    figT, axT = plt.subplots(1, 1)

    # Solves ode and generates array of time and temperature values for given parameters
    tT0, xT0 = solve_temperature_ode(
        ode_temperature_model,
        tT[0],
        tT[-1],
        1,
        Temp[0],
        pars=[
            TCguess,
            pT[0],
            pT[1],
            pT[2],
            ap,
            bp,
            np.interp(np.arange(start=tT[0], stop=tT[-1]), tP0, xP0),
            P0,
        ],
    )
    # plot these values
    axT.plot(tT0, xT0, "r-", label="test")
    axT.set_title("Final Temperature Model")
    axT.set_xlabel("Year")
    axT.set_ylabel("Temperature (Celsius)")
    axT.plot(YearT, Temp, "ko")

    # toggle to either save or show graphs
    show_not_save = True
    if show_not_save == True:
        plt.show()
    else:
        figP.savefig("Final_Pressure_Model.png")
        plt.close(figP)
        figT.savefig("Final_Temperature_Model.png")
        plt.close(figT)


def plot_model(Future_Productions, Future_Time, Labels, uncertainty=True):
    """Plot the model

    Parameters:
    -----------
    Future_Productions : array-like
        Array of future productions to be tested
    Future_Time : int
        Future year that you want to predict to
    Labels : array-like
        array of len(Future_Productions) that contains the labels of each line to be plotted
        eg ["Current Production", "Cease all production", "Double current production"]

    Returns:
    --------
    none

    Notes:
    ------
    This function called within if __name__ == "__main__":

    """

    # PLOTTING PRESSURE FIRST

    # creates pressure array
    tP = np.arange(Yearp[0], (Yearp[-1] + 0.25), 0.25)
    # interp pressure values at time array points
    press = np.interp(tP, Yearp, Pressure)
    # create plot
    figP, axP = plt.subplots(1, 1)
    # sigma values created using ad hoc calibration
    sigma = [0.15] * len(press)

    # fit curve
    p, cov = curve_fit(
        fit_pressure_model, tP, press, sigma=sigma, p0=[0.0015, 0.035, 0.6]
    )

    # curvefit doesn't give good values so we've generated our own using manual calibration
    ap = 0.0015
    bp = 0.035
    cp = 0.6
    p = [ap, bp, cp]

    # Generate model for past values
    tP0, xP0 = solve_pressure_ode(
        ode_pressure_model,
        1950,
        tP[-1],
        0.25,
        1.6e06,
        pars=[0, 0, P0, p[0], p[1], p[2]],
    )

    cov = cov / 4

    # plot in red
    axP.plot(tP0, xP0, "r-")
    # Plot know pressures as well
    axP.plot(Yearp, Pressure, "ko")

    # generate 100 samples 
    multi_var_samples = 100

    # if we want uncertainty to be plotted
    if uncertainty == True:
        # create multivariate for uncertainty with multi_var_samples # of samples
        psP = np.random.multivariate_normal(p, cov, multi_var_samples)

        # preallocate arrays
        tp0 = [0] * multi_var_samples
        xp0 = [0] * multi_var_samples

        # for each sample
        for i in range(multi_var_samples):
            pi = psP[i]
            # solve with parameters for that sample of parameters
            tp0[i], xp0[i] = solve_pressure_ode(
                ode_pressure_model,
                1950,
                tP[-1],
                0.25,
                1.6e06,
                pars=[0, 0, P0, pi[0], pi[1], pi[2]],
            )
            # plot
            axP.plot(tp0[i], xp0[i], "k-", alpha=0.2, lw=0.5)

    # Preallocate arrays
    tPset = [0] * len(Future_Productions)
    xPset = [0] * len(Future_Productions)
    HandlesP = [0] * len(Future_Productions)

    # colours array
    colours = ["r", "g", "b", "y", "m", "c"]

    tPsetuncert = tPset * 1
    xPsetuncert = xPset * 1

    # Loop through each future prediction
    for i in range(len(Future_Productions)):
        # get t and x values  for this future production
        tPset[i], xPset[i] = solve_pressure_ode(
            ode_pressure_model,
            tP[-1],
            Future_Time,
            0.25,
            xP0[-1],
            pars=[0, 0, P0, p[0], p[1], p[2]],
            future_prediction=Future_Productions[i] * 365,
        )
        # add label
        (HandlesP[i],) = axP.plot(tPset[i], xPset[i], colours[i % 6] + "-")
        # print which scenario number this is
        print('Scenario ' + str(i+1))
        # print the predicted pressure in 2080 for this scenario
        print(xPset[i][-1]/1e+06)

        # if we want undertainties plotted
        if uncertainty == True:
            # preallocate array
            tPsetuncert[i] = [0] * multi_var_samples
            xPsetuncert[i] = [0] * multi_var_samples
            uncertainties = [0]*multi_var_samples
            
            # loop trhough each sample
            for j in range(multi_var_samples):
                pi = psP[j]
                # solve with parameters for that sample
                tPsetuncert[i][j], xPsetuncert[i][j] = solve_pressure_ode(
                    ode_pressure_model,
                    tP[-1],
                    Future_Time,
                    0.25,
                    xp0[j][-1],
                    pars=[0, 0, P0, pi[0], pi[1], pi[2]],
                    future_prediction=Future_Productions[i] * 365,
                )
                # plot 
                axP.plot(
                    tPsetuncert[i][j],
                    xPsetuncert[i][j],
                    colours[i % 6] + "-",
                    alpha=0.2,
                    lw=0.5,
                )
                uncertainties[j] = xPsetuncert[i][j][-1]
            # print the 5th and 95th percentiles for this scenario
            print('Percentiles')
            print(np.percentile(uncertainties, 5)/1e+06)
            print(np.percentile(uncertainties, 95)/1e+06)

    # names of stakeholders
    stakeholder_labels = [
        "ROTORUA CITY COUNCIL",
        "T??HOURANGI NG??TI W??HIAO",
        "LOCAL CHAMBER OF COMMERCE",
    ]

    # loop through each stakeholder and print name on graph in appropriate position
    # aligning with the scenario that relates to their desired outcome
    for stakeholder_P in range(len(stakeholder_labels)):
        axP.text(
            tPset[stakeholder_P][-1],
            xPset[stakeholder_P][-1] - 105000,
            stakeholder_labels[stakeholder_P],
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=7,
            fontweight="bold",
        )

    # label plots
    axP.legend(handles=HandlesP, labels=Labels)
    if uncertainty == True:
        axP.set_title("Pressure Model - Predictions (with uncertainty)")
    else:
        axP.set_title("Pressure Model - Predictions")
    axP.set_xlabel("Year")
    axP.set_ylabel("Pressure (Pa)")



    # NOW PLOTTING TEMPERATURE
    # create time array with step size 1
    tT = np.arange(YearT[0], (YearT[-1] + 1), 1)
    # interpolate temperature values at times tT
    temperature = np.interp(tT, YearT, Temp)
    # generate array of sigma values with length of temperature
    sigmaT = [0.35] * len(temperature)
    # use curvefit to fit parameters to data
    pT, covT = curve_fit(
        fit_temperature_model, tT, temperature, sigma=sigmaT, p0=[200, 5e-10, 0.025]
    )
    # create plot
    figT, axT = plt.subplots(1, 1)

    # solve ode for given paramters
    tT0, xT0 = solve_temperature_ode(
        ode_temperature_model,
        tT[0],
        tT[-1],
        1,
        Temp[0],
        pars=[
            TCguess,
            pT[0],
            pT[1],
            pT[2],
            ap,
            bp,
            np.interp(np.arange(start=tT[0], stop=tT[-1]), tP0, xP0),
            P0,
        ],
    )

    # plot solution
    axT.plot(tT0, xT0, "r-", label="test")
    axT.plot(YearT, Temp, "ko")

    # if we want uncertainties
    if uncertainty == True:
        # generate sample of multivariates
        psT = np.random.multivariate_normal(pT, covT * 4, multi_var_samples)

        tt0 = [0] * multi_var_samples
        xt0 = [0] * multi_var_samples

        # for each sample, solve ode with those parameters and plot it 
        for i in range(multi_var_samples):
            Ti = psT[i]
            tt0[i], xt0[i] = solve_temperature_ode(
                ode_temperature_model,
                tT[0],
                tT[-1],
                1,
                Temp[0],
                pars=[
                    TCguess,
                    Ti[0],
                    Ti[1],
                    Ti[2],
                    ap,
                    bp,
                    np.interp(np.arange(start=tT[0], stop=tT[-1]), tP0, xP0),
                    P0,
                ],
            )
            axT.plot(tt0[i], xt0[i], "k-", alpha=0.2, lw=0.5)

    # preallocate arrays
    tTset = [0] * len(Future_Productions)
    xTset = [0] * len(Future_Productions)
    HandlesT = [0] * len(Future_Productions)

    # generate sample of multivariates
    psT = np.random.multivariate_normal(pT, covT, multi_var_samples)

    # loop through each potential future scenerio
    for i in range(len(Future_Productions)):
        # solve with the original parameters for this future scenario
        tTset[i], xTset[i] = solve_temperature_ode(
            ode_temperature_model,
            tT[-1],
            Future_Time,
            1,
            xT0[-1],
            pars=[
                TCguess,
                pT[0],
                pT[1],
                pT[2],
                ap,
                bp,
                np.interp(
                    np.arange(start=tT[-1], stop=Future_Time), tPset[i], xPset[i]
                ),
                P0,
            ],
        )
        # attach handle (for labeling) to plot of this scenario
        (HandlesT[i],) = axT.plot(tTset[i], xTset[i], colours[i % 6] + "-")
        # print Temperature in 2080 for this scenario
        print('Scenario ' + str(i+1))
        print(xTset[i][-1])

        # if we want uncertainties, loop through each multivariate sample
        # solve ode with parameters for that sample and plot
        if uncertainty == True:
            uncertainty_data = [0]*multi_var_samples
            for j in range(multi_var_samples):
                Ti = psT[j]
                tTset[i], xTset[i] = solve_temperature_ode(
                    ode_temperature_model,
                    tT[-1],
                    Future_Time,
                    1,
                    xt0[j][-1],
                    pars=[
                        TCguess,
                        Ti[0],
                        Ti[1],
                        Ti[2],
                        ap,
                        bp,
                        np.interp(
                            np.arange(start=tT[-1], stop=Future_Time),
                            tPset[i],
                            xPset[i],
                        ),
                        P0,
                    ],
                )
                axT.plot(tTset[i], xTset[i], colours[i % 6] + "-", alpha=0.2, lw=0.5)
                uncertainty_data[j] = xTset[i][-1]
            # print 5th and 95th percentiles for each scenario
            print('Percentiles')
            print(np.percentile(uncertainty_data, 5))
            print(np.percentile(uncertainty_data, 95))

    # loop through each stakeholder and print name on graph in appropriate position
    # aligning with the scenario that relates to their desired outcome
    for stakeholder_T in range(len(stakeholder_labels)):
        axT.text(
            tTset[stakeholder_T][-1],
            xTset[stakeholder_T][-1] - 0.1,
            stakeholder_labels[stakeholder_T],
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=7,
            fontweight="bold",
        )
    axT.hlines(y=147, xmin=1955, xmax=2085, colors="k", linestyles="--")
    axT.set_xlim(1957.5, 2082.5)
    axT.legend(handles=HandlesT, labels=Labels)
    if uncertainty == True:
        axT.set_title("Temperature Model - Predictions (with uncertainty)")
    else:
        axT.set_title("Temperature Model - Predictions")
    axT.set_xlabel("Year")
    axT.set_ylabel("Temperature (Celsius)")

    # porosity graph calculations
    np.random.seed(314)
    lpm_values_array = np.random.multivariate_normal(p, cov / 100, 10000)
    porosity_vals = np.zeros(len(lpm_values_array))
    for i in range(0, len(lpm_values_array)):
        porosity_vals[i] = porosity_equation(
            lpm_values_array[i][0],
            lpm_values_array[i][1],
            lpm_values_array[i][2],
            0.3,
            28000000,
        )

    # Working out percentile values
    percentile_95 = np.percentile(porosity_vals, 95)
    percentile_5 = np.percentile(porosity_vals, 5)
    
    # plotting graph
    f1, ax1 = plt.subplots(nrows=1, ncols=1)
    sns.distplot(
        porosity_vals,
        hist=True,
        kde=True,
        bins=40,
        color="darkblue",
        hist_kws={"edgecolor": "darkblue"},
    )
    # adding percentile lines
    ax1.vlines(x=percentile_95, ymin=0, ymax=200, colors="r", linestyles="--")
    ax1.vlines(x=percentile_5, ymin=0, ymax=200, colors="r", linestyles="--")
    
    # setting y limits
    ax1.set_ylim(0, 20)

    # Adding labels and titles
    ax1.set_xlabel("Porosity")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Porosity Posterior Distribution")
    save_figure = False

    if not save_figure:
        plt.show()
    else:
        f1.savefig("Porosity.png", dpi=300)
    return tT0, xT0, tP0, xP0


def plot_misfit(xp, fp, xt, ft):
    """
    Takes in x-axis and y-axis values for our pressure and temperature models and calculates the misfit
    between the data and the model

    Parameters:
    -----------
    xp : array-like
            array of time values for pressure model
    fp : array-like
            array of pressure values for pressure model
    xt : array-like
            array of time values for temperature model
    ft : array-like
            array of temperature values for temperature model

    """

    temperature_points = np.interp(YearT, xt, ft)
    temp_error = temperature_points - Temp

    pressure_points = np.interp(Yearp, xp, fp)
    pressure_error = pressure_points - Pressure

    f1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax2.plot(YearT, temp_error, "r-", label="error")
    ax2.hlines(y=0, xmin=1955, xmax=2020, colors="k", linestyles="--")
    ax1.plot(Yearp, pressure_error, "r-", label="error")
    ax2.set_xlim(1957.5, 2015)
    ax1.hlines(y=0, xmin=1955, xmax=2020, colors="k", linestyles="--")
    ax1.set_xlim(1982.5, 2015)
    ax1.set_title("Pressure misfit")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Error")

    ax2.set_title("Temperature misfit")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error")

    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    f1.suptitle("Misfit in model vs observations")
    f1.set_size_inches(9, 6)
    # EITHER show the plot to the screen OR save a version of it to the disk
    save_figure = False
    if not save_figure:
        plt.show()
    else:
        plt.savefig("misfit.png", dpi=300)


def porosity_equation(a, b, c, S0, A):
    """Return porosity (phi) for a geothermal field.

        Parameters:
    -----------
    a : float
            lumped parameter a
    b : float
            lumped parameter b
    c : float
            lumped parameter c
    S0 : float
            vector of time values, for bore hole 2
    A : float
            area of geothermal field

    Returns
    -------
    phi : float
            porosity of geothermal field.
    """
    g = 9.81                            # acceleration due to gravity
    a_adjusted = a / 1000               # converting a to SI units
    b_adjusted = b / (3.154 * 10 ** 7)  # converting b to SI units
    c_adjusted = c * (3.154 * 10 ** 2)  # converting c to SI units

    # Equation to find porosity
    phi = (g * (a_adjusted - (b_adjusted * c_adjusted))) / (
        (1 - S0) * A * a_adjusted ** 2
    )
    return phi
