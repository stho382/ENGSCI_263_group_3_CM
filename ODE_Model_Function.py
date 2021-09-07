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


### MAKE SURE TO MOVE TO THE MAIN PY FUNCTION RATHER THAN KEEPING HERE
# These take the data we have and set to global variables

# Do we delete this, not sure if it is the only place where we load in the data?
WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT = load_ODE_Model_data()
#Pressure should be in Pa rather than MPa
Pressure = (1+((WaterLevel - 296.85)))*1000000
# Production should be per year rather than per day
Prodq2 = Prodq2*365
Prodq1 = Prodq1*365
P0 = 1.6e+06
TCguess = 20

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
    #think this is valid?
    Tt = T

    if P > P0:
        dTdt = -at * (bp / ap) * (P - P0) * (Tt - T) - bt * (T - T0)
    else:
        dTdt = -at * (bp / ap) * (P - P0) * (Tc - T) - bt * (T - T0)
    return dTdt



def solve_pressure_ode(f, t0, t1, dt, x0, pars, future_prediction='False', benchmark=False,):
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
        if future_prediction == 'False':
            # get interpolated production values
            prod = interpolate_production_values(ts)
            # calculate dqdt at each point
            dqdt = find_dqdt(prod, dt)

        # if doing future predictions, set production. 
		# We know from previous data (with a 10000 tonne/day limit) that even with a constant overall production there
		# is some dqdt over time, and this has a significant impact on the equation for pressure
		# therefore we create a dqdt relative to the size of the q
        if future_prediction != 'False':
            # production constant of future predicted valyes
            prod = [future_prediction] * len(ts)
            #dqdt = [0] * len(ts)
            
			# dqdt seems to be about 1/300th of q if constant q
            dqdt = [future_prediction/300] * len(ts)

			# as overall q is not changing, dqdt should be negative as often as positive
			# generate random binary array
            arrays = np.random.randint(2, size=len(ts))
            # look through array and make dqdt negative if 0, (therefore positive if 1)
            for i in arrays:
                if i == 0:
                    dqdt[i] = -dqdt[i]
            # assuming change spread over gradual_change/4 years
            gradual_change = 80
            if future_prediction != 3650000:
                for i in range(gradual_change):
                    dqdt[i] = dqdt[i] + (future_prediction-3650000)/gradual_change

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

    dqdt = 0.0 * q

    

    for i in range(len(q)):
        if i == 0:
            dqdt[i] = (q[i + 1] - q[i]) / h
        if i == (len(q) - 1):
            dqdt[i] = (q[i] - q[i - 1]) / h
        if (i > 0) and (i < (len(q) - 1)):
            dqdt[i] = (q[i + 1] - q[i - 1]) / (2 * h)

    return dqdt


def fit_pressure_model(t, ap, bp, cp):

    (t,p) = solve_pressure_ode(
        ode_pressure_model,
        t[1]-0.25,
        t[-1],
        0.25,
        Pressure[0],
        pars=[0, 0, P0, ap, bp, cp],
    )

    return p

def fit_temperature_model(tT, T0, at, bt):
    #[Tc, T0, at, bt, ap, bp, P, P0] need to be parsed into ode_temperature model

    # given P0
    P0 = 1.6e+06
    
    # ap and bp from fit of pressure model
    ap = 0.0015
    bp = 0.035

    #get pressure values by solving pressure ODE
    (t, p) = solve_pressure_ode(
        ode_pressure_model,
        1960,
        tT[-1],
        0.25,
        1349003,
        pars=[0, 0, P0, 0.0015, 0.035, 0.6])
    
    #create pressure array for next step
    pressure = [0] * int(np.ceil(len(t)/4))

    #we need 1/4 as many pressure value as we generated because the step size for temp is 1,
    #compared to 0.25 for pressure
    #Loop through and take every 1 of 4 values
    for i in range( int(np.ceil(len(t)/4) )):
        time = t[i*4]
        pressure[i] = p[i*4]
    
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

    #p1 = np.interp(t, t1, prod1)
    # We decided as a group to use total q which is q2
    p2 = np.interp(t, t2, prod2)
    #prod = p1 + p2
    return p2

def plot_model(Future_Productions, Future_Time, Labels):
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
    
    #PLOTTING PRESSURE FIRST

    #creates pressure array
    tP = np.arange(Yearp[0], (Yearp[-1] + 0.25), 0.25)
    #interp pressure values at time array points
    press = np.interp(tP, Yearp, Pressure)
    # create plot
    figP, axP = plt.subplots(1, 1)
    # not really sure what sigma value to use
    sigma = [0.2] * len(press)

    # replacing 10 of the values in press with P0 at t0 to make the curvefit also fit that point
    for i in range(10):
        tP[i*4] = 1950
        press[i*4] = 1.6e+06
    #fit curve
    p, cov = curve_fit(fit_pressure_model, tP, press, sigma=sigma, p0 = [0.0015, 0.035, 0.6])

    #curvefit doesn't give good values so we've generated our own using manual calibration
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
        1.6e+06,
        pars=[0, 0, P0, p[0], p[1], p[2]])

    #plot in red
    axP.plot(tP0, xP0, "r-")
    # Plot know pressures as well
    axP.plot(Yearp, Pressure, "ko")

    # Preallocate arrays 
    tPset = [0] * len(Future_Productions)
    xPset = [0] * len(Future_Productions)
    HandlesP = [0] * len(Future_Productions)

    # colours array
    colours = ["r", "g", "b", "y", "m", "c"]

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
        future_prediction = Future_Productions[i]*365
        )
        (HandlesP[i], ) = axP.plot(tPset[i], xPset[i], colours[i%6] + "-")
    
    axP.legend(handles = HandlesP, labels = Labels)
    plt.title(label = 'Pressure')
    plt.show()


    #NOW PLOTTING TEMPERATURE
    tT = np.arange(YearT[0], (YearT[-1]+1), 1)
    temperature = np.interp(tT, YearT, Temp)
    sigmaT = [0.2] * len(temperature)
    pT, covT = curve_fit(fit_temperature_model, tT, temperature, sigma=sigmaT, p0 = [200, 5e-10, 0.025])
    figT, axT = plt.subplots(1, 1)

    tT0, xT0 = solve_temperature_ode(
        ode_temperature_model,
        tT[0],
        tT[-1],
        1,
        Temp[0],
        pars=[TCguess, pT[0], pT[1], pT[2], ap, bp, np.interp(np.arange(start=tT[0],stop=tT[-1]),tP0,xP0), P0]
    )
    axT.plot(tT0, xT0, "r-", label="test")
    axT.plot(YearT, Temp, "ko")

    tTset = [0] * len(Future_Productions)
    xTset = [0] * len(Future_Productions)
    HandlesT = [0] * len(Future_Productions)

    for i in range(len(Future_Productions)):
        tTset[i], xTset[i] = solve_temperature_ode(
        ode_temperature_model,
        tT[-1],
        Future_Time,
        1,
        xT0[-1],
        pars=[TCguess, pT[0], pT[1], pT[2], ap, bp, np.interp(np.arange(start=tT[-1],stop=Future_Time),tPset[i], xPset[i]), P0]
        )
        (HandlesT[i], ) = axT.plot(tTset[i], xTset[i], colours[i%6] + "-")
    
    axT.legend(handles = HandlesT, labels = Labels)
    plt.title(label = 'Temperature')
    plt.show()

if __name__ == "__main__":
    Future_Productions = [10000, 0, 20000, 5000]
    Future_Time = 2080
    Labels=["Current Production", "Cease all production", "Double current production", "Half Current Production"]
    plot_model(Future_Productions, Future_Time, Labels)

    '''t = np.arange(Yearq2[0], (Yearq2[-1] + 0.25), 0.25)
    t1 = np.arange(Yearp[0], (Yearp[-1] + 0.25), 0.25)
    press = np.interp(t1, Yearp, Pressure)
    fig, ax = plt.subplots(1, 1)
    sigma = [0.2] * len(press)
    #p, cov = curve_fit(fit_pressure_model, t1, press, sigma=sigma, p0 = [1, 4.9e-07, 1, 1])


    TimeT = np.arange(YearT[0], (YearT[-1] + 0.25), 0.25)
    t0, x0 = solve_pressure_ode(
        ode_pressure_model,
        1960,
        t[-1],
        0.25,
        -1.2,
        pars=[0, 0, p[0], p[1], p[2], p[3]])

    ax.plot(t0, x0, "k-", label="test")

    t1, x1 = solve_pressure_ode(
        ode_pressure_model,
        t1[0],
        t1[-1],
        0.25,
        Pressure[0],
        pars=[0, 0, p[0], p[1], p[2], p[3]],
    )
    t2, x2 = solve_pressure_ode(
        ode_pressure_model,
        t0=t[-1],
        t1=2050,
        dt=0.25,
        x0=x1[-1],
        pars=[0, 0, p[0], p[1], p[2], p[3]],
        future_prediction=10000,
    )
    t3, x3 = solve_pressure_ode(
        ode_pressure_model,
        t0=t[-1],
        t1=2050,
        dt=0.25,
        x0=x1[-1],
        pars=[0, 0, p[0], p[1], p[2], p[3]],
        future_prediction=0,
    )
    t4, x4 = solve_pressure_ode(
        ode_pressure_model,
        t0=t[-1],
        t1=2050,
        dt=0.25,
        x0=x1[-1],
        pars=[0, 0, p[0], p[1], p[2], p[3]],
        future_prediction=20000,
    )
    ax.plot(t1, x1, "r-", label="test")
    (line1,) = ax.plot(t2, x2, "r-", label="test1")
    (line2,) = ax.plot(t3, x3, "b-", label="test2")
    (line3,) = ax.plot(t4, x4, "g-", label="test3")
    ax.legend(
        handles=[line1, line2, line3],
        labels=[
            "Current Production",
            "Cease all production",
            "Double current production",
        ],
    )

    ps = np.random.multivariate_normal(p, cov, 100)
    for pi in ps:
        tp1, xp1 = solve_pressure_ode(
            ode_pressure_model,
            t[0],
            t[-1],
            0.25,
            -0.20629999999999882,
            pars=[0, 0, pi[0], pi[1], pi[2], pi[3]],
        )
        tp2, xp2 = solve_pressure_ode(
            ode_pressure_model,
            t0=tp1[-1],
            t1=2050,
            dt=0.25,
            x0=xp1[-1],
            pars=[0, 0, pi[0], pi[1], pi[2], pi[3]],
            future_prediction=10000,
        )
        tp3, xp3 = solve_pressure_ode(
            ode_pressure_model,
            t0=tp1[-1],
            t1=2050,
            dt=0.25,
            x0=xp1[-1],
            pars=[0, 0, pi[0], pi[1], pi[2], pi[3]],
            future_prediction=0,
        )
        tp4, xp4 = solve_pressure_ode(
            ode_pressure_model,
            t0=tp1[-1],
            t1=2050,
            dt=0.25,
            x0=xp1[-1],
            pars=[0, 0, pi[0], pi[1], pi[2], pi[3]],
            future_prediction=20000,
        )
        ax.plot(tp1, xp1, "k-", alpha=0.2, lw=0.5)
        ax.plot(tp2, xp2, "r-", alpha=0.2, lw=0.5)
        ax.plot(tp3, xp3, "b-", alpha=0.2, lw=0.5)
        ax.plot(tp4, xp4, "g-", alpha=0.2, lw=0.5)

    ax.plot(Yearp, Pressure, "ko")

    plt.show()

    tT = np.arange(YearT[0], (YearT[-1]+1), 1)
    temperature = np.interp(tT, YearT, Temp)
    sigmaT = [0.2] * len(temperature)
    pT, covT = curve_fit(fit_temperature_model, tT, temperature, sigma=sigmaT)
    figT, axT = plt.subplots(1, 1)

    tT1, xT1 = solve_temperature_ode(
        ode_temperature_model,
        tT[0],
        tT[-1],
        1,
        Temp[0],
        pars=[20, pT[0], pT[1], pT[2], 3.120653006350713e-06, 0.11456452594196342e-05, 0, 0.29016966989598225]
    )
    axT.plot(tT1, xT1, "r-", label="test")
    axT.plot(YearT, Temp, "ko")
    #axT.plot(tT, temperature, "r-")

    plt.show()'''


    