# ENGSCI263: Lab Exercise 2
# lab2.py

# PURPOSE:
# IMPLEMENT a lumped parameter model and CALIBRATE it to data.

# PREPARATION:
# Review the lumped parameter model notes and use provided data from the kettle experiment.

# SUBMISSION:
# - Show your calibrated LPM to the instructor (in Week 3).

# imports
from ast import fix_missing_locations
import numpy as np
from matplotlib import pyplot as plt
import math as mt

def ode_model(t, x, q, a, b, x0):
    ''' Return the derivative dx/dt at time, t, for given parameters.

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

    '''
    dxdt = (a*q) - (b*(x - x0))
    
    return dxdt

def improved_euler_step(f, tk, yk, h, pars):
	""" Compute a single Improved Euler step.
	
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
	# calculate f0 (= f(tk, yk) incl. input parameters)
	f0 = f(tk, yk, *pars)
	# calculate f0 (= f(tk + h, yk + h*f(tk, yk)) incl. input parameters)
	f1 = f((tk + h), yk + h*f0, *pars)

	# calculate yk after 1 step using average gradients of f0 and f1
	yk1 = yk + h*(f0 + f1)/2

	# return yk after 1 step
	return yk1


def solve_ode(f, t0, t1, dt, x0, pars):
    ''' Solve an ODE numerically.

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
    '''

    # initialise
    nt = int(np.ceil((t1-t0)/dt))		# compute number of Euler steps to take
    t = t0+np.arange(nt+1)*dt			# x array
    x = 0.*t							# array to store solution
    x[0] = x0							# set initial value
	
	# your code here
	
	# loop through each euler step
    for k in range(nt):
		# y value at the next step is found using the IE method with improved_euler_step()
	    x[k+1] = improved_euler_step(f, t[k], x[k], dt, pars)

	# return arrays of t (independent variable) and x (dependent variable)
    return t, x

def dxdt_trial_1(x,t):
    return(mt.sin((4*t)-3))


def x_trial_1(t):
    return(-0.25*(mt.cos((4*t)-3)))

def plot_benchmark(f, t0, t1, dt, x0, pars):
    ''' Compare analytical and numerical solutions.

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
        
    '''
    # create plot
    fx,ax1 = plt.subplots(nrows=1,ncols=1)   
    
    t, x = solve_ode(f, t0, t1, dt, x0, pars)
    ax1.plot(t, x, 'b+', label='ODE Solver')
    
    xs=[0]*len(t)
    for pos in range(len(t)):
        xs[pos] = -(1 - mt.exp(-t[pos]))

    ax1.plot(t, xs, 'r-', label='Analytic')
    #plt.show()

    fy,ax2 = plt.subplots(nrows=1,ncols=1) 
    x_error = (x[1:] - xs[1:])/xs[1:]
    x_error_exp = np.log(x_error)
    #ax2.plot(t, x_error, 'ko', label='Jesus')
    ax2.plot(t[1:], x_error_exp, 'r-', label='Jesus')
    plt.show()


def load_kettle_temperatures():
    ''' Returns time and temperature measurements from kettle experiment.

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
    '''
    pass

def interpolate_kettle_heatsource(t):
    ''' Return heat source parameter q for kettle experiment.

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
    '''
    # suggested approach
    # hard code vectors tv and qv which define a piecewise heat source for your kettle 
    # experiment
    # use a built in Python interpolation function 

    data = np.genfromtxt("263_Kettle_Experiment_22-07-19.csv",delimiter =",",skip_header=7)

    time = data[:,0]
    voltage = data[:,1]
    current = data[:,2]
    qt = voltage * current 

    q = np.interp(t, time, qt)

    return q

def improved_euler_step_kettle(f, tk, yk, h, qk, qk1, pars):
	""" Compute a single Improved Euler step.
	
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
	# calculate f0 (= f(tk, yk) incl. input parameters)
	f0 = f(tk, yk, qk, *pars)
	# calculate f0 (= f(tk + h, yk + h*f(tk, yk)) incl. input parameters)
	f1 = f((tk + h), yk + h*f0, qk1, *pars)

	# calculate yk after 1 step using average gradients of f0 and f1
	yk1 = yk + h*(f0 + f1)/2

	# return yk after 1 step
	return yk1

def solve_ode_kettle(f, t0, t1, dt, x0, pars):
    ''' Solve an ODE numerically.

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
    '''

    # initialise
    nt = int(np.ceil((t1-t0)/dt))		# compute number of Euler steps to take
    t = t0+np.arange(nt+1)*dt			# x array
    x = 0.*t							# array to store solution
    x[0] = x0							# set initial value
	
	# your code here
    q = interpolate_kettle_heatsource(t)

	# loop through each euler step
    for k in range(nt):
		# y value at the next step is found using the IE method with improved_euler_step()
	    x[k+1] = improved_euler_step_kettle(f, t[k], x[k], dt, q[k], q[k+1], pars)

	# return arrays of t (independent variable) and x (dependent variable)
    return t, x

def plot_kettle_model():
    ''' Plot the kettle LPM over top of the data.

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

    '''

    data = np.genfromtxt("263_Kettle_Experiment_22-07-19.csv",delimiter =",",skip_header=7)
    time = data[:,0]
    Temp = data[:,3]

    fx,ax1 = plt.subplots(nrows=1,ncols=1)   

    ax1.plot(time, Temp, 'ko', label='observations')

    
    t,x = solve_ode_kettle(ode_model, 0, 1200, 0.1, 22, [0.00052, 0.00065, 22])
    t1,x1 = solve_ode_kettle(ode_model, 0, 1200, 0.5, 22, [0.000484, 0.00065, 22])
    ax1.plot(t, x, 'b-', label='model')
    ax1.plot(t1, x1, 'r-', label='model1')


    plt.show()

    pass


if __name__ == "__main__":
    plot_kettle_model()

