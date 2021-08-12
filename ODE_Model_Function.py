from types import FrameType
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.npyio import load
from numpy.linalg import norm
from glob import glob
import os

def load_ODE_Model_data():
    ''' Returns data from data file for Project 3.

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
	''' 
    # using glob to find the folder which contains the desired data file 
    files = glob("data" + os.sep + '*')
    
    # loop to extract all the data from the files
    for file in files:
        if file == 'data\\gr_p.txt' or 'data\\gr_q2.txt':
            Data = np.genfromtxt(file, delimiter =",",skip_header=1)
            WaterLevel = Data[:,0]
            Yearp = Data[:,1]
        if file == 'data\\gr_q1.txt':
            Data = np.genfromtxt(file, delimiter =",",skip_header=1)
            Prodq1 = Data[:,0]
            Yearq1 = Data[:,1]
        if file == 'data\\gr_q2.txt':
            Data = np.genfromtxt(file, delimiter =",",skip_header=1)
            Prodq2 = Data[:,0]
            Yearq2 = Data[:,1]
        else:
            Data = np.genfromtxt(file, delimiter =",",skip_header=1)
            Temp = Data[:,0]
            YearT = Data[:,1]

    return WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT 
    
def ode_pressure_model(t, P, P0, ap, bp, cp, q, dqdt): 
	''' 
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
	'''
	dPdt = -ap * q - bp * (P-P0) - cp * dqdt

	return dPdt

def ode_temperature_model(t, T, Tt, Tc, T0, at, bt, ap, bp, P, P0):
	'''
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
	'''

	if P > P0:
		dTdt = at * (bp/ap) * (P - P0) * (Tt - T0) - bt * (T - T0)
	else:
		dTdt = at * (bp/ap) * (P - P0) * (Tc - T0) - bt * (T - T0)
	return dTdt

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
	'''

		# initialise
	nt = int(np.ceil((t1-t0)/dt))		# compute number of Euler steps to take
	ts = t0+np.arange(nt+1)*dt			# x array
	xs = 0.*ts							# array to store solution
	xs[0] = x0							# set initial value
	
	# loop that iterates improved euler'smethod
	for i in range(nt):
		xs[i + 1] = improved_euler_step(f, ts[i], xs[i], dt, pars)
	
	return ts, xs

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

	f0 = f(tk ,yk ,*pars)								# finding predictor
	f1 = f((h + tk), (yk + f0 * h), *pars)				# finding corrector
	yk2 = yk + h * (f0 / 2 + f1 / 2)					# finding solution

	return yk2

WaterLevel, Yearp, Prodq1, Yearq1, Prodq2, Yearq2, Temp, YearT = load_ODE_Model_data()


