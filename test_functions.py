import pytest
from ODE_Model_Function import *
import numpy as np
from copy import copy
from warnings import warn

def test_solve_pressure_ode():
    """
	Tests if function ode_pressure_model is working properly by comparing it with a known result.
	
    """    
    dPdt = [ 1.00000000e+00, -1.20896250e+04, -1.96462656e+04]
    T, dpdt = solve_pressure_ode(ode_pressure_model, 0, 1, 0.5, 1, [0,0,1,2,3,4])
    assert abs(dpdt[0] - dPdt[0]) < 1.e-4
    assert abs(dpdt[1] - dPdt[1]) < 1.e-4
    assert abs(dpdt[2] - dPdt[2]) < 1.e-4

