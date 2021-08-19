import pytest
from ODE_Model_Function import *
import numpy as np
from copy import copy
from warnings import warn

def test_solve_pressure_ode():
    """
	Tests if function ode_pressure_model is working properly by comparing it with a known result.
	
    """    
    dPdt = [ 1 , -3.875 , -6.921875]
    T, dpdt = solve_pressure_ode(ode_pressure_model, 0, 1, 0.5, 1, [0,0,1,2,3,4])
    assert norm(dpdt - dPdt) < 1.e-10
    
def test_solve_temperature_ode():
    """
	Tests if function ode_temperature_model is working properly by comparing it with a known result.
	
    """    
    # test for when P0> P
    dtdt1 = [1, 1.3125 , 1.5078125]
    T1, dTdt1 = solve_temperature_ode(ode_temperature_model,0, 1, 0.5, 1, [0,0,1,2,3,4,5,6,7])

    # test for when P0 < P
    
    dtdt2 = [1, 62.88746875, 101.56713672])
    T2, dTdt2 = solve_temperature_ode(ode_temperature_model, 0, 1, 0.5, 1, [0,0,1,2,3,4,5,7,6])

    assert norm(dTdt1 - dtdt1) < 1.e-10
    assert norm(dTdt2 - dtdt2) < 1.e-10
