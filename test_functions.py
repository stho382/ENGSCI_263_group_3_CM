import pytest
from ODE_Model_Function import *
import numpy as np
from copy import copy
from warnings import warn


def test_solve_pressure_ode():
    """
    Tests if function ode_pressure_model is working properly by comparing it with a known result.

    """
    dPdt = [1., -1.20896250e04, -1.96462656e04]
    T, dpdt = solve_pressure_ode(ode_pressure_model, 0, 1, 0.5, 1, [0, 0, 1, 2, 3, 4])

    assert abs(dpdt[0] - dPdt[0]) < 1.0e-4
    assert abs(dpdt[1] - dPdt[1]) < 1.0e-4
    assert abs(dpdt[2] - dPdt[2]) < 1.0e-4


def test_solve_temperature_ode_low_pressure():
    """
    Tests if function ode_temperature_model is working properly by comparing it with a known result, when P0 is greater than P.
    """
    # test for when P0 > P
    dtdt1 = [1.        , 1.46875   , 2.42089844]
    T1, dTdt1 = solve_temperature_ode(ode_temperature_model, 0, 1, 0.5, 1, [0, 1, 2, 3, 4, 5, [6, 6, 6], 7])

    # Asserts for when P0 > P
    assert abs(dTdt1[0] - dtdt1[0]) < 1.0e-4
    assert abs(dTdt1[1] - dtdt1[1]) < 1.0e-4
    assert abs(dTdt1[2] - dtdt1[2]) < 1.0e-4


def test_solve_temperature_ode_high_pressure():
    """
    Tests if function ode_temperature_model is working properly by comparing it with a known result, when P0 is less than P.
    """
    dtdt = [1., 1., 1.]
    T, dTdt = solve_temperature_ode(ode_temperature_model, 0, 1, 0.5, 1, [0, 1, 2, 3, 4, 5, [7, 8, 9], 6])
    
    # Asserts for when P0 < P
    assert abs(dTdt[0] - dtdt[0]) < 1.0e-4
    assert abs(dTdt[1] - dtdt[1]) < 1.0e-4
    assert abs(dTdt[2] - dtdt[2]) < 1.0e-4

T, dTdt = solve_temperature_ode(ode_temperature_model, 0, 1, 0.5, 1, [0, 1, 2, 3, 4, 5, [7, 8, 9], 6])
a=1