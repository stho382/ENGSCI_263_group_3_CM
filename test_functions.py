import pytest
from ODE_Model_Function import *
import numpy as np
from copy import copy
from warnings import warn


def test_solve_pressure_ode():
    """
    Tests if function ode_pressure_model is working properly by comparing it with a known result.

    """
    dPdt = [1.00000000e00, -1.20896250e04, -1.96462656e04]
    T, dpdt = solve_pressure_ode(ode_pressure_model, 0, 1, 0.5, 1, [0, 0, 1, 2, 3, 4])
    assert abs(dpdt[0] - dPdt[0]) < 1.0e-4
    assert abs(dpdt[1] - dPdt[1]) < 1.0e-4
    assert abs(dpdt[2] - dPdt[2]) < 1.0e-4


def test_solve_temperature_ode():
    """
    Tests if function ode_temperature_model is working properly by comparing it with a known result.
    """
    # test for when P0 > P
    dtdt1 = [1, 1.3125, 1.5078125]
    T1, dTdt1 = solve_temperature_ode(
        ode_temperature_model, 0, 1, 0.5, 1, [0, 1, 2, 3, 4, 5, [6, 6, 6], 7]
    )

    # test for when P0 < P

    dtdt2 = [1, 62.88746875, 101.56713672]
    T2, dTdt2 = solve_temperature_ode(
        ode_temperature_model, 0, 1, 0.5, 1, [0, 1, 2, 3, 4, 5, [7, 7, 7], 6]
    )

    # Asserts for when P0 > P
    assert abs(dTdt1[0] - dtdt1[0]) < 1.0e-4
    assert abs(dTdt1[1] - dtdt1[1]) < 1.0e-4
    assert abs(dTdt1[2] - dtdt1[2]) < 1.0e-4

    # Asserts for when P0 < P
    assert abs(dTdt2[0] - dtdt2[0]) < 1.0e-4
    assert abs(dTdt2[1] - dtdt2[1]) < 1.0e-4
    assert abs(dTdt2[2] - dtdt2[2]) < 1.0e-4
