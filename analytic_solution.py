import numpy as np

# Pressure
def pressure_analytic(t, P, q, dqdt, P0, ap, bp, cp):
    # returns analytic solution P for a constant production rate q

    return -ap * q / bp * (1 - np.exp(-bp * t)) + P0


# Temperature
def temperature_analytic(t, T, Tt, Tc, T0, at, bt, ap, bp, P, P0, q):
    # returns analytic solution for temperature ODE with constant q and negligible conduction

    return Tc + (T0 - Tc) * np.exp(-at * q / ap * (np.exp(-bp * t) + bp * t))
