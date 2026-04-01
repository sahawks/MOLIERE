# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Lawrence Livermore National Security, LLC
# LLNL-CODE-2017385
# CP 2025-187
# Author: Steven A. Hawks

"""Tkinter GUI for the MOLIERE outgassing model."""

import json
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator
from scipy.optimize import differential_evolution, dual_annealing, minimize
from scipy.sparse import csc_matrix, lil_matrix

LLNL_RELEASE_ID = "LLNL-CODE-2017385"
APP_TITLE = "Outgassing Model Simulator"
MAX_UI_SOURCES = 20

# ============================================================================
# Core simulation functions (from notebook)
# ============================================================================

def temperature_vec(t, dt1, T0, Tfinal, RR):
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    T = np.empty_like(t)
    dTdt = np.zeros_like(t)
    
    # Handle both heating (Tfinal > T0) and cooling (Tfinal < T0)
    if Tfinal > T0:
        # Heating
        t_ramp_end = dt1 + (Tfinal - T0) / RR
        m1 = t <= dt1
        m3 = t > t_ramp_end
        m2 = ~m1 & ~m3
        T[m1] = T0
        T[m2] = T0 + RR * (t[m2] - dt1)
        dTdt[m2] = RR
        T[m3] = Tfinal
    elif Tfinal < T0:
        # Cooling
        t_ramp_end = dt1 + (T0 - Tfinal) / abs(RR)
        m1 = t <= dt1
        m3 = t > t_ramp_end
        m2 = ~m1 & ~m3
        T[m1] = T0
        T[m2] = T0 - abs(RR) * (t[m2] - dt1)
        dTdt[m2] = -abs(RR)
        T[m3] = Tfinal
    else:
        # Isothermal (Tfinal == T0)
        T[:] = T0
    
    if scalar:
        return float(T[0]), float(dTdt[0])
    return T, dTdt

def flow_rate_vec(t, tEq, tFlush, Q_val):
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    period = tEq + tFlush
    if period <= 0:
        Q = np.full_like(t, Q_val, dtype=float)
    else:
        Q = np.where(t % period < tEq, 0.0, Q_val)
    return float(Q[0]) if scalar else Q

def feed_conc_vec(t, n_steps, delta, step_time, base_conc,
                  hold_time_initial, hold_time_final):
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    result = np.full_like(t, base_conc)

    if n_steps <= 0 or step_time <= 0:
        return float(result[0]) if scalar else result

    total_time = hold_time_initial + 2 * n_steps * step_time + hold_time_final
    if total_time <= 0:
        return float(result[0]) if scalar else result

    t_cycle = t % total_time
    active = (t_cycle >= hold_time_initial) & (t_cycle < total_time - hold_time_final)
    if np.any(active):
        t_act = t_cycle[active] - hold_time_initial
        cs = np.clip((t_act / step_time).astype(int), 0, 2 * n_steps - 1)
        rising = cs < n_steps
        result[active] = np.where(
            rising,
            base_conc + (cs + 1) * delta,
            base_conc + (2 * n_steps - cs - 1) * delta,
        )
    return float(result[0]) if scalar else result

def generate_jacobian_sparsity(N, num_srcs, is_inf_F):
    total = N + num_srcs if is_inf_F else N + 1 + num_srcs
    sp = lil_matrix((total, total), dtype=np.int8)
    for i in range(N):
        sp[i, i] = 1
    for i in range(1, N - 1):
        sp[i, i - 1] = 1
        sp[i, i + 1] = 1
    if N > 1:
        sp[0, 1] = 1
    if is_inf_F:
        if N - 1 >= 1: sp[N - 1, N - 2] = 1
        if N - 1 >= 2: sp[N - 1, N - 3] = 1
        for si in range(num_srcs):
            sp[N + si, N + si] = 1
            sp[:N - 1, N + si] = 1
    else:
        if N - 1 >= 1: sp[N - 1, N - 2] = 1
        sp[N - 1, N] = 1
        sp[N, N - 1] = 1
        sp[N, N] = 1
        for si in range(num_srcs):
            row = N + 1 + si
            sp[row, row] = 1
            sp[:N, row] = 1
    return csc_matrix(sp)


def _as_float_array(values, name):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numeric values.")
    return arr


def _validate_simulation_inputs(
    tfinal,
    temp_params,
    flow_params,
    cfeed_params,
    m,
    R,
    mSample,
    rhoSample,
    Vvessel,
    MW_analyte,
    src_params,
    EaK,
    K_ref,
    D_ref,
    EaD,
    cD,
    F,
    c0free,
    cgas_init,
    N,
    rtol,
    atol,
    Tref,
):
    if int(N) < 3:
        raise ValueError("Grid Points (N) must be at least 3.")
    finite_values = {
        'Final Time': tfinal,
        'Sample radius / half-thickness': R,
        'Sample mass': mSample,
        'Density': rhoSample,
        'Vessel volume': Vvessel,
        'MW_analyte': MW_analyte,
        'K_ref': K_ref,
        'D_ref': D_ref,
        'Relative tolerance': rtol,
        'Absolute tolerance': atol,
        'dt1': temp_params['dt1'],
        'T0': temp_params['T0'],
        'Tfinal': temp_params['Tfinal'],
        'RR': temp_params['RR'],
        'tEq': flow_params['tEq'],
        'tFlush': flow_params['tFlush'],
        'Q': flow_params['Q'],
        'delta': cfeed_params['delta'],
        'step_time': cfeed_params['step_time'],
        'base_conc': cfeed_params['base_conc'],
        'hold_time_initial': cfeed_params['hold_time_initial'],
        'hold_time_final': cfeed_params['hold_time_final'],
    }
    for name, value in finite_values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if tfinal <= 0:
        raise ValueError("Final Time must be > 0.")
    if rtol <= 0 or atol <= 0:
        raise ValueError("Solver tolerances must be > 0.")
    if m not in (0, 1, 2):
        raise ValueError("Geometry must be 0 (slab), 1 (cylinder), or 2 (sphere).")
    if R <= 0:
        raise ValueError("Sample radius / half-thickness must be > 0.")
    if mSample <= 0:
        raise ValueError("Sample mass must be > 0.")
    if rhoSample <= 0:
        raise ValueError("Density must be > 0.")
    if Vvessel <= 0:
        raise ValueError("Vessel volume must be > 0.")
    if MW_analyte < 0:
        raise ValueError("MW_analyte must be >= 0.")
    if K_ref <= 0:
        raise ValueError("K_ref must be > 0.")
    if D_ref <= 0:
        raise ValueError("D_ref must be > 0.")
    if flow_params['Q'] < 0:
        raise ValueError("Flow rate Q must be >= 0.")
    if flow_params['tEq'] < 0 or flow_params['tFlush'] < 0:
        raise ValueError("Flow timing parameters must be >= 0.")
    if flow_params['tEq'] + flow_params['tFlush'] <= 0:
        raise ValueError("tEq + tFlush must be > 0.")
    if temp_params['dt1'] < 0:
        raise ValueError("Time at initial temperature must be >= 0.")
    for label, value in [('T0', temp_params['T0']), ('Tfinal', temp_params['Tfinal']), ('Tref', Tref)]:
        if value <= -273.15:
            raise ValueError(f"{label} must be greater than -273.15 °C.")
    if temp_params['Tfinal'] != temp_params['T0'] and temp_params['RR'] <= 0:
        raise ValueError("Ramp rate RR must be > 0 when T0 and Tfinal differ.")
    if cfeed_params['n_steps'] < 0:
        raise ValueError("Number of feed-concentration steps must be >= 0.")
    if cfeed_params['n_steps'] > 0 and cfeed_params['step_time'] <= 0:
        raise ValueError("Step Time must be > 0 when Number of Steps is positive.")
    if cfeed_params['hold_time_initial'] < 0 or cfeed_params['hold_time_final'] < 0:
        raise ValueError("Feed hold times must be >= 0.")
    if np.isfinite(cD) and cD <= 0:
        raise ValueError("Plasticizer power cD must be > 0 when finite.")
    if np.isfinite(F) and F < 0:
        raise ValueError("Surface mass-transfer coefficient F must be >= 0 when finite.")
    if c0free < 0 or cgas_init < 0:
        raise ValueError("Initial concentrations must be >= 0.")
    if src_params.size % 3 != 0:
        raise ValueError("src_params must contain triples of [c0, A, Ea] values.")
    if src_params.size:
        if np.any(src_params[::3] < 0):
            raise ValueError("Source initial concentrations must be >= 0.")
        if np.any(src_params[1::3] < 0):
            raise ValueError("Source pre-exponential factors must be >= 0.")
    for name, value in {
        'EaK': EaK,
        'EaD': EaD,
        'cD': cD,
        'F': F,
        'c0free': c0free,
        'cgas_init': cgas_init,
        'Tref': Tref,
    }.items():
        if not np.isfinite(value) and not (name in {'cD', 'F'} and np.isinf(value)):
            raise ValueError(f"{name} must be finite (or 'inf' for cD/F).")


def make_ode_system(N, x, R, h, cD, beta, dt1, T0, Tfinal, RR, tEq, tFlush, Q_val,
                    n_steps, delta, step_time, base_conc, hold_time_initial,
                    hold_time_final, src_params, Vheadspace, F, Rgas, D0, EaD, K0, EaK, m_geom):
    num_srcs = len(src_params) // 3
    is_inf_F = np.isinf(F)
    is_inf_cD = np.isinf(cD)

    h2 = h * h
    inv_h2 = 1.0 / h2
    inv_2h = 1.0 / (2.0 * h)
    mp1 = m_geom + 1
    inv_R_beta = 1.0 / (R * beta)
    inv_Vhs = 1.0 / Vheadspace
    neg_EaD_Rgas = -EaD / Rgas
    EaK_Rgas = -EaK / Rgas
    
    # Handle both heating and cooling
    if Tfinal > T0:
        t_ramp_end = dt1 + (Tfinal - T0) / RR
    elif Tfinal < T0:
        t_ramp_end = dt1 + (T0 - Tfinal) / abs(RR)
    else:
        t_ramp_end = dt1
    
    period = tEq + tFlush
    total_feed_time = hold_time_initial + 2 * n_steps * step_time + hold_time_final
    inv_cD = 0.0 if is_inf_cD else 1.0 / cD
    x_int = x[1:-1].copy()
    m_over_x_int = m_geom / x_int if m_geom != 0 else np.zeros(N - 2)

    if num_srcs > 0:
        src_A = src_params[1::3].copy()
        src_Ea = src_params[2::3].copy()
    else:
        src_A = np.empty(0)
        src_Ea = np.empty(0)

    n_total = (N + num_srcs) if is_inf_F else (N + 1 + num_srcs)

    def _temp(t):
        if t <= dt1:
            return T0, 0.0
        elif t <= t_ramp_end:
            if Tfinal > T0:
                return T0 + RR * (t - dt1), RR
            elif Tfinal < T0:
                return T0 - abs(RR) * (t - dt1), -abs(RR)
            else:
                return T0, 0.0
        else:
            return Tfinal, 0.0

    def _flow(t):
        if period <= 0:
            return Q_val
        return 0.0 if (t % period) < tEq else Q_val

    def _feed(t):
        if n_steps == 0 or step_time <= 0 or total_feed_time <= 0:
            return base_conc
        tc = t % total_feed_time
        if tc < hold_time_initial or tc >= total_feed_time - hold_time_final:
            return base_conc
        cs = min(max(int((tc - hold_time_initial) / step_time), 0), 2 * n_steps - 1)
        return base_conc + (cs + 1) * delta if cs < n_steps else base_conc + (2 * n_steps - cs - 1) * delta

    if is_inf_F and is_inf_cD:
        src_off = N

        def rhs(t, c):
            TC, dTdt_val = _temp(t)
            T = TC + 273.15
            inv_T = 1.0 / T
            Q = _flow(t)
            cfeed = _feed(t) * 0.001 / 22.414 * 273.15 * inv_T

            K = K0 * np.exp(EaK_Rgas * inv_T)
            dKdT_K = -EaK_Rgas * dTdt_val * inv_T * inv_T
            D = D0 * np.exp(neg_EaD_Rgas * inv_T)

            cs = c[:N]

            src_total = 0.0
            if num_srcs > 0:
                exp_ea = np.exp(-src_Ea / (Rgas * T))
                rates = src_A * exp_ea * c[src_off:src_off + num_srcs]
                src_total = float(np.sum(rates))

            dcdt = np.empty(n_total)

            dcdx = (cs[2:] - cs[:-2]) * inv_2h
            d2cdx2 = (cs[2:] - 2.0 * cs[1:-1] + cs[:-2]) * inv_h2
            dcdt[1:N - 1] = D * (d2cdx2 + m_over_x_int * dcdx) + src_total

            dcdt[0] = 2.0 * D * mp1 * (cs[1] - cs[0]) * inv_h2 + src_total

            dcRdx = (3.0 * cs[-1] - 4.0 * cs[-2] + cs[-3]) * inv_2h
            dcdt[N - 1] = (-mp1 * K * D * dcRdx * inv_R_beta
                           - cs[-1] * (Q * inv_Vhs - dKdT_K)
                           + Q * K * cfeed * inv_Vhs)

            if num_srcs > 0:
                dcdt[src_off:src_off + num_srcs] = -rates

            return dcdt

        sparsity = generate_jacobian_sparsity(N, num_srcs, True)
        return rhs, dict(jac_sparsity=sparsity)

    elif is_inf_F and not is_inf_cD:
        src_off = N

        def rhs(t, c):
            TC, dTdt_val = _temp(t)
            T = TC + 273.15
            inv_T = 1.0 / T
            Q = _flow(t)
            cfeed = _feed(t) * 0.001 / 22.414 * 273.15 * inv_T

            K = K0 * np.exp(EaK_Rgas * inv_T)
            dKdT_K = -EaK_Rgas * dTdt_val * inv_T * inv_T
            cs = c[:N]

            D = D0 * np.exp(cs * inv_cD + neg_EaD_Rgas * inv_T)
            dDdc = D * inv_cD

            src_total = 0.0
            if num_srcs > 0:
                rates = src_A * np.exp(-src_Ea / (Rgas * T)) * c[src_off:src_off + num_srcs]
                src_total = float(np.sum(rates))

            dcdt = np.empty(n_total)
            dcdx = (cs[2:] - cs[:-2]) * inv_2h
            d2cdx2 = (cs[2:] - 2.0 * cs[1:-1] + cs[:-2]) * inv_h2

            dcdt[1:N - 1] = (dDdc[1:-1] * dcdx ** 2
                             + D[1:-1] * (d2cdx2 + m_over_x_int * dcdx) + src_total)
            dcdt[0] = 2.0 * D[0] * mp1 * (cs[1] - cs[0]) * inv_h2 + src_total

            dcRdx = (3.0 * cs[-1] - 4.0 * cs[-2] + cs[-3]) * inv_2h
            dcdt[N - 1] = (-mp1 * K * D[-1] * dcRdx * inv_R_beta
                           - cs[-1] * (Q * inv_Vhs - dKdT_K)
                           + Q * K * cfeed * inv_Vhs)

            if num_srcs > 0:
                dcdt[src_off:src_off + num_srcs] = -rates
            return dcdt

        sparsity = generate_jacobian_sparsity(N, num_srcs, True)
        return rhs, dict(jac_sparsity=sparsity)

    elif not is_inf_F and is_inf_cD:
        src_off = N + 1

        def rhs(t, c):
            TC, _ = _temp(t)
            T = TC + 273.15
            inv_T = 1.0 / T
            Q = _flow(t)
            cfeed = _feed(t) * 0.001 / 22.414 * 273.15 * inv_T

            K = K0 * np.exp(EaK_Rgas * inv_T)
            D = D0 * np.exp(neg_EaD_Rgas * inv_T)

            cs = c[:N]
            cR = c[N - 1]
            c_gas = c[N]

            src_total = 0.0
            if num_srcs > 0:
                rates = src_A * np.exp(-src_Ea / (Rgas * T)) * c[src_off:src_off + num_srcs]
                src_total = float(np.sum(rates))

            dcdt = np.empty(n_total)
            dcdx = (cs[2:] - cs[:-2]) * inv_2h
            d2cdx2 = (cs[2:] - 2.0 * cs[1:-1] + cs[:-2]) * inv_h2

            dcdt[1:N - 1] = D * (d2cdx2 + m_over_x_int * dcdx) + src_total
            dcdt[0] = 2.0 * D * mp1 * (cs[1] - cs[0]) * inv_h2 + src_total

            dcdx_R = 60.0 * F * (c_gas * K - cR) / D
            d2cdx2_R = 2.0 * (h * dcdx_R + cs[N - 2] - cR) * inv_h2
            dcdt[N - 1] = D * (d2cdx2_R + m_geom * dcdx_R / R) + src_total
            dcdt[N] = (mp1 * 60.0 * F * (cR - c_gas * K) / (beta * R)
                       + Q * inv_Vhs * (cfeed - c_gas))

            if num_srcs > 0:
                dcdt[src_off:src_off + num_srcs] = -rates
            return dcdt

        sparsity = generate_jacobian_sparsity(N, num_srcs, False)
        return rhs, dict(jac_sparsity=sparsity)

    else:
        src_off = N + 1

        def rhs(t, c):
            TC, _ = _temp(t)
            T = TC + 273.15
            inv_T = 1.0 / T
            Q = _flow(t)
            cfeed = _feed(t) * 0.001 / 22.414 * 273.15 * inv_T

            K = K0 * np.exp(EaK_Rgas * inv_T)
            cs = c[:N]
            cR = c[N - 1]
            c_gas = c[N]

            D = D0 * np.exp(cs * inv_cD + neg_EaD_Rgas * inv_T)
            dDdc = D * inv_cD

            src_total = 0.0
            if num_srcs > 0:
                rates = src_A * np.exp(-src_Ea / (Rgas * T)) * c[src_off:src_off + num_srcs]
                src_total = float(np.sum(rates))

            dcdt = np.empty(n_total)
            dcdx = (cs[2:] - cs[:-2]) * inv_2h
            d2cdx2 = (cs[2:] - 2.0 * cs[1:-1] + cs[:-2]) * inv_h2

            dcdt[1:N - 1] = (dDdc[1:-1] * dcdx ** 2
                             + D[1:-1] * (d2cdx2 + m_over_x_int * dcdx) + src_total)
            dcdt[0] = 2.0 * D[0] * mp1 * (cs[1] - cs[0]) * inv_h2 + src_total

            dcdx_R = 60.0 * F * (c_gas * K - cR) / D[-1]
            d2cdx2_R = 2.0 * (h * dcdx_R + cs[N - 2] - cR) * inv_h2
            dcdt[N - 1] = (dDdc[-1] * dcdx_R ** 2
                           + D[-1] * (d2cdx2_R + m_geom * dcdx_R / R) + src_total)
            dcdt[N] = (mp1 * 60.0 * F * (cR - c_gas * K) / (beta * R)
                       + Q * inv_Vhs * (cfeed - c_gas))

            if num_srcs > 0:
                dcdt[src_off:src_off + num_srcs] = -rates
            return dcdt

        sparsity = generate_jacobian_sparsity(N, num_srcs, False)
        return rhs, dict(jac_sparsity=sparsity)

def run_simulation(tfinal, temp_params, flow_params, cfeed_params,
                   m, R, mSample, rhoSample, Vvessel, MW_analyte,
                   src_params, EaK, K_ref, D_ref, EaD, cD, F, c0free, cgas_init, N,
                   rtol, atol, Tref=50):
    src_params = _as_float_array(src_params, 'src_params')
    _validate_simulation_inputs(
        tfinal, temp_params, flow_params, cfeed_params,
        m, R, mSample, rhoSample, Vvessel, MW_analyte,
        src_params, EaK, K_ref, D_ref, EaD, cD, F,
        c0free, cgas_init, N, rtol, atol, Tref,
    )

    num_srcs = len(src_params) // 3
    is_inf_F = np.isinf(F)
    vSample = mSample * 1E-3 / rhoSample
    Vheadspace = Vvessel - vSample
    if Vheadspace <= 0:
        raise ValueError(
            'Vessel volume must be larger than the sample volume so the headspace is positive.'
        )
    beta = Vheadspace / vSample
    Rgas = 8.31446261815324 / 1000
    K0 = K_ref * np.exp(EaK / Rgas / (Tref + 273.15))
    D0 = 60 * D_ref * np.exp(EaD / Rgas / (Tref + 273.15))

    x = np.linspace(0, R, N)
    h = x[1] - x[0]
    K_t0 = K0 * np.exp(-EaK / Rgas / (temp_params['T0'] + 273.15))

    c_init = np.full(N, c0free)
    if is_inf_F:
        c_init[-1] = cgas_init * K_t0
        ic = np.hstack((c_init, src_params[::3])) if num_srcs > 0 else c_init
    else:
        ic = np.hstack((c_init, [cgas_init], src_params[::3])) if num_srcs > 0 else np.hstack((c_init, [cgas_init]))

    dt1, T0, Tfinal, RR = temp_params['dt1'], temp_params['T0'], temp_params['Tfinal'], temp_params['RR']
    tEq, tFlush, Q_val = flow_params['tEq'], flow_params['tFlush'], flow_params['Q']
    ns = cfeed_params['n_steps']
    dl = cfeed_params['delta']
    st = cfeed_params['step_time']
    bc = cfeed_params['base_conc']
    hti = cfeed_params['hold_time_initial']
    htf = cfeed_params['hold_time_final']

    rhs, solver_kw = make_ode_system(
        N, x, R, h, cD, beta, dt1, T0, Tfinal, RR, tEq, tFlush, Q_val,
        ns, dl, st, bc, hti, htf, src_params, Vheadspace, F, Rgas, D0, EaD, K0, EaK, m)

    t0 = time.time()
    result = solve_ivp(
        rhs,
        [0, tfinal],
        ic,
        method='BDF',
        rtol=rtol,
        atol=atol,
        max_step=tfinal / 10,
        **solver_kw,
    )
    elapsed = time.time() - t0
    if not result.success:
        raise RuntimeError(f"SciPy solve_ivp failed: {result.message}")
    if not np.all(np.isfinite(result.y)):
        raise RuntimeError('Solver returned non-finite values.')

    t = result.t
    y = result.y.T
    cR = y[:, N - 1]

    T_C, _ = temperature_vec(t, dt1, T0, Tfinal, RR)
    Q = flow_rate_vec(t, tEq, tFlush, Q_val)
    T_K = T_C + 273.15
    K = K0 * np.exp(-EaK / Rgas / T_K)
    S = K * (273.15 / T_K)
    DR = D0 * np.exp(-EaD / Rgas / T_K + cR * (0.0 if np.isinf(cD) else 1.0 / cD))
    DR_cm2s = DR / 60.0  # convert from cm²/min (internal) to cm²/s (output)

    c_feed_ppbv = feed_conc_vec(t, ns, dl, st, bc, hti, htf)
    c_feed_uM = c_feed_ppbv / 1000 / 22.414 * 273.15 / T_K

    c_gas = cR / K if is_inf_F else y[:, N]
    cGasPPBV = c_gas * 1000 * 22.414 / 273.15 * T_K

    # Calculate mass change
    xm = x ** m
    weighted = y[:, :N] * xm[np.newaxis, :]
    initial_integral = simpson(weighted[0], x=x)
    time_integrals = np.array([simpson(weighted[i], x=x) for i in range(len(t))])
    delta_mass_free = -(m + 1) / R ** (m + 1) * vSample * MW_analyte * (initial_integral - time_integrals)

    src_off = N if is_inf_F else N + 1
    if num_srcs > 0:
        src_remaining = y[:, src_off:src_off + num_srcs]
        src_initial = src_params[::3]
        delta_mass_src = -vSample * MW_analyte * np.sum(src_initial - src_remaining, axis=1)
    else:
        delta_mass_src = np.zeros_like(t)

    delta_mass_total = delta_mass_free + delta_mass_src

    # Mass balance
    c0_end = (m + 1) / R ** (m + 1) * simpson(weighted[-1], x=x)
    c_Gas_end = c_gas[-1] * beta
    src_rem = float(np.sum(src_params[::3] - y[-1, src_off:src_off + num_srcs])) if num_srcs > 0 else 0.0
    c_still = c0_end + c_Gas_end - src_rem
    denom = c0free + beta * cgas_init - c_still

    if np.abs(denom) < 1E-3:
        c0Chk = simpson(Q * (c_gas - c_feed_uM) / vSample, x=t)
        mass_bal_error = 100 * c0Chk
    else:
        c0Chk = simpson(Q * (c_gas - c_feed_uM) / vSample, x=t) / denom
        mass_bal_error = 100 * (1 - c0Chk)

    return {
        't': t,
        'T_C': T_C,
        'c_gas_ppbv': cGasPPBV,
        'c_feed_ppbv': c_feed_ppbv,
        'Q': Q,
        'D': DR_cm2s,
        'S': S,
        'delta_mass': delta_mass_total,
        'mass_bal_error': mass_bal_error,
        'solve_time': elapsed,
    }

# ============================================================================
# GUI Application
# ============================================================================

class OutgassingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1600x900")
        
        # Create main frames
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.plot_frame = ttk.Frame(root, padding="10")
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        
        self.source_entries = []  # Will hold dynamic source parameter entries
        self.current_results = None  # Store results for export (single run)
        self.all_sweep_results = []  # List of (label, results) for sweep runs
        self.config_file = os.path.join(os.path.expanduser("~"), ".outgassing_gui_config.json")
        self.defaults_file = self._load_config_defaults_path()

        # Sweep state
        self.sweep_enabled = tk.BooleanVar(value=False)

        # Optimization state
        self.opt_enabled = tk.BooleanVar(value=False)
        self.opt_param_rows = []      # list of dicts per row
        self.opt_n_params = 1         # current number of parameter rows
        self._opt_stop = False        # terminate flag

        # All sweepable parameters: display label -> (attr_name, is_log_sensible)
        self.SWEEP_PARAMS = {
            # ── Temperature ────────────────────────────────────────────────
            "T0 – Initial Temp (°C)":                       ("T0",        False),
            "Tfinal – Final Temp (°C)":                     ("Tfinal",    False),
            "RR – Ramp Rate (°C/min)":                      ("RR",        False),
            # ── Flow ──────────────────────────────────────────────────────
            "Q – Flow Rate (ml/min)":                       ("Q",         False),
            # ── Sample geometry ───────────────────────────────────────────
            "R – Sample Radius / Half-thickness (µm)":      ("R",         True),
            "mSample – Sample Mass (mg)":                   ("mSample",   False),
            "rhoSample – Density (g/ml)":                   ("rhoSample", False),
            "Vvessel – Vessel Volume (ml)":                 ("Vvessel",   False),
            # ── Transport properties ──────────────────────────────────────
            "K – Partition Coeff @ Tref":                    ("K_ref",       True),
            "EaK – Sorption Enthalpy (kJ/mol)":             ("EaK",       False),
            "D – Diffusivity @ Tref,c=0 (cm²/s)":                ("D_ref",       True),
            "EaD – Diffusivity Ea (kJ/mol)":                ("EaD",       False),
            "cD – Plasticization Power (µM)":               ("cD",        True),
            "F – Surface Mass-Transfer Coeff (cm/s)":       ("F",         True),
            # ── Initial conditions ────────────────────────────────────────
            "c0free – Initial Mobile Conc (µM)":            ("c0free",    True),
            "base_conc – Feed Base Conc (ppbv)":            ("base_conc", False),
            # ── Source terms (Source 1) ───────────────────────────────────
            "Src 1 – c₀ Initial Conc (µM)":                 ("src1_c0",   True),
            "Src 1 – A Pre-exponential (1/min)":            ("src1_A",    True),
            "Src 1 – Ea Activation Energy (kJ/mol)":        ("src1_Ea",   False),
            # ── Source terms (Source 2) ───────────────────────────────────
            "Src 2 – c₀ Initial Conc (µM)":                 ("src2_c0",   True),
            "Src 2 – A Pre-exponential (1/min)":            ("src2_A",    True),
            "Src 2 – Ea Activation Energy (kJ/mol)":        ("src2_Ea",   False),
            # ── Source terms (Source 3) ───────────────────────────────────
            "Src 3 – c₀ Initial Conc (µM)":                 ("src3_c0",   True),
            "Src 3 – A Pre-exponential (1/min)":            ("src3_A",    True),
            "Src 3 – Ea Activation Energy (kJ/mol)":        ("src3_Ea",   False),
        }

        self.create_controls()
        self.create_plots()
        
        # Load defaults if file exists
        self.load_defaults()
        
    def _load_config_defaults_path(self):
        """Load the remembered defaults file path from the app config file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    cfg = json.load(f)
                path = os.path.abspath(os.path.expanduser(
                    cfg.get('defaults_file', 'outgassing_defaults.json')
                ))
                if os.path.exists(path):
                    return path
        except Exception:
            pass
        return "outgassing_defaults.json"

    def _save_config(self):
        """Persist the current defaults file path to the app config file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({'defaults_file': os.path.abspath(self.defaults_file)}, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save config: {e}")

    def _update_defaults_label(self):
        """Update the status label showing the active parameter file."""
        display = os.path.basename(self.defaults_file)
        self.defaults_path_label.config(text=f"Param file: {display}")

    def save_defaults_as(self):
        """Save current GUI values to a user-chosen JSON file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Parameters As",
            initialdir=os.path.dirname(self.defaults_file) or ".",
            initialfile=os.path.basename(self.defaults_file),
        )
        if not filename:
            return
        old_file = self.defaults_file
        self.defaults_file = os.path.abspath(filename)
        self.save_defaults()
        # If save was successful, persist the new path
        if os.path.exists(filename):
            self._save_config()
            self._update_defaults_label()
        else:
            self.defaults_file = old_file  # revert on failure

    def load_defaults_from_file(self):
        """Browse for a JSON parameter file and load it into the GUI."""
        filename = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Parameters",
            initialdir=os.path.dirname(self.defaults_file) or ".",
        )
        if not filename:
            return
        if not os.path.exists(filename):
            messagebox.showerror("Load Error", f"File not found:\n{filename}")
            return
        self.defaults_file = os.path.abspath(filename)
        self.load_defaults()
        self._save_config()
        self._update_defaults_label()

    def create_controls(self):
        # Create scrollable frame for controls
        canvas = tk.Canvas(self.control_frame, width=450)
        scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # ── Simulation parameters ────────────────────────────────────────────
        ttk.Label(self.scrollable_frame, text="Simulation Parameters", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        self.tfinal = self.create_entry(self.scrollable_frame, "Final Time (min):", "150", row)
        row += 1
        self.N = self.create_entry(self.scrollable_frame, "Grid Points:", "201", row)
        row += 1
        self.rtol = self.create_entry(self.scrollable_frame, "Relative Tolerance:", "1e-7", row)
        row += 1
        self.atol = self.create_entry(self.scrollable_frame, "Absolute Tolerance:", "1e-8", row)
        row += 1

        # ── Temperature parameters ───────────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1
        ttk.Label(self.scrollable_frame, text="Temperature Profile", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        self.dt1 = self.create_entry(self.scrollable_frame, "Time at Initial Temp (min):", "60", row)
        row += 1
        self.T0 = self.create_entry(self.scrollable_frame, "Initial Temp (°C):", "50", row)
        row += 1
        self.Tfinal = self.create_entry(self.scrollable_frame, "Final Temp (°C):", "250", row)
        row += 1
        self.RR = self.create_entry(self.scrollable_frame, "Ramp Rate (°C/min):", "5", row)
        row += 1
        
        # ── Flow parameters ──────────────────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1
        ttk.Label(self.scrollable_frame, text="Flow Parameters", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        self.tEq = self.create_entry(self.scrollable_frame, "Initial time at Q = 0 (min):", "30", row)
        row += 1
        self.Q = self.create_entry(self.scrollable_frame, "Flow Rate (ml/min):", "20", row)
        row += 1
        
        # ── Feed concentration parameters ────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1
        ttk.Label(self.scrollable_frame, text="Feed Concentration Profile", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        self.n_steps = self.create_entry(self.scrollable_frame, "Number of Steps:", "0", row)
        row += 1
        self.delta = self.create_entry(self.scrollable_frame, "Δc per Step (ppbv):", "1000", row)
        row += 1
        self.step_time = self.create_entry(self.scrollable_frame, "Step Time (min):", "240", row)
        row += 1
        self.base_conc = self.create_entry(self.scrollable_frame, "Base Conc (ppbv):", "0", row)
        row += 1
        self.hold_time_initial = self.create_entry(self.scrollable_frame, "Initial Hold Time (min):", "60", row)
        row += 1
        self.hold_time_final = self.create_entry(self.scrollable_frame, "Final Hold Time (min):", "3000", row)
        row += 1
        
        # ── Sample parameters ────────────────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1
        ttk.Label(self.scrollable_frame, text="Sample Properties", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        ttk.Label(self.scrollable_frame, text="Geometry:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.m = ttk.Combobox(self.scrollable_frame, width=14, state="readonly")
        self.m['values'] = ('0 - Slab', '1 - Cylinder', '2 - Sphere')
        self.m.current(2)
        self.m.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1
        
        self.R = self.create_entry(self.scrollable_frame, "Sample Radius/Slab half-thickness (µm):", "200", row)
        row += 1
        self.mSample = self.create_entry(self.scrollable_frame, "Total sample mass (mg):", "50", row)
        row += 1
        self.rhoSample = self.create_entry(self.scrollable_frame, "Density (g/ml):", "1", row)
        row += 1
        self.Vvessel = self.create_entry(self.scrollable_frame, "Vessel Volume (ml):", "10", row)
        row += 1
        self.MW_analyte = self.create_entry(self.scrollable_frame, "MW (g/mol, for Δm calc):", "18", row)
        row += 1
        
        # ── Transport parameters ─────────────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1
        ttk.Label(self.scrollable_frame, text="Transport Properties", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        self.Tref = self.create_entry(self.scrollable_frame, "Reference Temperature, Tref (°C):", "50", row)
        row += 1
        self.K_ref = self.create_entry(self.scrollable_frame, "Partition coeff K @ Tref:", "150", row)
        row += 1
        self.EaK = self.create_entry(self.scrollable_frame, "Sorption Enthalpy (kJ/mol):", "-35", row)
        row += 1
        self.D_ref = self.create_entry(self.scrollable_frame, "Diffusivity D @ Tref,c=0 (cm²/s):", "1e-7", row)
        row += 1
        self.EaD = self.create_entry(self.scrollable_frame, "Diffusivity Ea (kJ/mol):", "15", row)
        row += 1
        self.cD = self.create_entry(self.scrollable_frame, "Plasticizer power cD (µM, 'inf'):", "inf", row)
        row += 1
        self.F = self.create_entry(self.scrollable_frame, "Surface mass-transfer coeff F (cm/s, 'inf'):", "inf", row)
        row += 1
        self.c0free = self.create_entry(self.scrollable_frame, "Initial mobile concentration (µM):", "10", row)
        row += 1
        
        # ── Source parameters ────────────────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1
        ttk.Label(self.scrollable_frame, text="Source Terms", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        ttk.Label(self.scrollable_frame, text="Number of Sources:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.num_sources = ttk.Entry(self.scrollable_frame, width=12)
        self.num_sources.insert(0, "1")
        self.num_sources.grid(row=row, column=1, sticky=tk.E, pady=2)
        self.num_sources.bind('<Return>', lambda e: self.update_source_fields())
        row += 1
        
        ttk.Button(self.scrollable_frame, text="Update Source Fields", 
                  command=self.update_source_fields).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        # Store row for dynamic source entries
        self.source_start_row = row
        
        # Initialize with one source
        self.update_source_fields()
        
        # Find next row after sources
        row = self.source_start_row + 4 * MAX_UI_SOURCES  # Reserve space for dynamic source rows

        # ── Parameter Sweep ──────────────────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        sweep_header = ttk.Frame(self.scrollable_frame)
        sweep_header.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(5, 2))
        ttk.Label(sweep_header, text="Parameter Sweep",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.sweep_check = ttk.Checkbutton(
            sweep_header, text="Enable",
            variable=self.sweep_enabled,
            command=self._toggle_sweep_ui)
        self.sweep_check.pack(side=tk.RIGHT, padx=4)
        row += 1

        ttk.Label(self.scrollable_frame, text="Sweep Parameter:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.sweep_param_var = tk.StringVar()
        self.sweep_param_combo = ttk.Combobox(
            self.scrollable_frame, textvariable=self.sweep_param_var,
            values=list(self.SWEEP_PARAMS.keys()), width=30, state='disabled')
        self.sweep_param_combo.current(0)
        self.sweep_param_combo.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        ttk.Label(self.scrollable_frame, text="Start Value:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.sweep_start = ttk.Entry(self.scrollable_frame, width=12, state='disabled')
        self.sweep_start.insert(0, "1e-8")
        self.sweep_start.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        ttk.Label(self.scrollable_frame, text="Stop Value:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.sweep_stop = ttk.Entry(self.scrollable_frame, width=12, state='disabled')
        self.sweep_stop.insert(0, "1e-6")
        self.sweep_stop.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        ttk.Label(self.scrollable_frame, text="Number of Points:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.sweep_n = ttk.Entry(self.scrollable_frame, width=12, state='disabled')
        self.sweep_n.insert(0, "5")
        self.sweep_n.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        ttk.Label(self.scrollable_frame, text="Log-spaced Values:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.sweep_log = tk.BooleanVar(value=False)
        self.sweep_log_check = ttk.Checkbutton(
            self.scrollable_frame, variable=self.sweep_log)
        self.sweep_log_check.config(state='disabled')
        self.sweep_log_check.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        ttk.Label(self.scrollable_frame, text="Colormap:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.sweep_cmap = ttk.Combobox(
            self.scrollable_frame, width=18, state="disabled",
            values=['viridis', 'plasma', 'coolwarm', 'tab10', 'rainbow', 'cividis'])
        self.sweep_cmap.set('viridis')
        self.sweep_cmap.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        # ── Parameter Optimization ───────────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        opt_header = ttk.Frame(self.scrollable_frame)
        opt_header.grid(row=row, column=0, columnspan=2, sticky='ew', pady=(5, 2))
        ttk.Label(opt_header, text="Parameter Optimization",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.opt_check = ttk.Checkbutton(
            opt_header, text="Enable",
            variable=self.opt_enabled,
            command=self._toggle_opt_ui)
        self.opt_check.pack(side=tk.RIGHT, padx=4)
        row += 1

        ttk.Label(self.scrollable_frame, text="# Parameters to Fit (1–6):").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.opt_n_entry = ttk.Entry(self.scrollable_frame, width=8, state='disabled')
        self.opt_n_entry.insert(0, "1")
        self.opt_n_entry.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        self.opt_build_btn = ttk.Button(
            self.scrollable_frame, text="Build Parameter Rows",
            command=self._build_opt_rows, state='disabled')
        self.opt_build_btn.grid(row=row, column=0, columnspan=2, pady=3)
        row += 1

        # Column headers
        self.opt_col_header = ttk.Frame(self.scrollable_frame)
        self.opt_col_header.grid(row=row, column=0, columnspan=2, sticky='ew')
        for text, w in [("Parameter", 28), ("Min", 8), ("Max", 8), ("Log?", 4)]:
            ttk.Label(self.opt_col_header, text=text, font=('Arial', 8, 'bold'),
                      width=w, anchor='center').pack(side=tk.LEFT, padx=1)
        row += 1

        # Dynamic parameter rows frame
        self.opt_rows_frame = ttk.Frame(self.scrollable_frame)
        self.opt_rows_frame.grid(row=row, column=0, columnspan=2, sticky='ew')
        row += 1

        self._build_opt_rows()

        ttk.Label(self.scrollable_frame, text="Fit Target:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.opt_target = ttk.Combobox(
            self.scrollable_frame,
            values=["Concentration (ppbv)", "Mass (ng)", "Both"],
            width=20, state='disabled')
        self.opt_target.set("Concentration (ppbv)")
        self.opt_target.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        ttk.Label(self.scrollable_frame, text="Method:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.opt_method = ttk.Combobox(
            self.scrollable_frame,
            values=["Differential Evolution", "Dual Annealing", "Nelder-Mead"],
            width=20, state='disabled')
        self.opt_method.set("Differential Evolution")
        self.opt_method.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        ttk.Label(self.scrollable_frame, text="Max Iterations:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        self.opt_maxiter = ttk.Entry(self.scrollable_frame, width=8, state='disabled')
        self.opt_maxiter.insert(0, "200")
        self.opt_maxiter.grid(row=row, column=1, sticky=tk.E, pady=2)
        row += 1

        # Run / Terminate buttons side by side
        opt_btn_frame = ttk.Frame(self.scrollable_frame)
        opt_btn_frame.grid(row=row, column=0, columnspan=2, pady=5)
        self.opt_run_btn = ttk.Button(
            opt_btn_frame, text="Run Optimization",
            command=self.run_optimization, state='disabled')
        self.opt_run_btn.pack(side=tk.LEFT, padx=5)
        self.opt_stop_btn = ttk.Button(
            opt_btn_frame, text="Terminate Fit",
            command=self._terminate_optimization, state='disabled')
        self.opt_stop_btn.pack(side=tk.LEFT, padx=5)
        row += 1

        # ── Run / Save / Export buttons ──────────────────────────────────────
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=5)
        row += 1

        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Run Simulation", 
                  command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save as Default", 
                  command=self.save_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Data", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        row += 1

        # Second row of buttons for parameter file management
        file_btn_frame = ttk.Frame(self.scrollable_frame)
        file_btn_frame.grid(row=row, column=0, columnspan=2, pady=(0, 5))
        ttk.Button(file_btn_frame, text="Save Params As…",
                  command=self.save_defaults_as).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_btn_frame, text="Load Params…",
                  command=self.load_defaults_from_file).pack(side=tk.LEFT, padx=5)
        row += 1

        # Label showing the active parameter file
        self.defaults_path_label = ttk.Label(
            self.scrollable_frame,
            text=f"Param file: {os.path.basename(self.defaults_file)}",
            foreground="gray", font=('Arial', 8))
        self.defaults_path_label.grid(row=row, column=0, columnspan=2, pady=(0, 3))
        row += 1
        
        # Status label
        self.status_label = ttk.Label(self.scrollable_frame, text="Ready", foreground="green")
        self.status_label.grid(row=row, column=0, columnspan=2, pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _toggle_sweep_ui(self):
        """Enable or disable sweep-specific widgets based on checkbox state."""
        new_state = 'normal' if self.sweep_enabled.get() else 'disabled'
        for widget in (self.sweep_param_combo, self.sweep_start,
                       self.sweep_stop, self.sweep_n, self.sweep_cmap):
            widget.config(state=new_state if widget is not self.sweep_param_combo
                          else ('readonly' if self.sweep_enabled.get() else 'disabled'))
        self.sweep_log_check.config(state=new_state)

    def create_entry(self, parent, label, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        entry = ttk.Entry(parent, width=12)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky=tk.E, pady=2)
        return entry
    
    def save_defaults(self):
        """Save current GUI values as defaults to a JSON file"""
        try:
            defaults = {
                'tfinal': self.tfinal.get(),
                'N': self.N.get(),
                'rtol': self.rtol.get(),
                'atol': self.atol.get(),
                'dt1': self.dt1.get(),
                'T0': self.T0.get(),
                'Tfinal': self.Tfinal.get(),
                'RR': self.RR.get(),
                'tEq': self.tEq.get(),
                'Q': self.Q.get(),
                'n_steps': self.n_steps.get(),
                'delta': self.delta.get(),
                'step_time': self.step_time.get(),
                'base_conc': self.base_conc.get(),
                'hold_time_initial': self.hold_time_initial.get(),
                'hold_time_final': self.hold_time_final.get(),
                'm': self.m.get(),
                'R': self.R.get(),
                'mSample': self.mSample.get(),
                'rhoSample': self.rhoSample.get(),
                'Vvessel': self.Vvessel.get(),
                'MW_analyte': self.MW_analyte.get(),
                'K_ref': self.K_ref.get(),
                'EaK': self.EaK.get(),
                'D_ref': self.D_ref.get(),
                'EaD': self.EaD.get(),
                'Tref': self.Tref.get(),
                'cD': self.cD.get(),
                'F': self.F.get(),
                'c0free': self.c0free.get(),
                'num_sources': self.num_sources.get(),
            }
            
            # Save source parameters
            sources = []
            for src_widgets in self.source_entries:
                sources.append({
                    'c0': src_widgets[1].get(),
                    'A': src_widgets[3].get(),
                    'Ea': src_widgets[5].get()
                })
            defaults['sources'] = sources
            
            # Save sweep settings
            defaults['sweep_enabled'] = self.sweep_enabled.get()
            defaults['sweep_param'] = self.sweep_param_var.get()
            defaults['sweep_start'] = self.sweep_start.get()
            defaults['sweep_stop'] = self.sweep_stop.get()
            defaults['sweep_n'] = self.sweep_n.get()
            defaults['sweep_log'] = self.sweep_log.get()
            defaults['sweep_cmap'] = self.sweep_cmap.get()

            # Save optimization settings
            defaults['opt_enabled'] = self.opt_enabled.get()
            defaults['opt_n'] = self.opt_n_entry.get()
            defaults['opt_target'] = self.opt_target.get()
            defaults['opt_method'] = self.opt_method.get()
            defaults['opt_maxiter'] = self.opt_maxiter.get()
            opt_rows_data = []
            for rd in self.opt_param_rows:
                opt_rows_data.append({
                    'param': rd['combo'].get(),
                    'min': rd['min_entry'].get(),
                    'max': rd['max_entry'].get(),
                    'log': rd['log_var'].get(),
                })
            defaults['opt_rows'] = opt_rows_data

            # Save experimental data
            defaults['exp_conc_data'] = self.conc_data_text.get("1.0", tk.END).strip()
            defaults['exp_mass_data'] = self.mass_data_text.get("1.0", tk.END).strip()
            defaults['conc_interp'] = self.conc_interp_var.get()
            defaults['conc_interp_n'] = self.conc_interp_n.get()
            defaults['conc_interp_method'] = self.conc_interp_method.get()
            defaults['mass_interp'] = self.mass_interp_var.get()
            defaults['mass_interp_n'] = self.mass_interp_n.get()
            defaults['mass_interp_method'] = self.mass_interp_method.get()
            
            with open(self.defaults_file, 'w') as f:
                json.dump(defaults, f, indent=2)
            
            display = os.path.basename(self.defaults_file)
            self.status_label.config(text=f"Saved to {display}", foreground="green")
            self._update_defaults_label()
            messagebox.showinfo("Save Defaults",
                                f"Parameters saved to:\n{self.defaults_file}")
            
        except Exception as e:
            self.status_label.config(text=f"Error saving defaults: {str(e)}", foreground="red")
            messagebox.showerror("Save Error", f"Failed to save defaults:\n{str(e)}")
    
    def load_defaults(self):
        """Load default values from JSON file if it exists"""
        if not os.path.exists(self.defaults_file):
            return
        
        try:
            with open(self.defaults_file, 'r') as f:
                defaults = json.load(f)
            
            # Load basic parameters
            self.tfinal.delete(0, tk.END)
            self.tfinal.insert(0, defaults.get('tfinal', '150'))
            
            self.N.delete(0, tk.END)
            self.N.insert(0, defaults.get('N', '201'))
            
            self.rtol.delete(0, tk.END)
            self.rtol.insert(0, defaults.get('rtol', '1e-7'))
            
            self.atol.delete(0, tk.END)
            self.atol.insert(0, defaults.get('atol', '1e-8'))
            
            self.dt1.delete(0, tk.END)
            self.dt1.insert(0, defaults.get('dt1', '60'))
            
            self.T0.delete(0, tk.END)
            self.T0.insert(0, defaults.get('T0', '50'))
            
            self.Tfinal.delete(0, tk.END)
            self.Tfinal.insert(0, defaults.get('Tfinal', '250'))
            
            self.RR.delete(0, tk.END)
            self.RR.insert(0, defaults.get('RR', '5'))
            
            self.tEq.delete(0, tk.END)
            self.tEq.insert(0, defaults.get('tEq', '30'))
            
            self.Q.delete(0, tk.END)
            self.Q.insert(0, defaults.get('Q', '20'))
            
            self.n_steps.delete(0, tk.END)
            self.n_steps.insert(0, defaults.get('n_steps', '0'))
            
            self.delta.delete(0, tk.END)
            self.delta.insert(0, defaults.get('delta', '1000'))
            
            self.step_time.delete(0, tk.END)
            self.step_time.insert(0, defaults.get('step_time', '240'))
            
            self.base_conc.delete(0, tk.END)
            self.base_conc.insert(0, defaults.get('base_conc', '0'))
            
            self.hold_time_initial.delete(0, tk.END)
            self.hold_time_initial.insert(0, defaults.get('hold_time_initial', '60'))
            
            self.hold_time_final.delete(0, tk.END)
            self.hold_time_final.insert(0, defaults.get('hold_time_final', '3000'))
            
            # Set geometry combobox
            m_val = defaults.get('m', '2 - Sphere')
            try:
                idx = self.m['values'].index(m_val)
                self.m.current(idx)
            except:
                self.m.current(2)
            
            self.R.delete(0, tk.END)
            self.R.insert(0, defaults.get('R', '200'))
            
            self.mSample.delete(0, tk.END)
            self.mSample.insert(0, defaults.get('mSample', '50'))
            
            self.rhoSample.delete(0, tk.END)
            self.rhoSample.insert(0, defaults.get('rhoSample', '1'))
            
            self.Vvessel.delete(0, tk.END)
            self.Vvessel.insert(0, defaults.get('Vvessel', '10'))
            
            self.MW_analyte.delete(0, tk.END)
            self.MW_analyte.insert(0, defaults.get('MW_analyte', '18'))
            
            self.K_ref.delete(0, tk.END)
            self.K_ref.insert(0, defaults.get('K_ref', '150'))
            
            self.EaK.delete(0, tk.END)
            self.EaK.insert(0, defaults.get('EaK', '-35'))
            
            self.D_ref.delete(0, tk.END)
            self.D_ref.insert(0, defaults.get('D_ref', '1e-7'))
            
            self.EaD.delete(0, tk.END)
            self.EaD.insert(0, defaults.get('EaD', '15'))
            
            self.Tref.delete(0, tk.END)
            self.Tref.insert(0, defaults.get('Tref', '50'))
            
            self.cD.delete(0, tk.END)
            self.cD.insert(0, defaults.get('cD', 'inf'))
            
            self.F.delete(0, tk.END)
            self.F.insert(0, defaults.get('F', 'inf'))
            
            self.c0free.delete(0, tk.END)
            self.c0free.insert(0, defaults.get('c0free', '10'))
            
            # Load number of sources and update fields
            self.num_sources.delete(0, tk.END)
            self.num_sources.insert(0, defaults.get('num_sources', '1'))
            self.update_source_fields()
            
            # Load source parameters
            if 'sources' in defaults:
                for i, src_data in enumerate(defaults['sources']):
                    if i < len(self.source_entries):
                        self.source_entries[i][1].delete(0, tk.END)
                        self.source_entries[i][1].insert(0, src_data.get('c0', '100'))
                        
                        self.source_entries[i][3].delete(0, tk.END)
                        self.source_entries[i][3].insert(0, src_data.get('A', '1e8'))
                        
                        self.source_entries[i][5].delete(0, tk.END)
                        self.source_entries[i][5].insert(0, src_data.get('Ea', '80'))
            
            # Load experimental data (always clear first so stale data is removed)
            self.conc_data_text.delete("1.0", tk.END)
            if defaults.get('exp_conc_data'):
                self.conc_data_text.insert("1.0", defaults['exp_conc_data'])

            self.mass_data_text.delete("1.0", tk.END)
            if defaults.get('exp_mass_data'):
                self.mass_data_text.insert("1.0", defaults['exp_mass_data'])

            # Load interpolation settings
            self.conc_interp_var.set(defaults.get('conc_interp', False))
            self.conc_interp_n.delete(0, tk.END)
            self.conc_interp_n.insert(0, defaults.get('conc_interp_n', '100'))
            conc_method = defaults.get('conc_interp_method', 'linear')
            if conc_method in self.conc_interp_method['values']:
                self.conc_interp_method.set(conc_method)
            self.mass_interp_var.set(defaults.get('mass_interp', False))
            self.mass_interp_n.delete(0, tk.END)
            self.mass_interp_n.insert(0, defaults.get('mass_interp_n', '100'))
            mass_method = defaults.get('mass_interp_method', 'linear')
            if mass_method in self.mass_interp_method['values']:
                self.mass_interp_method.set(mass_method)

            # Load sweep settings
            self.sweep_enabled.set(defaults.get('sweep_enabled', False))
            sweep_param = defaults.get('sweep_param', '')
            if sweep_param in self.sweep_param_combo['values']:
                self.sweep_param_var.set(sweep_param)
            # Enable/disable sweep widgets to match restored state
            self._toggle_sweep_ui()
            # Temporarily enable entries to insert values, then restore state
            for entry, key, fallback in [
                (self.sweep_start, 'sweep_start', '1e-8'),
                (self.sweep_stop,  'sweep_stop',  '1e-6'),
                (self.sweep_n,     'sweep_n',     '5'),
                (self.sweep_cmap,  'sweep_cmap',  'viridis'),
            ]:
                entry.config(state='normal')
                entry.delete(0, tk.END)
                entry.insert(0, defaults.get(key, fallback))
            self.sweep_log.set(defaults.get('sweep_log', False))
            # Re-apply correct disabled/enabled state
            self._toggle_sweep_ui()

            # Load optimization settings
            self.opt_enabled.set(defaults.get('opt_enabled', False))
            self.opt_n_entry.config(state='normal')
            self.opt_n_entry.delete(0, tk.END)
            self.opt_n_entry.insert(0, defaults.get('opt_n', '1'))
            self.opt_maxiter.config(state='normal')
            self.opt_maxiter.delete(0, tk.END)
            self.opt_maxiter.insert(0, defaults.get('opt_maxiter', '200'))
            opt_target_val = defaults.get('opt_target', 'Concentration (ppbv)')
            if opt_target_val in self.opt_target['values']:
                self.opt_target.set(opt_target_val)
            opt_method_val = defaults.get('opt_method', 'Differential Evolution')
            if opt_method_val in self.opt_method['values']:
                self.opt_method.set(opt_method_val)
            # Rebuild rows and restore their values
            self._build_opt_rows()
            opt_rows_data = defaults.get('opt_rows', [])
            param_names = list(self.SWEEP_PARAMS.keys())
            for i, (rd, rd_data) in enumerate(zip(self.opt_param_rows, opt_rows_data)):
                param = rd_data.get('param', '')
                if param in param_names:
                    rd['combo'].set(param)
                rd['min_entry'].config(state='normal')
                rd['min_entry'].delete(0, tk.END)
                rd['min_entry'].insert(0, rd_data.get('min', '1e-9'))
                rd['max_entry'].config(state='normal')
                rd['max_entry'].delete(0, tk.END)
                rd['max_entry'].insert(0, rd_data.get('max', '1e-5'))
                rd['log_var'].set(rd_data.get('log', True))
            # Re-apply correct disabled/enabled state for optimization
            self._toggle_opt_ui()
            
            self.status_label.config(
                text=f"Loaded: {os.path.basename(self.defaults_file)}", foreground="green")
            
        except Exception as e:
            print(f"Error loading defaults: {e}")
            self.status_label.config(text="Using built-in defaults", foreground="blue")
    
    def update_source_fields(self):
        existing_values = []
        for widget_list in self.source_entries:
            existing_values.append({
                'c0': widget_list[1].get(),
                'A': widget_list[3].get(),
                'Ea': widget_list[5].get(),
            })
            for widget in widget_list:
                widget.destroy()
        self.source_entries = []

        try:
            n_sources = max(0, int(self.num_sources.get()))
        except (ValueError, tk.TclError):
            n_sources = 1

        if n_sources > MAX_UI_SOURCES:
            n_sources = MAX_UI_SOURCES
            self.num_sources.delete(0, tk.END)
            self.num_sources.insert(0, str(MAX_UI_SOURCES))
            if hasattr(self, 'status_label'):
                self.status_label.config(
                    text=f"Source count capped at {MAX_UI_SOURCES} for GUI layout.",
                    foreground='blue',
                )

        row = self.source_start_row

        for i in range(n_sources):
            prev = existing_values[i] if i < len(existing_values) else {}

            ttk.Label(self.scrollable_frame, text=f"--- Source {i+1} ---",
                     font=('Arial', 9, 'italic')).grid(row=row, column=0, columnspan=2, pady=3)
            row += 1

            c0_label = ttk.Label(self.scrollable_frame, text=f"  c₀,{i+1} (µM):")
            c0_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            c0_entry = ttk.Entry(self.scrollable_frame, width=12)
            c0_entry.insert(0, prev.get('c0', '100'))
            c0_entry.grid(row=row, column=1, sticky=tk.E, pady=2)
            row += 1

            A_label = ttk.Label(self.scrollable_frame, text=f"  A{i+1} (1/min):")
            A_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            A_entry = ttk.Entry(self.scrollable_frame, width=12)
            A_entry.insert(0, prev.get('A', '1e8'))
            A_entry.grid(row=row, column=1, sticky=tk.E, pady=2)
            row += 1

            Ea_label = ttk.Label(self.scrollable_frame, text=f"  Ea,{i+1} (kJ/mol):")
            Ea_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            Ea_entry = ttk.Entry(self.scrollable_frame, width=12)
            Ea_entry.insert(0, prev.get('Ea', '80'))
            Ea_entry.grid(row=row, column=1, sticky=tk.E, pady=2)
            row += 1

            self.source_entries.append([c0_label, c0_entry, A_label, A_entry, Ea_label, Ea_entry])

    def create_plots(self):
        # Create frame for plots and experimental data
        plot_container = ttk.Frame(self.plot_frame)
        plot_container.pack(fill=tk.BOTH, expand=True)
        
        # Plots frame
        plots_frame = ttk.Frame(plot_container)
        plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(15, 11.5), dpi=100)
        
        # Create grid spec for custom layout with minimal spacing
        gs = self.fig.add_gridspec(3, 3, hspace=0.25, wspace=0.22, 
                                   left=0.055, right=0.98, top=0.97, bottom=0.04)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Temperature
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Feed Concentration (swapped)
        self.ax3 = self.fig.add_subplot(gs[0, 2])  # Flow Rate
        self.ax4 = self.fig.add_subplot(gs[1, 0])  # Diffusivity
        self.ax5 = self.fig.add_subplot(gs[1, 1])  # Solubility
        self.ax6 = self.fig.add_subplot(gs[1, 2])  # Sample Mass Change
        self.ax7 = self.fig.add_subplot(gs[2, 0])  # Gas Concentration (swapped)
        
        # Reserve space for experimental data inputs in bottom middle/right
        self.ax_exp = self.fig.add_subplot(gs[2, 1:])  # Spans bottom middle and right
        self.ax_exp.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add experimental data inputs embedded in the plot area
        exp_frame = ttk.LabelFrame(plots_frame, text="Experimental Data", padding="10")
        exp_frame.place(relx=0.38, rely=0.68, relwidth=0.61, relheight=0.30)
        
        # Concentration data
        conc_frame = ttk.Frame(exp_frame)
        conc_frame.pack(side=tk.LEFT, padx=15, expand=True, fill=tk.BOTH)
        ttk.Label(conc_frame, text="Concentration (time, ppbv):", font=('Arial', 9, 'bold')).pack()
        self.conc_data_text = tk.Text(conc_frame, height=5, width=34, font=('Courier', 9))
        self.conc_data_text.pack(expand=True, fill=tk.BOTH)
        conc_interp_frame = ttk.Frame(conc_frame)
        conc_interp_frame.pack(fill=tk.X, pady=(2, 0))
        self.conc_interp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(conc_interp_frame, text="Interp N:",
                         variable=self.conc_interp_var).pack(side=tk.LEFT)
        self.conc_interp_n = ttk.Entry(conc_interp_frame, width=5)
        self.conc_interp_n.insert(0, "100")
        self.conc_interp_n.pack(side=tk.LEFT, padx=2)
        self.conc_interp_method = ttk.Combobox(
            conc_interp_frame, width=10, state='readonly',
            values=["linear", "cubic", "pchip", "akima"])
        self.conc_interp_method.set("linear")
        self.conc_interp_method.pack(side=tk.LEFT, padx=2)
        
        # Mass data
        mass_frame = ttk.Frame(exp_frame)
        mass_frame.pack(side=tk.LEFT, padx=15, expand=True, fill=tk.BOTH)
        ttk.Label(mass_frame, text="Mass (time, ng):", font=('Arial', 9, 'bold')).pack()
        self.mass_data_text = tk.Text(mass_frame, height=5, width=34, font=('Courier', 9))
        self.mass_data_text.pack(expand=True, fill=tk.BOTH)
        mass_interp_frame = ttk.Frame(mass_frame)
        mass_interp_frame.pack(fill=tk.X, pady=(2, 0))
        self.mass_interp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mass_interp_frame, text="Interp N:",
                         variable=self.mass_interp_var).pack(side=tk.LEFT)
        self.mass_interp_n = ttk.Entry(mass_interp_frame, width=5)
        self.mass_interp_n.insert(0, "100")
        self.mass_interp_n.pack(side=tk.LEFT, padx=2)
        self.mass_interp_method = ttk.Combobox(
            mass_interp_frame, width=10, state='readonly',
            values=["linear", "cubic", "pchip", "akima"])
        self.mass_interp_method.set("linear")
        self.mass_interp_method.pack(side=tk.LEFT, padx=2)
    
    def parse_experimental_data(self, text_widget, interp_var=None, interp_n_widget=None,
                                interp_method_widget=None):
        """Parse experimental data from text widget (CSV or tab-delimited).
        Optionally interpolate to equally spaced points if interp_var is True.
        interp_method_widget: Combobox with method name (linear, cubic, pchip, akima)."""
        data_str = text_widget.get("1.0", tk.END).strip()

        if not data_str:
            return None, None

        try:
            lines = data_str.split('\n')
            time_data = []
            value_data = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Try comma delimiter first, then tab, then arbitrary whitespace.
                if ',' in line:
                    parts = [part.strip() for part in line.split(',')]
                elif '\t' in line:
                    parts = [part.strip() for part in line.split('\t')]
                else:
                    parts = line.split()

                if len(parts) >= 2:
                    time_data.append(float(parts[0]))
                    value_data.append(float(parts[1]))

            if not time_data:
                return None, None

            t_arr = np.asarray(time_data, dtype=float)
            v_arr = np.asarray(value_data, dtype=float)

            order = np.argsort(t_arr, kind='mergesort')
            t_arr = t_arr[order]
            v_arr = v_arr[order]

            unique_t, inverse = np.unique(t_arr, return_inverse=True)
            if unique_t.size != t_arr.size:
                counts = np.bincount(inverse)
                sums = np.bincount(inverse, weights=v_arr)
                t_arr = unique_t
                v_arr = sums / counts

            # Apply interpolation if requested
            if (interp_var is not None and interp_var.get()
                    and interp_n_widget is not None and len(t_arr) >= 2):
                try:
                    n_pts = max(2, int(interp_n_widget.get()))
                except (ValueError, tk.TclError):
                    n_pts = 100
                t_eq = np.linspace(t_arr.min(), t_arr.max(), n_pts)
                method = 'linear'
                if interp_method_widget is not None:
                    try:
                        method = interp_method_widget.get()
                    except (tk.TclError, AttributeError):
                        pass
                if method == 'akima' and len(t_arr) < 5:
                    method = 'linear'
                try:
                    if method == 'linear':
                        v_eq = np.interp(t_eq, t_arr, v_arr)
                    elif method == 'cubic':
                        v_eq = CubicSpline(t_arr, v_arr)(t_eq)
                    elif method == 'pchip':
                        v_eq = PchipInterpolator(t_arr, v_arr)(t_eq)
                    elif method == 'akima':
                        v_eq = Akima1DInterpolator(t_arr, v_arr)(t_eq)
                    else:
                        v_eq = np.interp(t_eq, t_arr, v_arr)
                except Exception:
                    v_eq = np.interp(t_eq, t_arr, v_arr)
                return t_eq, v_eq
            return t_arr, v_arr

        except Exception as e:
            print(f"Error parsing experimental data: {e}")
            return None, None

    def _base_params_dict(self):
        """Build the shared simulation-parameter header dict for export."""
        m_str = self.m.get()
        cD_str = self.cD.get().strip().lower()
        F_str  = self.F.get().strip().lower()
        cD_display = 'inf' if cD_str in ('inf', 'infinity') else cD_str
        F_display  = 'inf' if F_str  in ('inf', 'infinity') else F_str
        params = {
            'Final Time (min)':               self.tfinal.get(),
            'Grid Points':                    self.N.get(),
            'Relative Tolerance':             self.rtol.get(),
            'Absolute Tolerance':             self.atol.get(),
            'Time at Initial Temp (min)':     self.dt1.get(),
            'Initial Temp (C)':              self.T0.get(),
            'Final Temp (C)':                self.Tfinal.get(),
            'Ramp Rate (C/min)':             self.RR.get(),
            'Initial time at Q=0 (min)':      self.tEq.get(),
            'Flow Rate (ml/min)':             self.Q.get(),
            'Feed n_steps':                   self.n_steps.get(),
            'Feed delta (ppbv)':              self.delta.get(),
            'Feed step_time (min)':           self.step_time.get(),
            'Feed base_conc (ppbv)':          self.base_conc.get(),
            'Feed hold_time_initial (min)':   self.hold_time_initial.get(),
            'Feed hold_time_final (min)':     self.hold_time_final.get(),
            'Geometry':                       m_str,
            'Sample Radius (um)':            self.R.get(),
            'Total sample mass (mg)':         self.mSample.get(),
            'Density (g/ml)':                self.rhoSample.get(),
            'Vessel Volume (ml)':            self.Vvessel.get(),
            'MW (g/mol)':                    self.MW_analyte.get(),
            'Reference Temp (C)':            self.Tref.get(),
            'K @ Tref':                      self.K_ref.get(),
            'Sorption Enthalpy (kJ/mol)': self.EaK.get(),
            'D @ Tref,c=0 (cm^2/s)':             self.D_ref.get(),
            'Ea_D (kJ/mol)':                 self.EaD.get(),
            'Plasticizer power cD (uM)':      cD_display,
            'Surface mass-transfer coeff F (cm/s)': F_display,
            'Initial mobile concentration (uM)':    self.c0free.get(),
        }
        for i, sw in enumerate(self.source_entries):
            params[f'Source_{i+1}_c0 (uM)']    = sw[1].get()
            params[f'Source_{i+1}_A (1/min)']   = sw[3].get()
            params[f'Source_{i+1}_Ea (kJ/mol)'] = sw[5].get()
        return params

    @staticmethod
    def _results_to_df(results):
        return pd.DataFrame({
            't (min)':               results['t'],
            'T (C)':                 results['T_C'],
            'c_gas (ppbv)':          results['c_gas_ppbv'],
            'c_feed (ppbv)':         results['c_feed_ppbv'],
            'Q (ml/min)':            results['Q'],
            'D (cm^2/s)':            results['D'],
            'S (cm^3@STP/atm/cm^3)': results['S'],
            'delta_m (ng)':          results['delta_mass'],
        })

    @staticmethod
    def _compute_r_squared(sim_time, sim_data, exp_time, exp_data):
        """Compute R² (coefficient of determination) between simulation and experiment.

        The simulated curve is linearly interpolated at the experimental time
        points.  Returns None when fewer than 2 experimental points fall inside
        the simulation time range, or when SS_tot ≈ 0 (constant data).
        """
        if exp_time is None or exp_data is None or len(exp_time) < 2:
            return None
        # Only use experimental points within the simulation time range
        mask = (exp_time >= sim_time[0]) & (exp_time <= sim_time[-1])
        t_exp = exp_time[mask]
        y_exp = exp_data[mask]
        if len(t_exp) < 2:
            return None
        y_sim = np.interp(t_exp, sim_time, sim_data)
        ss_res = np.sum((y_exp - y_sim) ** 2)
        ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
        if ss_tot == 0.0:
            return None
        return 1.0 - ss_res / ss_tot

    def export_data(self):
        if self.current_results is None:
            messagebox.showwarning("No Data", "Please run a simulation first before exporting data.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Simulation Results"
        )
        if not filename:
            return

        try:
            is_sweep = bool(self.all_sweep_results)
            params   = self._base_params_dict()

            # Parse experimental data once
            exp_conc_time, exp_conc_data = self.parse_experimental_data(
                self.conc_data_text, self.conc_interp_var, self.conc_interp_n, self.conc_interp_method)
            exp_mass_time, exp_mass_data = self.parse_experimental_data(
                self.mass_data_text, self.mass_interp_var, self.mass_interp_n, self.mass_interp_method)

            with open(filename, 'w') as f:
                f.write(f"# Outgassing Model Simulation Results\n# Release: {LLNL_RELEASE_ID}\n#\n")

                if is_sweep:
                    sweep_label = self.sweep_param_var.get()
                    f.write(f"# SWEEP MODE: {sweep_label}\n")
                    f.write(f"# Number of sweep points: {len(self.all_sweep_results)}\n#\n")

                f.write("# BASE SIMULATION PARAMETERS:\n")
                for key, value in params.items():
                    f.write(f"#   {key}: {value}\n")
                f.write("#\n")

                if not is_sweep:
                    # ── Single run ───────────────────────────────────────────
                    res = self.current_results
                    f.write(f"# Mass Balance Error (%): {res['mass_bal_error']:.6e}\n")
                    f.write(f"# Solve Time (s): {res['solve_time']:.4f}\n")
                    # R² (coefficient of determination)
                    r2_conc = self._compute_r_squared(
                        res['t'], res['c_gas_ppbv'], exp_conc_time, exp_conc_data)
                    r2_mass = self._compute_r_squared(
                        res['t'], res['delta_mass'], exp_mass_time, exp_mass_data)
                    if r2_conc is not None:
                        f.write(f"# R-squared Concentration: {r2_conc:.8f}\n")
                    if r2_mass is not None:
                        f.write(f"# R-squared Mass: {r2_mass:.8f}\n")
                    f.write("#\n")
                    f.write("# DATA:\n")

                    self._results_to_df(res).to_csv(f, index=False, lineterminator="\n")

                    if exp_conc_time is not None or exp_mass_time is not None:
                        f.write("\n# EXPERIMENTAL DATA:\n")
                        if exp_conc_time is not None:
                            f.write("#\n# Concentration Data:\n")
                            pd.DataFrame({'time (min)': exp_conc_time,
                                          'concentration (ppbv)': exp_conc_data}).to_csv(f, index=False, lineterminator="\n")
                        if exp_mass_time is not None:
                            f.write("\n# Mass Data:\n")
                            pd.DataFrame({'time (min)': exp_mass_time,
                                          'mass (ng)': exp_mass_data}).to_csv(f, index=False, lineterminator="\n")

                else:
                    # ── Sweep runs ───────────────────────────────────────────
                    # --- Sheet 1: summary table (one row per sweep point) ---
                    f.write("# SWEEP SUMMARY (one row per sweep point):\n")
                    summary_rows = []
                    for lbl, res in self.all_sweep_results:
                        row_dict = {
                            'sweep_value':         lbl,
                            'mass_bal_error (%)':  res['mass_bal_error'],
                            'solve_time (s)':      res['solve_time'],
                            'max_c_gas (ppbv)':    float(res['c_gas_ppbv'].max()),
                            'final_c_gas (ppbv)':  float(res['c_gas_ppbv'][-1]),
                            'max_delta_m (ng)':    float(res['delta_mass'].max()),
                            'final_delta_m (ng)':  float(res['delta_mass'][-1]),
                        }
                        r2c = self._compute_r_squared(
                            res['t'], res['c_gas_ppbv'], exp_conc_time, exp_conc_data)
                        r2m = self._compute_r_squared(
                            res['t'], res['delta_mass'], exp_mass_time, exp_mass_data)
                        if r2c is not None:
                            row_dict['R2_concentration'] = r2c
                        if r2m is not None:
                            row_dict['R2_mass'] = r2m
                        summary_rows.append(row_dict)
                    pd.DataFrame(summary_rows).to_csv(f, index=False, lineterminator="\n")

                    # --- Per-run data blocks ---
                    f.write("# PER-RUN DATA:\n")
                    for lbl, res in self.all_sweep_results:
                        f.write(f"# --- Sweep value: {lbl} | "
                                f"mass_bal_error: {res['mass_bal_error']:.4e}% | "
                                f"solve_time: {res['solve_time']:.3f}s ---\n")
                        self._results_to_df(res).to_csv(f, index=False, lineterminator="\n")

                    # Experimental data appended once at the end
                    if exp_conc_time is not None or exp_mass_time is not None:
                        f.write("# EXPERIMENTAL DATA:\n")
                        if exp_conc_time is not None:
                            f.write("#\n# Concentration Data:\n")
                            pd.DataFrame({'time (min)': exp_conc_time,
                                          'concentration (ppbv)': exp_conc_data}).to_csv(f, index=False, lineterminator="\n")
                        if exp_mass_time is not None:
                            f.write("\n# Mass Data:\n")
                            pd.DataFrame({'time (min)': exp_mass_time,
                                          'mass (ng)': exp_mass_data}).to_csv(f, index=False, lineterminator="\n")

            # Save plots
            plot_filename = filename.rsplit('.', 1)[0] + '_plots.png'
            self.fig.savefig(plot_filename, dpi=300, bbox_inches='tight')

            n_runs = len(self.all_sweep_results) if is_sweep else 1
            messagebox.showinfo(
                "Export Success",
                f"{'Sweep' if is_sweep else 'Simulation'} data exported ({n_runs} run(s)):\n"
                f"{filename}\n\nPlots saved to:\n{plot_filename}"
            )

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
        
    def _collect_params(self):
        """Return a dict of all current GUI parameter values (base run)."""
        tfinal = float(self.tfinal.get())
        m_str = self.m.get()
        cD_str = self.cD.get().strip().lower()
        F_str = self.F.get().strip().lower()

        return dict(
            tfinal=tfinal,
            N=int(self.N.get()),
            rtol=float(self.rtol.get()),
            atol=float(self.atol.get()),
            dt1=float(self.dt1.get()),
            T0=float(self.T0.get()),
            Tfinal=float(self.Tfinal.get()),
            RR=float(self.RR.get()),
            tEq=float(self.tEq.get()),
            tFlush=tfinal,
            Q=float(self.Q.get()),
            n_steps=int(self.n_steps.get()),
            delta=float(self.delta.get()),
            step_time=float(self.step_time.get()),
            base_conc=float(self.base_conc.get()),
            hold_time_initial=float(self.hold_time_initial.get()),
            hold_time_final=float(self.hold_time_final.get()),
            m=int(m_str.split(' ')[0]),
            R=float(self.R.get()) * 1e-4,        # µm -> cm
            R_um=float(self.R.get()),              # keep original unit for labels
            mSample=float(self.mSample.get()),
            rhoSample=float(self.rhoSample.get()),
            Vvessel=float(self.Vvessel.get()),
            MW_analyte=float(self.MW_analyte.get()),
            EaK=float(self.EaK.get()),
            K_ref=float(self.K_ref.get()),
            D_ref=float(self.D_ref.get()),
            EaD=float(self.EaD.get()),
            Tref=float(self.Tref.get()),
            c0free=float(self.c0free.get()),
            cD=np.inf if cD_str in ('inf', 'infinity') else float(cD_str),
            F=np.inf if F_str in ('inf', 'infinity') else float(F_str),
            src_params=np.array([
                val
                for sw in self.source_entries
                for val in (float(sw[1].get()), float(sw[3].get()), float(sw[5].get()))
            ]),
        )

    def _run_one(self, p):
        """Run a single simulation from a parameter dict p; return results dict."""
        temp_params = {'dt1': p['dt1'], 'T0': p['T0'],
                       'Tfinal': p['Tfinal'], 'RR': p['RR']}
        flow_params = {'tEq': p['tEq'], 'tFlush': p['tFlush'], 'Q': p['Q']}
        cfeed_params = {
            'n_steps': p['n_steps'], 'delta': p['delta'],
            'step_time': p['step_time'], 'base_conc': p['base_conc'],
            'hold_time_initial': p['hold_time_initial'],
            'hold_time_final': p['hold_time_final'],
        }
        return run_simulation(
            p['tfinal'], temp_params, flow_params, cfeed_params,
            p['m'], p['R'], p['mSample'], p['rhoSample'],
            p['Vvessel'], p['MW_analyte'],
            p['src_params'], p['EaK'], p['K_ref'], p['D_ref'],
            p['EaD'], p['cD'], p['F'], p['c0free'], 0,
            p['N'], p['rtol'], p['atol'], p.get('Tref', 50)
        )

    def _toggle_opt_ui(self):
        """Enable or disable optimization widgets based on checkbox."""
        enabled = self.opt_enabled.get()
        s = 'normal' if enabled else 'disabled'
        s_ro = 'readonly' if enabled else 'disabled'
        for w in (self.opt_n_entry, self.opt_build_btn, self.opt_run_btn,
                  self.opt_maxiter):
            w.config(state=s)
        self.opt_stop_btn.config(state='disabled')   # only active while running
        self.opt_target.config(state=s_ro)
        self.opt_method.config(state=s_ro)
        # Also enable/disable all row widgets
        for row_dict in self.opt_param_rows:
            row_dict['combo'].config(state=s_ro)
            row_dict['min_entry'].config(state=s)
            row_dict['max_entry'].config(state=s)
            row_dict['log_check'].config(state=s)

    def _build_opt_rows(self):
        """Build the dynamic optimization parameter rows, preserving existing values."""
        # Save existing row data before destroying
        old_data = []
        for rd in self.opt_param_rows:
            old_data.append({
                'param': rd['combo'].get(),
                'min': rd['min_entry'].get(),
                'max': rd['max_entry'].get(),
                'log': rd['log_var'].get(),
            })

        # Destroy existing rows
        for w in self.opt_rows_frame.winfo_children():
            w.destroy()
        self.opt_param_rows = []

        try:
            n = max(1, min(6, int(self.opt_n_entry.get())))
        except (ValueError, tk.TclError):
            n = 1

        param_names = list(self.SWEEP_PARAMS.keys())
        enabled = self.opt_enabled.get()
        s = 'normal' if enabled else 'disabled'
        s_ro = 'readonly' if enabled else 'disabled'

        for i in range(n):
            fr = ttk.Frame(self.opt_rows_frame)
            fr.pack(fill=tk.X, pady=1)

            combo = ttk.Combobox(fr, values=param_names, width=28, state=s_ro)
            # Restore previous value or pick a default
            if i < len(old_data):
                prev_param = old_data[i]['param']
                if prev_param in param_names:
                    combo.set(prev_param)
                else:
                    combo.current(min(i, len(param_names) - 1))
            else:
                combo.current(min(i, len(param_names) - 1))
            combo.pack(side=tk.LEFT, padx=1)

            min_e = ttk.Entry(fr, width=8, state=s)
            min_val = old_data[i]['min'] if i < len(old_data) else "1e-9"
            min_e.insert(0, min_val)
            min_e.pack(side=tk.LEFT, padx=1)

            max_e = ttk.Entry(fr, width=8, state=s)
            max_val = old_data[i]['max'] if i < len(old_data) else "1e-5"
            max_e.insert(0, max_val)
            max_e.pack(side=tk.LEFT, padx=1)

            if i < len(old_data):
                log_default = old_data[i]['log']
            else:
                default_param = combo.get() or param_names[min(i, len(param_names) - 1)]
                log_default = self.SWEEP_PARAMS.get(default_param, ('', True))[1]
            log_var = tk.BooleanVar(value=log_default)
            log_chk = ttk.Checkbutton(fr, variable=log_var, state=s)
            log_chk.pack(side=tk.LEFT, padx=2)

            self.opt_param_rows.append({
                'combo': combo,
                'min_entry': min_e,
                'max_entry': max_e,
                'log_var': log_var,
                'log_check': log_chk,
            })

    def _apply_opt_params(self, x_vals, p, row_list):
        """Apply optimizer trial values x_vals into a copy of params dict p."""
        p = dict(p)
        p['src_params'] = p['src_params'].copy()
        for xi, row_dict in zip(x_vals, row_list):
            label = row_dict['combo'].get()
            if label not in self.SWEEP_PARAMS:
                raise ValueError('Select a valid parameter in every optimization row.')
            attr_name, _ = self.SWEEP_PARAMS[label]
            val = xi  # already in natural space (un-log'd outside if needed)
            if attr_name == 'R':
                p['R'] = val * 1e-4
                p['R_um'] = val
            elif attr_name.startswith('src'):
                parts = attr_name.split('_')
                src_idx = int(parts[0][3:]) - 1
                field_off = {'c0': 0, 'A': 1, 'Ea': 2}[parts[1]]
                arr_idx = src_idx * 3 + field_off
                if arr_idx >= len(p['src_params']):
                    raise ValueError(
                        f"Source {src_idx + 1} does not exist in the current source configuration."
                    )
                p['src_params'][arr_idx] = val
            else:
                p[attr_name] = val
        return p

    def _terminate_optimization(self):
        """Signal the running optimizer to stop after the current evaluation."""
        self._opt_stop = True
        self.opt_stop_btn.config(state='disabled')
        self.status_label.config(text="Termination requested — finishing current evaluation...",
                                 foreground="orange")

    def run_optimization(self):
        """Run parameter optimization against experimental data."""
        exp_conc_time, exp_conc_data = self.parse_experimental_data(
            self.conc_data_text, self.conc_interp_var, self.conc_interp_n, self.conc_interp_method)
        exp_mass_time, exp_mass_data = self.parse_experimental_data(
            self.mass_data_text, self.mass_interp_var, self.mass_interp_n, self.mass_interp_method)
        target = self.opt_target.get()

        has_conc = exp_conc_time is not None
        has_mass = exp_mass_time is not None

        if target == "Concentration (ppbv)" and not has_conc:
            messagebox.showerror(
                "No Experimental Data",
                "No concentration data found in the Experimental Data panel.\n\n"
                "Please enter time, ppbv pairs (one per line) in the "
                "'Concentration (time, ppbv)' box before running optimization.")
            return
        if target == "Mass (ng)" and not has_mass:
            messagebox.showerror(
                "No Experimental Data",
                "No mass data found in the Experimental Data panel.\n\n"
                "Please enter time, ng pairs (one per line) in the "
                "'Mass (time, ng)' box before running optimization.")
            return
        if target == "Both" and (not has_conc and not has_mass):
            messagebox.showerror(
                "No Experimental Data",
                "Target is 'Both' but neither concentration nor mass data were found.\n\n"
                "Please enter experimental data in at least one box "
                "before running optimization with target = 'Both'.")
            return
        if target == "Both" and not has_conc:
            messagebox.showwarning(
                "Missing Concentration Data",
                "Target is 'Both' but no concentration data was found.\n"
                "Fitting to Mass data only.")
            target = "Mass (ng)"
        if target == "Both" and not has_mass:
            messagebox.showwarning(
                "Missing Mass Data",
                "Target is 'Both' but no mass data was found.\n"
                "Fitting to Concentration data only.")
            target = "Concentration (ppbv)"

        try:
            base = self._collect_params()
        except Exception as e:
            messagebox.showerror("Parameter Error", str(e))
            return

        row_list = self.opt_param_rows
        if not row_list:
            messagebox.showerror("Optimization Error", "No parameter rows defined.")
            return

        selected_labels = [rd['combo'].get() for rd in row_list]
        if len(selected_labels) != len(set(selected_labels)):
            messagebox.showerror(
                "Optimization Error",
                "Each optimization row must use a different parameter.")
            return

        n_sources = len(base['src_params']) // 3
        for label in selected_labels:
            if label not in self.SWEEP_PARAMS:
                messagebox.showerror(
                    "Optimization Error",
                    "Select a valid parameter in every optimization row.")
                return
            attr_name, _ = self.SWEEP_PARAMS[label]
            if attr_name.startswith('src'):
                src_idx = int(attr_name.split('_')[0][3:]) - 1
                if src_idx >= n_sources:
                    messagebox.showerror(
                        "Optimization Error",
                        f"{label} was selected, but source {src_idx + 1} does not exist.")
                    return

        bounds_search = []
        log_flags = []
        try:
            for rd in row_list:
                lo = float(rd['min_entry'].get())
                hi = float(rd['max_entry'].get())
                use_log = rd['log_var'].get()
                if not np.isfinite(lo) or not np.isfinite(hi):
                    raise ValueError('Bounds must be finite numeric values.')
                if lo >= hi:
                    raise ValueError('Each lower bound must be strictly smaller than its upper bound.')
                if use_log and (lo <= 0 or hi <= 0):
                    raise ValueError('Log-scaled optimization bounds must be positive.')
                bounds_search.append((np.log10(lo), np.log10(hi)) if use_log else (lo, hi))
                log_flags.append(use_log)
        except Exception as e:
            messagebox.showerror("Bounds Error", f"Invalid bound values:\n{e}")
            return

        try:
            maxiter = max(1, int(self.opt_maxiter.get()))
        except ValueError:
            maxiter = 200

        method = self.opt_method.get()

        self._opt_stop = False
        self.opt_run_btn.config(state='disabled')
        self.opt_stop_btn.config(state='normal')
        self._opt_iter = 0
        self._opt_best_sse = np.inf
        self._opt_best_x = None
        result = None
        terminated_early = False

        class _StopOpt(Exception):
            pass

        def objective(x_opt):
            if self._opt_stop or self._opt_iter >= maxiter:
                raise _StopOpt()
            if any((xi < lo) or (xi > hi) for xi, (lo, hi) in zip(x_opt, bounds_search)):
                return 1e30

            x_nat = np.array([10 ** xi if lg else xi for xi, lg in zip(x_opt, log_flags)], dtype=float)
            try:
                p = self._apply_opt_params(x_nat, base, row_list)
                res = self._run_one(p)
            except _StopOpt:
                raise
            except Exception:
                return 1e30

            residuals = []
            if target in ("Concentration (ppbv)", "Both") and has_conc:
                sim_interp = np.interp(exp_conc_time, res['t'], res['c_gas_ppbv'])
                norm = np.std(exp_conc_data) if np.std(exp_conc_data) > 0 else 1.0
                residuals.append(np.sum(((sim_interp - exp_conc_data) / norm) ** 2))
            if target in ("Mass (ng)", "Both") and has_mass:
                sim_interp = np.interp(exp_mass_time, res['t'], res['delta_mass'])
                norm = np.std(exp_mass_data) if np.std(exp_mass_data) > 0 else 1.0
                residuals.append(np.sum(((sim_interp - exp_mass_data) / norm) ** 2))

            self._opt_iter += 1
            sse = float(np.sum(residuals))
            if sse < self._opt_best_sse:
                self._opt_best_sse = sse
                self._opt_best_x = x_opt.copy()
            self.status_label.config(
                text=f"Optimization iter {self._opt_iter}/{maxiter}: SSE = {sse:.4g}  (best {self._opt_best_sse:.4g})",
                foreground='orange')
            self.root.update_idletasks()
            return sse

        def de_callback(xk, convergence=None):
            return self._opt_stop or self._opt_iter >= maxiter

        self.status_label.config(text="Starting optimization...", foreground='orange')
        self.root.update()

        try:
            if method == "Differential Evolution":
                n_params = len(bounds_search)
                popsize = 15
                n_gen = max(1, maxiter // max(popsize * n_params, 1))
                result = differential_evolution(
                    objective,
                    bounds=bounds_search,
                    maxiter=n_gen,
                    tol=1e-8,
                    seed=42,
                    polish=False,
                    callback=de_callback,
                    updating='deferred',
                    workers=1,
                )
                x_best = result.x
                message = str(result.message)
            elif method == "Dual Annealing":
                def da_callback(x, f, context):
                    return self._opt_stop or self._opt_iter >= maxiter

                result = dual_annealing(
                    objective,
                    bounds=bounds_search,
                    maxiter=maxiter,
                    seed=42,
                    callback=da_callback,
                    no_local_search=False,
                )
                x_best = result.x
                message = str(result.message)
            else:
                x0 = np.array([(lo + hi) / 2.0 for lo, hi in bounds_search], dtype=float)
                result = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True,
                    },
                )
                x_best = result.x
                message = str(result.message)
        except _StopOpt:
            terminated_early = True
            if self._opt_best_x is None:
                self.status_label.config(
                    text='Optimization terminated before any valid evaluations completed.',
                    foreground='blue' if self._opt_stop else 'red',
                )
                return
            x_best = self._opt_best_x.copy()
            message = f"Terminated early after {self._opt_iter} evaluations."
        except Exception as exc:
            self.status_label.config(text=f"Optimization error: {exc}", foreground='red')
            messagebox.showerror("Optimization Error", str(exc))
            return
        finally:
            self.opt_run_btn.config(state='normal')
            self.opt_stop_btn.config(state='disabled')

        if result is not None and self._opt_best_x is not None:
            result_fun = float(getattr(result, 'fun', np.inf))
            if self._opt_best_sse < result_fun:
                x_best = self._opt_best_x.copy()
        elif self._opt_best_x is not None:
            x_best = self._opt_best_x.copy()

        x_nat_best = np.array([10 ** xi if lg else xi for xi, lg in zip(x_best, log_flags)], dtype=float)

        p_best = self._apply_opt_params(x_nat_best, base, row_list)
        res_best = self._run_one(p_best)
        self.current_results = res_best
        self.all_sweep_results = []
        self.update_plots([("Best Fit", res_best)])

        self._write_opt_results_to_gui(x_nat_best, row_list)

        best_sse = self._opt_best_sse
        if not np.isfinite(best_sse) and result is not None:
            best_sse = float(getattr(result, 'fun', np.nan))
        if result is not None and (self._opt_stop or (self._opt_iter >= maxiter and not getattr(result, 'success', False))):
            terminated_early = True

        lines = [
            f"{'Terminated' if terminated_early else 'Optimization complete!'}\n",
            f"Status: {message}",
            f"Total evaluations: {self._opt_iter}",
            f"Final SSE: {best_sse:.6g}\n",
            'Best-fit parameters:',
        ]
        for rd, xv in zip(row_list, x_nat_best):
            lines.append(f"  {rd['combo'].get()}: {xv:.6g}")

        self.status_label.config(
            text=f"{'Terminated' if terminated_early else 'Complete'} — SSE={best_sse:.4g} | Evals={self._opt_iter}",
            foreground='blue' if terminated_early else 'green')
        messagebox.showinfo("Optimization Results", '\n'.join(lines))

    def _write_opt_results_to_gui(self, x_nat, row_list):
        """Write optimized parameter values back to the corresponding GUI fields."""
        for xv, rd in zip(x_nat, row_list):
            label = rd['combo'].get()
            attr_name, _ = self.SWEEP_PARAMS[label]
            val_str = f"{xv:.6g}"

            # Map attr_name to GUI widget
            widget_map = {
                'T0': self.T0, 'Tfinal': self.Tfinal, 'RR': self.RR,
                'Q': self.Q, 'mSample': self.mSample,
                'rhoSample': self.rhoSample, 'Vvessel': self.Vvessel,
                'K_ref': self.K_ref, 'EaK': self.EaK,
                'D_ref': self.D_ref, 'EaD': self.EaD,
                'cD': self.cD, 'F': self.F,
                'c0free': self.c0free, 'base_conc': self.base_conc,
            }
            if attr_name == 'R':
                self.R.delete(0, tk.END)
                self.R.insert(0, val_str)
            elif attr_name.startswith('src'):
                parts = attr_name.split('_')
                src_idx = int(parts[0][3:]) - 1
                field = parts[1]
                field_widget_idx = {'c0': 1, 'A': 3, 'Ea': 5}[field]
                if src_idx < len(self.source_entries):
                    w = self.source_entries[src_idx][field_widget_idx]
                    w.delete(0, tk.END)
                    w.insert(0, val_str)
            elif attr_name in widget_map:
                w = widget_map[attr_name]
                w.delete(0, tk.END)
                w.insert(0, val_str)

    def run_simulation(self):
        try:
            base = self._collect_params()

            if not self.sweep_enabled.get():
                # ── Single run ────────────────────────────────────────────────
                self.status_label.config(text='Running simulation...', foreground='orange')
                self.root.update()
                results = self._run_one(base)
                self.current_results = results
                self.all_sweep_results = []
                self.update_plots([('Model', results)])

                # Compute R² against experimental data if available
                exp_conc_time, exp_conc_data = self.parse_experimental_data(
                    self.conc_data_text, self.conc_interp_var, self.conc_interp_n, self.conc_interp_method)
                exp_mass_time, exp_mass_data = self.parse_experimental_data(
                    self.mass_data_text, self.mass_interp_var, self.mass_interp_n, self.mass_interp_method)
                r2_conc = self._compute_r_squared(results['t'], results['c_gas_ppbv'],
                                                  exp_conc_time, exp_conc_data)
                r2_mass = self._compute_r_squared(results['t'], results['delta_mass'],
                                                  exp_mass_time, exp_mass_data)
                r2_parts = []
                if r2_conc is not None:
                    r2_parts.append(f"R²(conc)={r2_conc:.6f}")
                if r2_mass is not None:
                    r2_parts.append(f"R²(mass)={r2_mass:.6f}")
                r2_str = (" | " + " | ".join(r2_parts)) if r2_parts else ""

                self.status_label.config(
                    text=(f"Complete! Solve time: {results['solve_time']:.2f}s | "
                          f"Mass balance error: {results['mass_bal_error']:.2e}%{r2_str}"),
                    foreground='green')

            else:
                # ── Sweep run ─────────────────────────────────────────────────
                param_label = self.sweep_param_var.get()
                if param_label not in self.SWEEP_PARAMS:
                    raise ValueError('Select a valid sweep parameter.')

                attr_name, _ = self.SWEEP_PARAMS[param_label]
                start_val = float(self.sweep_start.get())
                stop_val = float(self.sweep_stop.get())
                n_pts = int(self.sweep_n.get())
                if n_pts < 2:
                    raise ValueError('Sweep must use at least 2 points.')
                log_space = self.sweep_log.get()
                if log_space and (start_val <= 0 or stop_val <= 0):
                    raise ValueError('Log-spaced sweeps require positive start and stop values.')

                if attr_name.startswith('src'):
                    src_idx = int(attr_name.split('_')[0][3:]) - 1
                    if src_idx >= len(base['src_params']) // 3:
                        raise ValueError(
                            f"Source {src_idx + 1} does not exist in the current source configuration. "
                            "Add more sources before sweeping this parameter."
                        )

                if log_space:
                    sweep_vals = np.logspace(np.log10(start_val), np.log10(stop_val), n_pts)
                else:
                    sweep_vals = np.linspace(start_val, stop_val, n_pts)

                cmap = plt.get_cmap(self.sweep_cmap.get())
                colors = [cmap(i / max(n_pts - 1, 1)) for i in range(n_pts)]

                labeled_results = []
                total_t0 = time.time()
                for idx, val in enumerate(sweep_vals):
                    self.status_label.config(
                        text=f"Sweep {idx + 1}/{n_pts}: {attr_name} = {val:.4g}",
                        foreground='orange')
                    self.root.update()

                    p = dict(base)
                    p['src_params'] = base['src_params'].copy()

                    if attr_name == 'R':
                        p['R'] = val * 1e-4
                        p['R_um'] = val
                    elif attr_name.startswith('src'):
                        parts = attr_name.split('_')
                        src_idx = int(parts[0][3:]) - 1
                        field = parts[1]
                        field_off = {'c0': 0, 'A': 1, 'Ea': 2}[field]
                        arr_idx = src_idx * 3 + field_off
                        if arr_idx >= len(p['src_params']):
                            raise ValueError(
                                f"Source {src_idx + 1} does not exist in the current source configuration. "
                                "Add more sources before sweeping this parameter."
                            )
                        p['src_params'][arr_idx] = val
                    else:
                        p[attr_name] = val

                    res = self._run_one(p)
                    res['_sweep_color'] = colors[idx]
                    lbl = f"{val:.2e}" if log_space else f"{val:.4g}"
                    labeled_results.append((lbl, res))

                elapsed_total = time.time() - total_t0
                self.current_results = labeled_results[-1][1]
                self.all_sweep_results = labeled_results
                self.update_plots(labeled_results, sweep_label=param_label)
                self.status_label.config(
                    text=f"Sweep complete! {n_pts} runs in {elapsed_total:.1f}s",
                    foreground='green')

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground='red')
            messagebox.showerror('Simulation Error', str(e))

    def update_plots(self, labeled_results, sweep_label=None):
        """
        Plot one or more simulation results.
        labeled_results : list of (label_str, results_dict)
        sweep_label     : human-readable name of the swept parameter (or None)
        """
        for ax in [self.ax1, self.ax2, self.ax3,
                   self.ax4, self.ax5, self.ax6, self.ax7]:
            ax.clear()

        is_sweep = len(labeled_results) > 1

        # Experimental data (shown only once, on top)
        exp_conc_time, exp_conc_data = self.parse_experimental_data(
            self.conc_data_text, self.conc_interp_var, self.conc_interp_n, self.conc_interp_method)
        exp_mass_time, exp_mass_data = self.parse_experimental_data(
            self.mass_data_text, self.mass_interp_var, self.mass_interp_n, self.mass_interp_method)

        # Colour / linewidth logic
        def _colour(res, default):
            return res.get('_sweep_color', default)

        lw_main = 1.5 if is_sweep else 2.0

        for label, res in labeled_results:
            t    = res['t']
            col  = _colour(res, None)   # None → use matplotlib default cycle
            kw   = dict(linewidth=lw_main, label=label, color=col) if col else dict(linewidth=lw_main, label=label)

            self.ax1.plot(t, res['T_C'],          **kw)
            self.ax2.plot(t, res['c_feed_ppbv'],  **kw)
            self.ax3.plot(t, res['Q'],             **kw)
            self.ax4.semilogy(t, res['D'],   **kw)
            self.ax5.plot(t, res['S'],             **kw)
            self.ax6.plot(t, res['delta_mass'],    **kw)
            self.ax7.plot(t, res['c_gas_ppbv'],    **kw)

        # ── Axis labels / titles ──────────────────────────────────────────────
        for ax, xlabel, ylabel, title in [
            (self.ax1, 'Time (min)', 'Temperature (°C)',           'Temperature'),
            (self.ax2, 'Time (min)', 'Feed Conc (ppbv)',           'Feed Concentration'),
            (self.ax3, 'Time (min)', 'Flow Rate (ml/min)',         'Flow Rate'),
            (self.ax4, 'Time (min)', 'Diffusivity (cm²/s)',        'Diffusivity at Surface'),
            (self.ax5, 'Time (min)', 'S (cm³@STP/atm/cm³)',        'Solubility'),
            (self.ax6, 'Time (min)', 'Δm (ng)',                    'Sample Mass Change'),
            (self.ax7, 'Time (min)', 'Concentration (ppbv)',       'Headspace Gas Concentration'),
        ]:
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)

        # Log-axis clean-up for diffusivity
        self.ax4.grid(True, which="both", ls="-", alpha=0.3)
        self.ax4.set_yscale('log')
        self.ax4.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
        self.ax4.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
        all_D = np.concatenate([r['D'] for _, r in labeled_results])
        positive_D = all_D[np.isfinite(all_D) & (all_D > 0)]
        if positive_D.size:
            ymin, ymax = float(positive_D.min()), float(positive_D.max())
            if ymax / ymin < 10:
                mid = (ymin * ymax) ** 0.5
                self.ax4.set_ylim(mid / 10, mid * 10)

        # Horizontal zero line on mass plot
        self.ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)

        # Experimental data overlays
        if exp_mass_time is not None:
            self.ax6.plot(exp_mass_time, exp_mass_data, 'ko',
                          markersize=5, label='Experimental', zorder=10)
        if exp_conc_time is not None:
            self.ax7.plot(exp_conc_time, exp_conc_data, 'ko',
                          markersize=5, label='Experimental', zorder=10)

        # R² annotations (single-run only)
        if not is_sweep and len(labeled_results) == 1:
            _, res0 = labeled_results[0]
            r2_conc = self._compute_r_squared(
                res0['t'], res0['c_gas_ppbv'], exp_conc_time, exp_conc_data)
            r2_mass = self._compute_r_squared(
                res0['t'], res0['delta_mass'], exp_mass_time, exp_mass_data)
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8)
            if r2_conc is not None:
                self.ax7.text(0.02, 0.97, f'R² = {r2_conc:.6f}',
                              transform=self.ax7.transAxes, fontsize=10,
                              verticalalignment='top', bbox=bbox_props)
            if r2_mass is not None:
                self.ax6.text(0.02, 0.97, f'R² = {r2_mass:.6f}',
                              transform=self.ax6.transAxes, fontsize=10,
                              verticalalignment='top', bbox=bbox_props)

        # ── Legends ───────────────────────────────────────────────────────────
        if is_sweep:
            # Compact legend on gas-concentration and mass panels
            short_label = (sweep_label or "").split('–')[-1].strip()
            for ax in (self.ax6, self.ax7):
                leg = ax.legend(
                    title=short_label,
                    fontsize=8, title_fontsize=8,
                    loc='best', framealpha=0.8,
                    ncol=1 if len(labeled_results) <= 8 else 2)
                leg.get_frame().set_linewidth(0.5)
        else:
            # Single run: show legend only where experimental data exists
            if exp_mass_time is not None:
                self.ax6.legend(fontsize=10, loc='best')
            if exp_conc_time is not None:
                self.ax7.legend(fontsize=10, loc='best')

        self.canvas.draw()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = OutgassingGUI(root)
    root.mainloop()