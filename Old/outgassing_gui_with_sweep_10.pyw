# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Lawrence Livermore National Security, LLC
# LLNL-CODE-2017385
# CP 2025-187
# Author: Steven A. Hawks

"""Tkinter GUI for the MOLIERE outgassing model."""

import json
import os
import threading
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
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix
from scipy.stats import qmc

LLNL_RELEASE_ID = "LLNL-CODE-2017385"
APP_TITLE = "Outgassing Model Simulator"
MAX_UI_SOURCES = 20

# Selectable units for the headspace gas concentration (plotting / experimental
# input).  Volumetric mixing ratios (ppbv, ppmv) depend on gas temperature;
# mass concentrations (µg/m³, mg/m³) do not.  Conversions go through the model's
# fundamental molar concentration c_gas (µM); see OutgassingGUI._conc_from_uM.
CONC_UNITS = ['ppbv', 'ppmv', 'µg/m³', 'mg/m³']

# Sample mass-change display/input units.  The model computes the cumulative
# mass change in nanograms (ng); µg and mg are simple decimal rescalings
# (1 µg = 1e3 ng, 1 mg = 1e6 ng).  See OutgassingGUI._mass_from_ng.
MASS_UNITS = ['ng', 'µg', 'mg']

# ============================================================================
# Core simulation functions (MOLIERE v11 -- conservative Kirchhoff-flux
# finite-volume scheme; exactly mass-conservative for any cD)
# ============================================================================


def temperature_vec(t, dt1, T0, T_target, RR):
    """Vectorized temperature profile -> (T_celsius, dTdt)."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    T = np.empty_like(t)
    dTdt = np.zeros_like(t)

    delta_T = T_target - T0
    if np.isclose(delta_T, 0.0):
        T.fill(T0)
    else:
        if RR == 0:
            raise ValueError('RR must be nonzero when T0 and Tfinal differ.')
        RR_eff = np.sign(delta_T) * abs(RR)
        t_ramp_end = dt1 + abs(delta_T) / abs(RR)
        m1 = t <= dt1
        m3 = t > t_ramp_end
        m2 = ~m1 & ~m3
        T[m1] = T0
        T[m2] = T0 + RR_eff * (t[m2] - dt1)
        dTdt[m2] = RR_eff
        T[m3] = T_target

    if scalar:
        return float(T[0]), float(dTdt[0])
    return T, dTdt


def flow_rate_vec(t, tEq, tFlush, Q_val):
    """Vectorized flow rate → Q array."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    Q = np.where(t % (tEq + tFlush) < tEq, 0.0, Q_val)
    return float(Q[0]) if scalar else Q


def feed_conc_vec(t, n_steps, delta, step_time, base_conc,
                  hold_time_initial, hold_time_final):
    """Vectorized feed concentration → ppbv array."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    total_time = hold_time_initial + 2 * n_steps * step_time + hold_time_final
    t_cycle = t % total_time
    result = np.full_like(t, base_conc)
    active = (t_cycle >= hold_time_initial) & (t_cycle < total_time - hold_time_final)
    if np.any(active):
        t_act = t_cycle[active] - hold_time_initial
        cs = (t_act / step_time).astype(int)
        rising = cs < n_steps
        result[active] = np.where(rising, base_conc + (cs + 1) * delta,
                                  base_conc + (2 * n_steps - cs - 1) * delta)
    return float(result[0]) if scalar else result


def feed_tank_segments(tfinal, Q_val, V_feed, n_steps, delta, step_time,
                       base_conc, hold_time_initial, hold_time_final):
    """Precompute the first-order (CSTR) roll-over of the staircase feed.

    A finite upstream "feed tank" of volume ``V_feed`` (ml) purged at the
    carrier flow ``Q_val`` (ml/min) smooths every step change in the ideal
    staircase ``feed_conc_vec`` with time constant tau = V_feed / Q_val.  The
    tank-outlet concentration obeys the linear lag

        dy_out/dt = (Q_val / V_feed) * (y_ideal(t) - y_out(t)),

    which, because ``y_ideal`` is piecewise constant, has an exact
    segment-by-segment exponential solution.  For a single absorption step
    (0 -> y_max) this reduces to Kumar eq. 16, y_out = y_max*(1 - e^{-Qt/V}),
    and for the following desorption step (y_max -> 0) to eq. 17,
    y_out = y_max*e^{-Qt/V}.

    Returns ``(seg_left, level, y_start, rate)`` arrays for fast lookup by
    :func:`feed_tank_eval`, or ``None`` when no roll-over applies (V_feed <= 0,
    Q_val <= 0, or a flat feed with n_steps == 0).  When ``None`` is returned
    the caller uses the un-rolled square-wave :func:`feed_conc_vec`.
    """
    if V_feed <= 0 or Q_val <= 0 or n_steps <= 0:
        return None
    total = hold_time_initial + 2 * n_steps * step_time + hold_time_final
    if total <= 0:
        return None
    # Breakpoints: every instant the staircase can change value, over [0, tfinal].
    edges = [0.0, float(tfinal)]
    p = 0
    while p * total < tfinal:
        b = p * total
        edges.append(b)
        for j in range(0, 2 * n_steps + 1):
            edges.append(b + hold_time_initial + j * step_time)
        edges.append(b + total)
        p += 1
    edges = np.unique(np.clip(np.asarray(edges, dtype=float), 0.0, float(tfinal)))
    # Staircase level on each segment (sampled at the segment midpoint, so the
    # exact rising/falling/hold logic of feed_conc_vec is reused verbatim).
    mids = 0.5 * (edges[:-1] + edges[1:])
    level = np.atleast_1d(feed_conc_vec(mids, n_steps, delta, step_time,
                                        base_conc, hold_time_initial, hold_time_final))
    rate = Q_val / V_feed
    seg_left = edges[:-1]
    y_start = np.empty_like(level)
    y = float(base_conc)                      # tank starts at the base concentration
    for k in range(len(level)):
        y_start[k] = y
        y = level[k] + (y - level[k]) * np.exp(-rate * (edges[k + 1] - edges[k]))
    return seg_left, level, y_start, rate


def feed_tank_eval(t, seg_left, level, y_start, rate):
    """Evaluate the rolled-over feed (ppbv) from a :func:`feed_tank_segments`
    table at scalar or array ``t``."""
    t = np.asarray(t, dtype=float)
    scalar = t.ndim == 0
    t = np.atleast_1d(t)
    idx = np.searchsorted(seg_left, t, side='right') - 1
    idx = np.clip(idx, 0, len(level) - 1)
    out = level[idx] + (y_start[idx] - level[idx]) * np.exp(-rate * (t - seg_left[idx]))
    return float(out[0]) if scalar else out


GRID_SCHEMES = ('uniform', 'tanh', 'geometric')


def make_grid(R, N, scheme='uniform', stretch=2.0):
    """
    Generate the spatial grid x[0..N-1] on [0, R].

    For outgassing/desorption problems the concentration gradient is steepest
    at the sample surface (x = R) at early times, while the profile is flat at
    the symmetry point (x = 0).  A boundary-refined grid therefore places
    nodes densely near x = R and sparsely near x = 0.

    Schemes
    -------
    'uniform'    : x_i = R * i/(N-1).  (Default; reproduces previous behavior.)
    'tanh'       : Roberts-type one-sided stretching,
                   x = R * tanh(stretch * xi) / tanh(stretch),  xi in [0, 1].
                   `stretch` = beta > 0 controls clustering at x = R; the
                   coarse-to-fine spacing ratio is h_max/h_min = cosh^2(beta)
                   (beta = 1.5 -> ~5.5x, beta = 2 -> ~14x, beta = 2.5 -> ~38x).
                   Recommended non-uniform scheme (smooth mapping, ~2nd-order
                   stencil behavior retained).
    'geometric'  : spacings form a geometric progression, refined toward
                   x = R; `stretch` = h_max/h_min (> 1) is the total
                   coarse-to-fine spacing ratio.

    Returns
    -------
    ndarray of shape (N,), strictly increasing, with x[0] = 0 and x[-1] = R.
    """
    xi = np.linspace(0.0, 1.0, N)
    if scheme == 'uniform':
        x = R * xi
    elif scheme == 'tanh':
        beta = float(stretch)
        if beta <= 0:
            raise ValueError("Grid Stretch (beta) must be > 0 for the 'tanh' scheme.")
        x = R * np.tanh(beta * xi) / np.tanh(beta)
    elif scheme == 'geometric':
        s = float(stretch)
        if s <= 0:
            raise ValueError("Grid Stretch must be > 0 for the 'geometric' scheme.")
        if np.isclose(s, 1.0):
            x = R * xi
        else:
            q = s ** (-1.0 / (N - 2))           # per-interval ratio (< 1 refines toward R)
            hs = q ** np.arange(N - 1)
            hs *= R / hs.sum()
            x = np.concatenate(([0.0], np.cumsum(hs)))
    else:
        raise ValueError("Grid Scheme must be 'uniform', 'tanh', or 'geometric'.")
    x[0] = 0.0
    x[-1] = R          # enforce exact endpoints against floating-point drift
    if np.any(np.diff(x) <= 0):
        raise ValueError('Generated grid is not strictly increasing.')
    return x

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
    grid_scheme='uniform',
    grid_stretch=2.0,
):
    if int(N) < 3:
        raise ValueError("Grid Points (N) must be at least 3.")
    if grid_scheme not in GRID_SCHEMES:
        raise ValueError("Grid Scheme must be one of: " + ", ".join(GRID_SCHEMES) + ".")
    if grid_scheme in ('tanh', 'geometric'):
        if not np.isfinite(grid_stretch) or grid_stretch <= 0:
            raise ValueError("Grid Stretch must be a finite value > 0 for the "
                             f"'{grid_scheme}' scheme.")
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
        'feed_tank_volume': cfeed_params.get('feed_tank_volume', 0.0),
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
    if cfeed_params.get('feed_tank_volume', 0.0) < 0:
        raise ValueError("Feed Tank Volume must be >= 0.")
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

# ============================================================================
# Sparse Jacobian sparsity pattern
# ============================================================================

def generate_jacobian_sparsity(N, num_srcs, is_inf_F):
    """Jacobian sparsity for the conservative (Kirchhoff-flux) scheme.

    Tridiagonal solid block from the two-point face fluxes.  For F = inf the
    surface row couples to (N-2, N-1) and the source columns feed every solid
    row (generation inside the surface cell is bookkept); for finite F the
    headspace ODE adds one row/column coupled to the surface node.
    """
    total = N + num_srcs if is_inf_F else N + 1 + num_srcs
    sp = lil_matrix((total, total), dtype=np.int8)
    for i in range(N):
        sp[i, i] = 1
    for i in range(1, N - 1):
        sp[i, i - 1] = 1
        sp[i, i + 1] = 1
    if N > 1:
        sp[0, 1] = 1
        sp[N - 1, N - 2] = 1
    if is_inf_F:
        for si in range(num_srcs):
            sp[N + si, N + si] = 1
            sp[:N, N + si] = 1
    else:
        sp[N - 1, N] = 1
        sp[N, N - 1] = 1
        sp[N, N] = 1
        for si in range(num_srcs):
            row = N + 1 + si
            sp[row, row] = 1
            sp[:N, row] = 1
    return csc_matrix(sp)

# ============================================================================
# Conservative (Kirchhoff-flux) finite-volume discretization
# ============================================================================
#
# The diffusion term is discretized in divergence form,
#
#     dc/dt = (1/x^m) d/dx [ x^m * (-q) ],     q = -D dc/dx = -dpsi/dx,
#
# using the Kirchhoff potential  psi(c) = INT_0^c D(c',T) dc'.  For the
# exponential law  D = D0 exp(c/cD - Ea,D/RgasT)  psi has the closed form
#
#     psi_{i+1} - psi_i = cD (D_{i+1} - D_i) = cD D_i expm1(dc/cD),
#
# so the exact two-point flux between adjacent nodes is available in closed
# form (the expm1 form is numerically stable when |dc|/cD is small).  As
# cD -> inf this reduces to the constant-D flux  -D (c_{i+1}-c_i)/dx.  Each
# node i owns the control volume between the midpoint faces x_{i-1/2},
# x_{i+1/2} (half cells at x = 0 and x = R), with face areas x_f^m and
# volumes V_i = (x_r^{m+1}-x_l^{m+1})/(m+1).  Interior fluxes telescope, so
# the discrete solid inventory  sum_i V_i c_i  changes only through the
# surface flux and the source terms: mass is conserved to ODE-solver
# tolerance for ANY cD, on uniform and non-uniform grids alike.
#
# Boundary treatment:
#   * F = inf : the surface cell and the headspace form a single equilibrium
#     reservoir (c_gas = c_R / K).  Their combined mass balance gives
#         [gam*V_{N-1} + Vhs/K] dc_R/dt = gam*Fl_{N-3/2}
#                                         - Q (c_R/K - c_feed)
#                                         + c_R (Vhs/K) (1/K)(dK/dt)
#                                         + gam*V_{N-1} * src_total,
#     which conserves (solid + headspace) mass exactly, including the dK/dt
#     repartitioning during temperature ramps.
#   * finite F : the Robin flux  q_R = F (c_R - K c_gas)  is applied at the
#     outer face of the surface cell, and the identical flux feeds the
#     headspace ODE, Eqn (5) -- exact solid/gas mass exchange by construction.


def fv_geometry(x, m_geom):
    """Finite-volume geometry on the (possibly non-uniform) grid x.

    Node i owns the control volume between the midpoint faces; the first and
    last cells are half cells touching x = 0 and x = R.

    Returns
    -------
    xf : ndarray, shape (N-1,) -- interior face positions (midpoints)
    Af : ndarray, shape (N-1,) -- face area factors x_f^m
    V  : ndarray, shape (N,)   -- cell volumes INT x^m dx over each cell;
                                  sum(V) = R^(m+1)/(m+1) exactly (telescoping)
    """
    xf = 0.5 * (x[:-1] + x[1:])
    xl = np.concatenate(([x[0]], xf))
    xr = np.concatenate((xf, [x[-1]]))
    mp1 = m_geom + 1
    V = (xr ** mp1 - xl ** mp1) / mp1
    Af = xf ** m_geom if m_geom != 0 else np.ones_like(xf)
    return xf, Af, V


def make_ode_system(N, x, R, cD, beta, dt1, T0, Tfinal, RR, tEq,
                                 tFlush, Q_val, n_steps, delta, step_time,
                                 base_conc, hold_time_initial, hold_time_final,
                                 src_params, Vheadspace, F, Rgas, D0, EaD, K0,
                                 EaK, m_geom, feed_tank_table=None):
    """Build the specialized ODE right-hand side; returns (rhs_func, solver_kwargs).

    An analytical sparse Jacobian is supplied for the most common case
    (F = inf, cD = inf); concentration-dependent cases use the numerical
    Jacobian with the conservative sparsity pattern.
    """
    num_srcs = len(src_params) // 3
    is_inf_F = np.isinf(F)
    is_inf_cD = np.isinf(cD)

    # ---- Finite-volume geometry (uniform or non-uniform grid) ----
    mp1 = m_geom + 1
    xf, Af, V = fv_geometry(x, m_geom)
    dx = np.diff(x)
    Af_dx = Af / dx                       # face area / node spacing, (N-1,)
    invV = 1.0 / V
    vSample = Vheadspace / beta
    gam = vSample * mp1 / R ** mp1        # ml per unit of INT c x^m dx
    gamV_last = gam * V[-1]               # physical volume of surface cell (ml)

    # Conservative constant-D operator, pre-combined (tridiagonal):
    #   dc_i/dt = D * (Csub*c_{i-1} + Cdia*c_i + Csup*c_{i+1}),  i = 1..N-2
    Csub = Af_dx[:-1] * invV[1:N - 1]
    Csup = Af_dx[1:] * invV[1:N - 1]
    Cdia = -(Csub + Csup)
    orgC = Af_dx[0] * invV[0]             # origin: dc/dt|_0 = orgC * D * (c1 - c0)

    inv_R_beta = 1.0 / (R * beta)
    inv_Vhs = 1.0 / Vheadspace
    neg_EaD_Rgas = -EaD / Rgas
    # GUI convention: EaK is the *signed* sorption enthalpy (van 't Hoff),
    # so K = K0*exp(-EaK/Rgas/T) decreases on heating for EaK < 0.
    EaK_Rgas = -EaK / Rgas
    delta_T = Tfinal - T0
    if np.isclose(delta_T, 0.0):
        RR_eff = 0.0
        t_ramp_end = dt1
    else:
        RR_eff = np.sign(delta_T) * abs(RR)
        t_ramp_end = dt1 + abs(delta_T) / abs(RR)
    period = tEq + tFlush
    total_feed_time = hold_time_initial + 2 * n_steps * step_time + hold_time_final
    inv_cD = 0.0 if is_inf_cD else 1.0 / cD

    if num_srcs > 0:
        src_A = src_params[1::3].copy()
        src_Ea = src_params[2::3].copy()
    else:
        src_A = np.empty(0)
        src_Ea = np.empty(0)

    n_total = (N + num_srcs) if is_inf_F else (N + 1 + num_srcs)

    # ---- Inline scalar helpers ----
    def _temp(t):
        if t <= dt1:
            return T0, 0.0
        if t <= t_ramp_end:
            return T0 + RR_eff * (t - dt1), RR_eff
        return Tfinal, 0.0

    def _flow(t):
        return 0.0 if (t % period) < tEq else Q_val

    def _feed(t):
        if feed_tank_table is not None:
            return feed_tank_eval(t, *feed_tank_table)
        if n_steps == 0: return base_conc
        tc = t % total_feed_time
        if tc < hold_time_initial or tc >= total_feed_time - hold_time_final:
            return base_conc
        cs = int((tc - hold_time_initial) / step_time)
        return base_conc + (cs + 1) * delta if cs < n_steps else base_conc + (2 * n_steps - cs - 1) * delta

    def _dpsi(cs, inv_T):
        """Kirchhoff potential differences psi_{i+1} - psi_i at all faces."""
        if is_inf_cD:
            D_T = D0 * np.exp(neg_EaD_Rgas * inv_T)
            return D_T * np.diff(cs)
        Dn = D0 * np.exp(cs * inv_cD + neg_EaD_Rgas * inv_T)
        return cD * Dn[:-1] * np.expm1(np.diff(cs) * inv_cD)

    # ==================================================================
    # CASE 1: isinf(F)  --  equilibrium surface (combined cell/headspace)
    # ==================================================================
    if is_inf_F:
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
            Fl = -Af_dx * _dpsi(cs, inv_T)   # area-weighted face fluxes (+x)

            src_total = 0.0
            if num_srcs > 0:
                rates = src_A * np.exp(-src_Ea / (Rgas * T)) * c[src_off:src_off + num_srcs]
                src_total = float(np.sum(rates))

            dcdt = np.empty(n_total)
            dcdt[0] = -Fl[0] * invV[0] + src_total
            dcdt[1:N - 1] = (Fl[:-1] - Fl[1:]) * invV[1:N - 1] + src_total

            # Combined surface-cell + headspace balance (c_gas = c_R / K)
            VhK = Vheadspace / K
            dcdt[N - 1] = (gam * Fl[-1]
                           - Q * (cs[-1] / K - cfeed)
                           + cs[-1] * VhK * dKdT_K
                           + gamV_last * src_total) / (gamV_last + VhK)

            if num_srcs > 0:
                dcdt[src_off:src_off + num_srcs] = -rates
            return dcdt

        if not is_inf_cD:
            sparsity = generate_jacobian_sparsity(N, num_srcs, True)
            return rhs, dict(jac_sparsity=sparsity)

        # ---- Analytical Jacobian for the constant-D case ----
        idx_int = np.arange(1, N - 1)
        jac_rows = np.concatenate([
            [0, 0],                                          # center
            idx_int, idx_int, idx_int,                        # interior sub/diag/super
            [N - 1, N - 1],                                  # surface (2-pt flux)
            np.arange(src_off, src_off + num_srcs),          # source self
            np.tile(np.arange(N), num_srcs),                 # source -> all solid rows
        ]).astype(np.int32)
        jac_cols = np.concatenate([
            [0, 1],
            idx_int - 1, idx_int, idx_int + 1,
            [N - 2, N - 1],
            np.arange(src_off, src_off + num_srcs),
            np.repeat(np.arange(src_off, src_off + num_srcs), N),
        ]).astype(np.int32)
        n_entries = len(jac_rows)
        _jac_vals = np.empty(n_entries)

        def jac(t, c):
            TC, dTdt_val = _temp(t)
            T = TC + 273.15
            inv_T = 1.0 / T
            Q = _flow(t)

            K = K0 * np.exp(EaK_Rgas * inv_T)
            dKdT_K = -EaK_Rgas * dTdt_val * inv_T * inv_T
            D = D0 * np.exp(neg_EaD_Rgas * inv_T)
            VhK = Vheadspace / K
            inv_denomS = 1.0 / (gamV_last + VhK)

            vals = _jac_vals
            v = orgC * D
            vals[0] = -v
            vals[1] = v

            base = 2
            n_int = N - 2
            vals[base:base + n_int] = D * Csub
            vals[base + n_int:base + 2 * n_int] = D * Cdia
            vals[base + 2 * n_int:base + 3 * n_int] = D * Csup

            base_bnd = base + 3 * n_int
            a = gam * Af_dx[-1] * D * inv_denomS
            vals[base_bnd] = a                                              # d/dc_{N-2}
            vals[base_bnd + 1] = -a + (-Q / K + VhK * dKdT_K) * inv_denomS  # d/dc_{N-1}

            base_src = base_bnd + 2
            if num_srcs > 0:
                k_src = src_A * np.exp(-src_Ea / (Rgas * T))
                vals[base_src:base_src + num_srcs] = -k_src
                base_coup = base_src + num_srcs
                w_last = gamV_last * inv_denomS
                for si in range(num_srcs):
                    blk = vals[base_coup + si * N:base_coup + (si + 1) * N]
                    blk[:N - 1] = k_src[si]
                    blk[N - 1] = w_last * k_src[si]

            return csc_matrix(coo_matrix(
                (vals.copy(), (jac_rows, jac_cols)), shape=(n_total, n_total)))

        return rhs, dict(jac=jac)

    # ==================================================================
    # CASE 2: finite F  --  Robin flux applied at the outer face
    # ==================================================================
    else:
        src_off = N + 1
        Rm = R ** m_geom

        def rhs(t, c):
            TC, _ = _temp(t)
            T = TC + 273.15
            inv_T = 1.0 / T
            Q = _flow(t)
            cfeed = _feed(t) * 0.001 / 22.414 * 273.15 * inv_T

            K = K0 * np.exp(EaK_Rgas * inv_T)
            cs = c[:N]
            cR = cs[-1]
            c_gas = c[N]

            Fl = -Af_dx * _dpsi(cs, inv_T)
            qR = 60.0 * F * (cR - c_gas * K)     # outward surface flux (Eqn 4)

            src_total = 0.0
            if num_srcs > 0:
                rates = src_A * np.exp(-src_Ea / (Rgas * T)) * c[src_off:src_off + num_srcs]
                src_total = float(np.sum(rates))

            dcdt = np.empty(n_total)
            dcdt[0] = -Fl[0] * invV[0] + src_total
            dcdt[1:N - 1] = (Fl[:-1] - Fl[1:]) * invV[1:N - 1] + src_total
            dcdt[N - 1] = (Fl[-1] - Rm * qR) * invV[N - 1] + src_total
            dcdt[N] = (mp1 * 60.0 * F * (cR - c_gas * K) * inv_R_beta
                       + Q * inv_Vhs * (cfeed - c_gas))

            if num_srcs > 0:
                dcdt[src_off:src_off + num_srcs] = -rates
            return dcdt

        sparsity = generate_jacobian_sparsity(N, num_srcs, False)
        return rhs, dict(jac_sparsity=sparsity)

# ============================================================================
# Main simulation
# ============================================================================

def run_simulation(tfinal, temp_params, flow_params, cfeed_params,
                   m, R, mSample, rhoSample, Vvessel, MW_analyte,
                   src_params, EaK, K_ref, D_ref, EaD, cD, F, c0free, cgas_init, N,
                   rtol, atol, Tref=50, grid_scheme='uniform', grid_stretch=2.0):
    """Run the outgassing simulation (conservative Kirchhoff-flux scheme).

    The diffusion term is discretized with the conservative finite-volume
    (Kirchhoff-flux) scheme: exactly mass-conservative for any cD, on uniform
    and non-uniform grids, for all geometries and boundary types.

    Parameter conventions (GUI):
      EaK : signed sorption enthalpy (kJ/mol); K(T) = K_ref *
            exp[(EaK/Rgas)(1/Tref_K - 1/T_K)] decreases on heating for EaK < 0.
      EaD : Arrhenius activation energy (kJ/mol, > 0 for normal diffusion);
            D(T, c) = D_ref * exp[(EaD/Rgas)(1/Tref_K - 1/T_K)] * exp(c/cD).
      rtol, atol : BDF solver tolerances (exposed in the GUI).
    """
    src_params = _as_float_array(src_params, 'src_params')
    _validate_simulation_inputs(
        tfinal, temp_params, flow_params, cfeed_params,
        m, R, mSample, rhoSample, Vvessel, MW_analyte,
        src_params, EaK, K_ref, D_ref, EaD, cD, F,
        c0free, cgas_init, N, rtol, atol, Tref,
        grid_scheme, grid_stretch,
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

    x = make_grid(R, N, scheme=grid_scheme, stretch=grid_stretch)
    K_t0 = K0 * np.exp(-EaK / Rgas / (temp_params['T0'] + 273.15))

    # Finite-volume geometry; the cell volumes double as the exact mass quadrature
    _, _, Vfv = fv_geometry(x, m)
    gam = vSample * (m + 1) / R ** (m + 1)

    # Initial conditions (dtype=float guards against silent integer
    # truncation of the boundary value when c0free is passed as an int)
    c_init = np.full(N, c0free, dtype=float)
    if is_inf_F:
        # Mass-consistent IC for the combined surface-cell/headspace state:
        # the surface cell equilibrates with the headspace at t = 0+, so the
        # combined inventory gam*V[-1]*c0free + Vheadspace*cgas_init is
        # preserved exactly.  Reduces to cgas_init*K_t0 as the cell shrinks,
        # and to c0free when the system starts pre-equilibrated.
        c_init[-1] = ((gam * Vfv[-1] * c0free + Vheadspace * cgas_init)
                      / (gam * Vfv[-1] + Vheadspace / K_t0))
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
    V_feed = cfeed_params.get('feed_tank_volume', 0.0)

    # Optional finite feed-tank volume rolls the square-wave feed over with
    # time constant tau = V_feed / Q (Kumar eqs. 16-17).  V_feed = 0 -> the
    # ideal staircase is used unchanged.  The same table drives both the ODE
    # right-hand side and the reported feed/mass-balance curves below.
    feed_tank_table = feed_tank_segments(tfinal, Q_val, V_feed,
                                         ns, dl, st, bc, hti, htf)

    rhs, solver_kw = make_ode_system(
        N, x, R, cD, beta, dt1, T0, Tfinal, RR, tEq, tFlush, Q_val,
        ns, dl, st, bc, hti, htf, src_params, Vheadspace, F, Rgas, D0, EaD, K0, EaK, m,
        feed_tank_table=feed_tank_table)

    t0 = time.time()
    try:
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
    except Exception as e:
        raise RuntimeError(f"SciPy solve_ivp failed: {e}") from e
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
    DR_cm2s = DR / 60.0  # convert from cm^2/min (internal) to cm^2/s (output)

    c_feed_ppbv = (feed_tank_eval(t, *feed_tank_table) if feed_tank_table is not None
                   else feed_conc_vec(t, ns, dl, st, bc, hti, htf))
    c_feed_uM = c_feed_ppbv / 1000 / 22.414 * 273.15 / T_K

    c_gas = cR / K if is_inf_F else y[:, N]
    cGasPPBV = c_gas * 1000 * 22.414 / 273.15 * T_K

    # Solid-phase inventory integrals INT c x^m dx, using the FV cell volumes
    # (the scheme's exact discrete invariant).
    initial_integral = y[0, :N] @ Vfv
    time_integrals = y[:, :N] @ Vfv
    delta_mass_free = -(m + 1) / R ** (m + 1) * vSample * MW_analyte * (initial_integral - time_integrals)

    src_off = N if is_inf_F else N + 1
    if num_srcs > 0:
        src_remaining = y[:, src_off:src_off + num_srcs]
        src_initial = src_params[::3]
        delta_mass_src = -vSample * MW_analyte * np.sum(src_initial - src_remaining, axis=1)
    else:
        delta_mass_src = np.zeros_like(t)

    delta_mass_total = delta_mass_free + delta_mass_src

    # Mass balance check
    c0_end = (m + 1) / R ** (m + 1) * (y[-1, :N] @ Vfv)
    c_Gas_end = c_gas[-1] * beta
    initial_total = c0free + beta * cgas_init + float(np.sum(src_params[::3]))
    final_source_total = float(np.sum(y[-1, src_off:src_off + num_srcs])) if num_srcs > 0 else 0.0
    final_total = c0_end + c_Gas_end + final_source_total
    denom = initial_total - final_total

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
        'c_gas_uM': c_gas,        # molar headspace concentration (model unit)
        'MW': MW_analyte,         # carried so mass-unit conversions are self-contained
        'c_feed_ppbv': c_feed_ppbv,
        'c_feed_uM': c_feed_uM,   # molar feed concentration (for unit conversions)
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
        # Size to (most of) the screen so the 3-column controls and the plots
        # both have room; stays resizable and is capped for very large monitors.
        try:
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            win_w = max(1280, min(1850, int(sw * 0.95)))
            win_h = max(720, min(1000, int(sh * 0.92)))
            self.root.geometry(f"{win_w}x{win_h}")
        except Exception:
            self.root.geometry("1700x950")

        # ── Top-level layout ────────────────────────────────────────────────
        # Row 0 : full-width action bar (run/save/export, plot-scale options,
        #         parameter-file buttons, result message) — always visible,
        #         never inside a scroll region.
        # Row 1 : a draggable split — controls (left) and plots (right).  The
        #         divider defaults to showing every parameter without scrolling;
        #         drag it left to give the plots more room (the controls then
        #         scroll), or right for the reverse.
        self.action_frame = ttk.Frame(root, padding=(10, 6))
        self.action_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_pane.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.control_frame = ttk.Frame(self.main_pane, padding="10")
        self.plot_frame = ttk.Frame(self.main_pane, padding="10")
        # weight 0 = controls keep their natural width; weight 1 = plots absorb
        # any extra (or reduced) window width when the whole window is resized.
        self.main_pane.add(self.control_frame, weight=0)
        self.main_pane.add(self.plot_frame, weight=1)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        
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
            "V_feed – Feed Tank Volume (ml)":               ("feed_tank_volume", False),
            # ── Sample geometry ───────────────────────────────────────────
            "R – Sample Radius / Half-thickness (µm)":      ("R",         True),
            "mSample – Sample Mass (mg)":                   ("mSample",   False),
            "rhoSample – Density (g/ml)":                   ("rhoSample", False),
            "Vvessel – Vessel Volume (ml)":                 ("Vvessel",   False),
            # ── Numerical / grid resolution ───────────────────────────────
            "N – Number of Grid Points":                    ("N",            True),
            "Grid Stretch – β / ratio (non-uniform grids)": ("grid_stretch", False),
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

        # Sweep attributes that must take whole-number values (rounded; the
        # solver builds the spatial grid with np.linspace(..., N) and cannot
        # accept a float). Used to round, de-duplicate, and label such sweeps.
        self.SWEEP_INT_ATTRS = {"N"}

        # Attributes that are numerical/grid controls rather than physical
        # parameters; offered for sweeping but excluded from the optimizer,
        # since fitting a resolution knob to experimental data is meaningless.
        self.OPT_EXCLUDED_ATTRS = {"N", "grid_stretch"}

        self.create_action_bar()
        self.create_controls()
        self.create_plots()
        
        # Load defaults if file exists
        self.load_defaults()

        # Place the divider so the controls start at their natural width (every
        # parameter visible, no scrolling) and the plots take the remaining
        # space.  Deferred until idle so requested sizes are known and the
        # window has been mapped.
        self.root.after_idle(self._init_sash_position)

    def _init_sash_position(self):
        """Set the initial controls/plots divider to the controls' natural width."""
        try:
            self.root.update_idletasks()
            want = self.control_frame.winfo_reqwidth()
            total = self.main_pane.winfo_width()
            # On a narrow window, still leave a usable strip for the plots.
            if total > 0:
                want = min(want, max(360, total - 360))
            self.main_pane.sashpos(0, max(1, int(want)))
        except Exception:
            pass

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

    def create_action_bar(self):
        """Build the always-visible top bar.

        Groups (left to right): simulation actions (Run / Save as Default /
        Export Data), plot-scale display options (log-time, log-concentration,
        time-axis min/max), and parameter-file management (Save Params As /
        Load Params).  The result/status message sits on its own line beneath
        them.  This bar lives outside every scroll region, so Run Simulation is
        always reachable without scrolling.
        """
        bar = ttk.LabelFrame(self.action_frame, text="Run & Display", padding=(8, 4))
        bar.pack(fill=tk.X)

        row1 = ttk.Frame(bar)
        row1.pack(fill=tk.X)

        # ── Simulation actions ───────────────────────────────────────────────
        self.run_sim_btn = ttk.Button(row1, text="Run Simulation",
                                       command=self.run_simulation)
        self.run_sim_btn.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row1, text="Save as Default",
                   command=self.save_defaults).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Export Data",
                   command=self.export_data).pack(side=tk.LEFT, padx=4)

        ttk.Separator(row1, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # ── Plot-scale display options (apply to single runs and sweeps) ─────
        ttk.Label(row1, text="Plot scales:").pack(side=tk.LEFT, padx=(0, 4))
        self.logtime_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Log time", variable=self.logtime_var,
                        command=self._refresh_plots).pack(side=tk.LEFT, padx=2)
        self.logconc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Log conc.", variable=self.logconc_var,
                        command=self._refresh_plots).pack(side=tk.LEFT, padx=2)

        # Headspace gas-concentration plot unit (applies to the gas-concentration
        # panel and its experimental overlay; the model curve is converted from
        # its molar output).  Changing it only re-renders, no re-simulation.
        ttk.Label(row1, text="  Conc unit:").pack(side=tk.LEFT)
        self.conc_unit = ttk.Combobox(row1, width=7, state="readonly",
                                      values=CONC_UNITS)
        self.conc_unit.set('ppbv')
        self.conc_unit.pack(side=tk.LEFT, padx=2)
        self.conc_unit.bind("<<ComboboxSelected>>",
                             lambda e: (self._update_derived(), self._refresh_plots()))

        # Sample mass-change plot unit (applies to the mass panel and its
        # experimental overlay; the model curve is converted from its native ng).
        ttk.Label(row1, text="  Mass unit:").pack(side=tk.LEFT)
        self.mass_unit = ttk.Combobox(row1, width=5, state="readonly",
                                      values=MASS_UNITS)
        self.mass_unit.set('ng')
        self.mass_unit.pack(side=tk.LEFT, padx=2)
        self.mass_unit.bind("<<ComboboxSelected>>",
                            lambda e: self._refresh_plots())

        # Time-axis limits applied for BOTH linear and log scales.  Leave blank
        # for autoscale; on a log axis a non-positive/blank min falls back to
        # the smallest positive time so t = 0 is excluded.
        ttk.Label(row1, text="   t min:").pack(side=tk.LEFT)
        self.time_min = ttk.Entry(row1, width=9)
        self.time_min.insert(0, "1e-6")
        self.time_min.pack(side=tk.LEFT, padx=2)
        self.time_min.bind("<Return>", lambda e: self._refresh_plots())
        ttk.Label(row1, text="t max:").pack(side=tk.LEFT)
        self.time_max = ttk.Entry(row1, width=9)
        self.time_max.pack(side=tk.LEFT, padx=2)
        self.time_max.bind("<Return>", lambda e: self._refresh_plots())

        ttk.Separator(row1, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # ── Parameter-file management ────────────────────────────────────────
        ttk.Button(row1, text="Save Params As…",
                   command=self.save_defaults_as).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Load Params…",
                   command=self.load_defaults_from_file).pack(side=tk.LEFT, padx=4)
        self.defaults_path_label = ttk.Label(
            row1, text=f"Param file: {os.path.basename(self.defaults_file)}",
            foreground="gray", font=('Arial', 8))
        self.defaults_path_label.pack(side=tk.LEFT, padx=(8, 0))

        # ── Result / status message (own line; wraps instead of widening) ────
        self.status_label = ttk.Label(bar, text="Ready", foreground="green",
                                       anchor='w', justify='left', wraplength=1300)
        self.status_label.pack(fill=tk.X, pady=(4, 0))

    def create_controls(self):
        """Three-column parameter layout.

        Sections are grouped into three side-by-side columns so the whole
        parameter set fits on screen without scrolling at the default window
        size.  The surrounding canvas/scrollbar is only a safety net for small
        windows or unusually long inputs (e.g. many sources or fit parameters).
        Run Simulation and friends live in the top action bar, not here.
        """
        # Scrollable container.  At the default divider position the three
        # columns fit entirely, so no scrollbar is needed; the bars only engage
        # if the window is short or the user drags the divider to shrink this
        # pane.  Vertical and horizontal scrolling are both available.
        self.control_frame.rowconfigure(0, weight=1)
        self.control_frame.columnconfigure(0, weight=1)

        canvas = tk.Canvas(self.control_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical",
                                  command=canvas.yview)
        hscroll = ttk.Scrollbar(self.control_frame, orient="horizontal",
                                command=canvas.xview)
        self.scrollable_frame = ttk.Frame(canvas)

        # Natural (minimum) content size is captured once the columns are built
        # (see end of this method).  The inner frame is then stretched to fill
        # the viewport whenever the pane is larger than that, so the three
        # columns spread out instead of leaving blank space; when the pane is
        # smaller, the scrollbars take over.
        self._ctrl_natural_w = 1
        self._ctrl_natural_h = 1
        self._ctrl_window = canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")

        def _fit_inner(event=None):
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            canvas.itemconfigure(
                self._ctrl_window,
                width=max(cw, self._ctrl_natural_w),
                height=max(ch, self._ctrl_natural_h))
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.scrollable_frame.bind("<Configure>",
                                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", _fit_inner)
        self._fit_ctrl_inner = _fit_inner

        canvas.configure(yscrollcommand=scrollbar.set, xscrollcommand=hscroll.set)

        # Mouse-wheel scrolls vertically; Shift+wheel horizontally — only while
        # the pointer is over the controls.
        def _ctrl_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _ctrl_shift_wheel(event):
            canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: (
            canvas.bind_all("<MouseWheel>", _ctrl_wheel),
            canvas.bind_all("<Shift-MouseWheel>", _ctrl_shift_wheel)))
        canvas.bind("<Leave>", lambda e: (
            canvas.unbind_all("<MouseWheel>"),
            canvas.unbind_all("<Shift-MouseWheel>")))


        # ── Two columns ──────────────────────────────────────────────────────
        # Two columns (rather than three) keep the controls narrow enough that
        # the plots get a fair share of the width at every window size, while
        # still fitting vertically without scrolling.  Equal weights + sticky
        # 'nsew' let the columns spread to fill the pane; the weighted row lets
        # them fill vertically.
        self.scrollable_frame.rowconfigure(0, weight=1)
        for _c in (0, 1):
            self.scrollable_frame.columnconfigure(_c, weight=1)

        col0 = ttk.Frame(self.scrollable_frame)
        col0.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        col1 = ttk.Frame(self.scrollable_frame)
        col1.grid(row=0, column=1, sticky='nsew', padx=(8, 0))

        # Sections expand to share any extra column height (even vertical
        # spacing) so a column never has a large blank tail on a tall window.
        def section(parent, title):
            lf = ttk.LabelFrame(parent, text=title, padding=(8, 4))
            lf.pack(fill=tk.X, expand=True, pady=(0, 8))
            return lf

        # =====================================================================
        # COLUMN 0 — Simulation / Temperature / Flow / Sample / Transport
        # =====================================================================
        sim = section(col0, "Simulation Parameters")
        r = 0
        self.tfinal = self.create_entry(sim, "Final Time (min):", "150", r); r += 1
        self.N = self.create_entry(sim, "Grid Points:", "201", r); r += 1
        self.rtol = self.create_entry(sim, "Relative Tolerance:", "1e-7", r); r += 1
        self.atol = self.create_entry(sim, "Absolute Tolerance:", "1e-8", r); r += 1
        ttk.Label(sim, text="Grid Scheme:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.grid_scheme = ttk.Combobox(sim, width=14, state="readonly")
        self.grid_scheme['values'] = GRID_SCHEMES
        self.grid_scheme.current(0)   # 'uniform' — general-purpose default
        self.grid_scheme.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        # Stretch parameter: beta for 'tanh' (use ~2-2.5 to resolve early-time
        # transients); h_max/h_min for 'geometric'; ignored for uniform.
        self.grid_stretch = self.create_entry(sim, "Grid Stretch (β / ratio):", "2.0", r); r += 1

        temp = section(col0, "Temperature Profile")
        r = 0
        self.dt1 = self.create_entry(temp, "Time at Initial Temp (min):", "60", r); r += 1
        self.T0 = self.create_entry(temp, "Initial Temp (°C):", "50", r); r += 1
        self.Tfinal = self.create_entry(temp, "Final Temp (°C):", "250", r); r += 1
        self.RR = self.create_entry(temp, "Ramp Rate (°C/min):", "5", r); r += 1

        flow = section(col0, "Flow Parameters")
        r = 0
        self.tEq = self.create_entry(flow, "Initial time at Q = 0 (min):", "30", r); r += 1
        self.Q = self.create_entry(flow, "Flow Rate (ml/min):", "20", r); r += 1
        self.tau_vessel_var = tk.StringVar(value="—")
        self.create_readout(flow, "Vessel gas residence time, τ_v (min):",
                            r, self.tau_vessel_var); r += 1

        feed = section(col1, "Feed Concentration Profile")
        r = 0
        self.n_steps = self.create_entry(feed, "Number of Steps:", "0", r); r += 1
        self.delta = self.create_entry(feed, "Δc per Step (ppbv):", "1000", r); r += 1
        self.step_time = self.create_entry(feed, "Step Time (min):", "240", r); r += 1
        self.base_conc = self.create_entry(feed, "Base Conc (ppbv):", "0", r); r += 1
        self.hold_time_initial = self.create_entry(feed, "Initial Hold Time (min):", "60", r); r += 1
        self.hold_time_final = self.create_entry(feed, "Final Hold Time (min):", "3000", r); r += 1
        # Finite feed-tank volume rolls the square-wave feed over exponentially
        # (Kumar eqs. 16-17); 0 = perfect square waves.
        self.feed_tank_volume = self.create_entry(feed, "Feed Tank Volume (ml):", "0", r); r += 1
        self.tau_feed_var = tk.StringVar(value="—")
        self.create_readout(feed, "Feed tank residence time, τ_f (min):",
                            r, self.tau_feed_var); r += 1

        # =====================================================================
        # COLUMN 1 — Feed / Source Terms / Sweep / Optimization
        # =====================================================================
        samp = section(col0, "Sample Properties")
        r = 0
        ttk.Label(samp, text="Geometry:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.m = ttk.Combobox(samp, width=14, state="readonly")
        self.m['values'] = ('0 - Slab', '1 - Cylinder', '2 - Sphere')
        self.m.current(2)
        self.m.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.R = self.create_entry(samp, "Sample radius / half-thickness (µm):", "200", r); r += 1
        self.mSample = self.create_entry(samp, "Total sample mass (mg):", "50", r); r += 1
        self.rhoSample = self.create_entry(samp, "Density (g/ml):", "1", r); r += 1
        self.Vvessel = self.create_entry(samp, "Vessel Volume (ml):", "10", r); r += 1
        self.MW_analyte = self.create_entry(samp, "Species MW (g/mol):", "18", r); r += 1

        trans = section(col0, "Transport Properties")
        r = 0
        self.Tref = self.create_entry(trans, "Reference temperature Tref (°C):", "50", r); r += 1
        self.K_ref = self.create_entry(trans, "Partition coeff K @ Tref:", "150", r); r += 1
        self.EaK = self.create_entry(trans, "Sorption Enthalpy (kJ/mol):", "-35", r); r += 1
        self.D_ref = self.create_entry(trans, "Diffusivity D @ Tref,c=0 (cm²/s):", "1e-7", r); r += 1
        self.EaD = self.create_entry(trans, "Diffusivity Ea (kJ/mol):", "15", r); r += 1
        self.cD = self.create_entry(trans, "Plasticizer power cD (µM, 'inf'):", "inf", r); r += 1
        self.F = self.create_entry(trans, "Surface transfer coeff F (cm/s, 'inf'):", "inf", r); r += 1
        self.c0free = self.create_entry(trans, "Initial mobile concentration (µM):", "10", r); r += 1
        # Mobile c0 expressed as a mass fraction (µg/g or mg/g) via c0*MW/rho.
        self.c0free_mf_var = tk.StringVar(value="—")
        self.create_readout(trans, "Initial mobile, mass basis (c₀·MW/ρ):",
                            r, self.c0free_mf_var); r += 1
        # Initial headspace gas concentration. For desorption with F = inf,
        # starting at equilibrium (cgas_init = c0free / K(T0)) removes the
        # unphysical t = 0 concentration jump between sample and headspace.
        # Entered in the selected gas-concentration unit (volumetric units are
        # interpreted at the initial temperature T0); the label updates with it.
        self.cgas_init_label = ttk.Label(trans, text="Initial gas concentration (µM):")
        self.cgas_init_label.grid(row=r, column=0, sticky=tk.W, pady=2)
        self.cgas_init = ttk.Entry(trans, width=12)
        self.cgas_init.insert(0, "0")
        self.cgas_init.grid(row=r, column=1, sticky=tk.E, pady=2)
        r += 1
        # Characteristic diffusion time ~ R^2 / D, at Tref and c = 0.
        self.tau_diff_var = tk.StringVar(value="—")
        self.create_readout(trans, "Char. diffusion time, R²/D (min):",
                            r, self.tau_diff_var); r += 1
        # Phase ratio β = (vessel volume − sample volume) / sample volume, with
        # the sample volume from the input mass and density.
        self.beta_var = tk.StringVar(value="—")
        self.create_readout(trans, "Phase ratio, β:",
                            r, self.beta_var); r += 1
        # No-flow mobile-species equilibrium concentration, c₀/(K + β) (uses the
        # K @ Tref above and the initial mobile concentration c₀).  Reported in
        # the currently selected gas-concentration unit (volumetric units use
        # the reference temperature Tref).
        self.noflow_var = tk.StringVar(value="—")
        self.create_readout(trans, "No-flow equil. conc., c₀/(K+β):",
                            r, self.noflow_var); r += 1

        src = section(col1, "Source Terms")
        r = 0
        ttk.Label(src, text="Number of Sources:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.num_sources = ttk.Entry(src, width=12)
        self.num_sources.insert(0, "1")
        self.num_sources.grid(row=r, column=1, sticky=tk.E, pady=2)
        self.num_sources.bind('<Return>', lambda e: self.update_source_fields())
        r += 1
        ttk.Button(src, text="Update Source Fields",
                   command=self.update_source_fields).grid(
            row=r, column=0, columnspan=2, pady=5)
        r += 1
        # Dynamic source rows live in their own sub-frame so adding/removing
        # sources never disturbs the rest of the column layout.
        self.source_container = ttk.Frame(src)
        self.source_container.grid(row=r, column=0, columnspan=2, sticky='ew')
        self.update_source_fields()

        # =====================================================================
        # Parameter Sweep / Parameter Optimization  (right column)
        # =====================================================================
        sweep = section(col1, "Parameter Sweep")
        sweep_header = ttk.Frame(sweep)
        sweep_header.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        self.sweep_check = ttk.Checkbutton(
            sweep_header, text="Enable",
            variable=self.sweep_enabled,
            command=self._toggle_sweep_ui)
        self.sweep_check.pack(side=tk.LEFT)
        r = 1
        ttk.Label(sweep, text="Sweep Parameter:").grid(
            row=r, column=0, sticky=tk.W, pady=2)
        self.sweep_param_var = tk.StringVar()
        self.sweep_param_combo = ttk.Combobox(
            sweep, textvariable=self.sweep_param_var,
            values=list(self.SWEEP_PARAMS.keys()), width=22, state='disabled')
        self.sweep_param_combo.current(0)
        self.sweep_param_combo.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        ttk.Label(sweep, text="Start Value:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.sweep_start = ttk.Entry(sweep, width=12, state='disabled')
        self.sweep_start.insert(0, "1e-8")
        self.sweep_start.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        ttk.Label(sweep, text="Stop Value:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.sweep_stop = ttk.Entry(sweep, width=12, state='disabled')
        self.sweep_stop.insert(0, "1e-6")
        self.sweep_stop.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        ttk.Label(sweep, text="Number of Points:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.sweep_n = ttk.Entry(sweep, width=12, state='disabled')
        self.sweep_n.insert(0, "5")
        self.sweep_n.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        ttk.Label(sweep, text="Log-spaced Values:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.sweep_log = tk.BooleanVar(value=False)
        self.sweep_log_check = ttk.Checkbutton(sweep, variable=self.sweep_log)
        self.sweep_log_check.config(state='disabled')
        self.sweep_log_check.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        ttk.Label(sweep, text="Colormap:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.sweep_cmap = ttk.Combobox(
            sweep, width=18, state="disabled",
            values=['viridis', 'plasma', 'coolwarm', 'tab10', 'rainbow', 'cividis'])
        self.sweep_cmap.set('viridis')
        self.sweep_cmap.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        opt = section(col1, "Parameter Optimization")
        opt_header = ttk.Frame(opt)
        opt_header.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 2))
        self.opt_check = ttk.Checkbutton(
            opt_header, text="Enable",
            variable=self.opt_enabled,
            command=self._toggle_opt_ui)
        self.opt_check.pack(side=tk.LEFT)
        r = 1
        ttk.Label(opt, text="# Parameters to Fit (1–6):").grid(
            row=r, column=0, sticky=tk.W, pady=2)
        self.opt_n_entry = ttk.Entry(opt, width=8, state='disabled')
        self.opt_n_entry.insert(0, "1")
        self.opt_n_entry.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        self.opt_build_btn = ttk.Button(
            opt, text="Build Parameter Rows",
            command=self._build_opt_rows, state='disabled')
        self.opt_build_btn.grid(row=r, column=0, columnspan=2, pady=3); r += 1

        # Column headers
        self.opt_col_header = ttk.Frame(opt)
        self.opt_col_header.grid(row=r, column=0, columnspan=2, sticky='ew')
        for text, w in [("Parameter", 26), ("Min", 7), ("Max", 7), ("Log?", 4)]:
            ttk.Label(self.opt_col_header, text=text, font=('Arial', 8, 'bold'),
                      width=w, anchor='center').pack(side=tk.LEFT, padx=1)
        r += 1

        # Dynamic parameter rows frame
        self.opt_rows_frame = ttk.Frame(opt)
        self.opt_rows_frame.grid(row=r, column=0, columnspan=2, sticky='ew'); r += 1

        self._build_opt_rows()

        ttk.Label(opt, text="Fit Target:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.opt_target = ttk.Combobox(
            opt, values=["Concentration (ppbv)", "Mass (ng)", "Both"],
            width=20, state='disabled')
        self.opt_target.set("Concentration (ppbv)")
        self.opt_target.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        ttk.Label(opt, text="Multi-Starts:").grid(row=r, column=0, sticky=tk.W, pady=2)
        # Number of Nelder-Mead fits to run: start 1 uses the current field
        # values; additional starts are drawn from a Latin-hypercube sample of
        # the bounds. The per-start results table reveals fit (non-)uniqueness.
        self.opt_nstarts = ttk.Entry(opt, width=8, state='disabled')
        self.opt_nstarts.insert(0, "1")
        self.opt_nstarts.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        ttk.Label(opt, text="Max Evals per Start:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.opt_maxiter = ttk.Entry(opt, width=8, state='disabled')
        self.opt_maxiter.insert(0, "200")
        self.opt_maxiter.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        # Run / Terminate / Results buttons
        opt_btn_frame = ttk.Frame(opt)
        opt_btn_frame.grid(row=r, column=0, columnspan=2, pady=5)
        self.opt_run_btn = ttk.Button(
            opt_btn_frame, text="Run Optimization",
            command=self.run_optimization, state='disabled')
        self.opt_run_btn.pack(side=tk.LEFT, padx=3)
        self.opt_stop_btn = ttk.Button(
            opt_btn_frame, text="Terminate Fit",
            command=self._terminate_optimization, state='disabled')
        self.opt_stop_btn.pack(side=tk.LEFT, padx=3)
        self.opt_results_btn = ttk.Button(
            opt_btn_frame, text="View Fit Results",
            command=self._show_multistart_window, state='disabled')
        self.opt_results_btn.pack(side=tk.LEFT, padx=3)

        # Lay out the canvas with its two scrollbars, then seed the canvas's
        # requested size from the assembled three-column content so the pane
        # opens wide/tall enough to show everything.  Grid sticky + the row/
        # column weights set above let the divider shrink it later, at which
        # point the scrollbars take over.
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")

        self.scrollable_frame.update_idletasks()
        # Natural minimum size = the assembled three-column content.  Note it
        # with expand temporarily off so it reflects the true content height,
        # then seed the canvas so the pane opens just wide/tall enough.
        self._ctrl_natural_w = max(1, self.scrollable_frame.winfo_reqwidth())
        self._ctrl_natural_h = max(1, self.scrollable_frame.winfo_reqheight())
        canvas.configure(width=self._ctrl_natural_w, height=self._ctrl_natural_h)
        self._fit_ctrl_inner()

        # Wire up the computed read-outs and show their initial values.
        self._bind_derived_updates()
        self._update_derived()

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

    def create_readout(self, parent, label, row, textvariable):
        """A non-editable, computed-value display row (label + right-aligned
        value).  Distinct foreground marks it as derived, not an input."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        val = ttk.Label(parent, textvariable=textvariable, anchor=tk.E,
                        width=12, foreground="#1a5fb4")
        val.grid(row=row, column=1, sticky=tk.E, pady=2)
        return val

    def _bind_derived_updates(self):
        """Recompute the derived read-outs whenever an input they depend on
        changes."""
        for w in (self.feed_tank_volume, self.Q, self.Vvessel, self.mSample,
                  self.rhoSample, self.R, self.D_ref, self.MW_analyte,
                  self.c0free, self.K_ref, self.Tref):
            w.bind("<KeyRelease>", self._update_derived, add="+")
            w.bind("<FocusOut>", self._update_derived, add="+")

    def _update_derived(self, *_event):
        """Update the computed read-outs (residence times, diffusion time).

        Each value is computed independently; a bad/blank field only blanks its
        own read-out (shown as '—') rather than raising.
        """
        def _f(widget):
            return float(widget.get())

        # Keep the initial-gas-concentration input label in step with the
        # selected gas-concentration unit.
        if hasattr(self, 'cgas_init_label'):
            try:
                self.cgas_init_label.config(
                    text=f"Initial gas concentration ({self._conc_plot_unit()}):")
            except tk.TclError:
                pass

        # Feed-tank residence time, tau_f = V_feed / Q  (min)
        try:
            V_feed = _f(self.feed_tank_volume)
            Q = _f(self.Q)
            if V_feed <= 0:
                self.tau_feed_var.set("0 (square wave)")
            elif Q <= 0:
                self.tau_feed_var.set("∞ (Q = 0)")
            else:
                self.tau_feed_var.set(f"{V_feed / Q:.4g}")
        except (ValueError, tk.TclError):
            self.tau_feed_var.set("—")

        # Vessel gas (headspace) residence time, tau_v = V_headspace / Q  (min)
        try:
            Q = _f(self.Q)
            vSample = _f(self.mSample) * 1e-3 / _f(self.rhoSample)   # mg, g/ml -> ml
            Vhead = _f(self.Vvessel) - vSample
            if Vhead <= 0:
                self.tau_vessel_var.set("— (no headspace)")
            elif Q <= 0:
                self.tau_vessel_var.set("∞ (Q = 0)")
            else:
                self.tau_vessel_var.set(f"{Vhead / Q:.4g}")
        except (ValueError, ZeroDivisionError, tk.TclError):
            self.tau_vessel_var.set("—")

        # Characteristic diffusion time, tau_D ~ R^2 / D  (min), at Tref, c = 0
        try:
            R_cm = _f(self.R) * 1e-4              # µm -> cm
            D = _f(self.D_ref)                   # cm^2/s
            if R_cm <= 0 or D <= 0:
                self.tau_diff_var.set("—")
            else:
                self.tau_diff_var.set(f"{(R_cm * R_cm / D) / 60.0:.4g}")
        except (ValueError, ZeroDivisionError, tk.TclError):
            self.tau_diff_var.set("—")

        # Phase ratio β = (vessel − sample volume) / sample volume, with the
        # sample volume taken from the input sample mass and density.
        beta = None
        try:
            vSample = _f(self.mSample) * 1e-3 / _f(self.rhoSample)   # mg, g/ml -> ml
            Vhead = _f(self.Vvessel) - vSample
            if vSample > 0 and Vhead > 0:
                beta = Vhead / vSample
                self.beta_var.set(f"{beta:.4g}")
            elif vSample > 0 and Vhead <= 0:
                self.beta_var.set("— (no headspace)")
            else:
                self.beta_var.set("—")
        except (ValueError, ZeroDivisionError, tk.TclError):
            self.beta_var.set("—")

        # No-flow mobile-species equilibrium concentration c₀/(K + β), using the
        # partition coefficient K @ Tref and the initial mobile concentration.
        # Reported in the selected gas-concentration unit; volumetric units
        # (ppbv/ppmv) are evaluated at the reference temperature Tref.
        try:
            if beta is not None:
                denom = _f(self.K_ref) + beta
                c_eq = _f(self.c0free) / denom if denom > 0 else None
            else:
                c_eq = None
            if c_eq is not None:
                unit = self._conc_plot_unit()
                try:
                    T_ref_K = _f(self.Tref) + 273.15
                except (ValueError, tk.TclError):
                    T_ref_K = 298.15
                val = float(self._conc_from_uM(c_eq, T_ref_K, unit, self._safe_MW()))
                self.noflow_var.set(f"{val:.4g} {unit}")
            else:
                self.noflow_var.set("—")
        except (ValueError, ZeroDivisionError, tk.TclError):
            self.noflow_var.set("—")

        # Mass fractions (µg/g or mg/g) for the initial mobile conc and every
        # source c0, via c0*MW/rho (see _mass_fraction_str).
        try:
            MW = _f(self.MW_analyte)
        except (ValueError, tk.TclError):
            MW = float('nan')
        try:
            rho = _f(self.rhoSample)
        except (ValueError, tk.TclError):
            rho = float('nan')

        try:
            self.c0free_mf_var.set(self._mass_fraction_str(_f(self.c0free), MW, rho))
        except (ValueError, tk.TclError):
            self.c0free_mf_var.set("—")

        for sw in getattr(self, 'source_entries', []):
            if len(sw) < 7:
                continue
            try:
                c0 = _f(sw[1])
            except (ValueError, tk.TclError):
                c0 = float('nan')
            sw[6].config(text=self._mass_fraction_str(c0, MW, rho))

    @staticmethod
    def _mass_fraction_str(c0_uM, MW, rho):
        """Format a concentration c0 (µM) as a sample mass fraction using
        c0*MW/rho, auto-selecting µg/g or mg/g.  Returns '—' for non-finite or
        non-positive-density inputs.

        c0 [µmol/L] · MW [g/mol] / rho [g/ml] · 1e-3  →  µg per g of sample.
        """
        try:
            c0 = float(c0_uM); mw = float(MW); r = float(rho)
        except (TypeError, ValueError):
            return "—"
        if not (np.isfinite(c0) and np.isfinite(mw) and np.isfinite(r)) or r <= 0:
            return "—"
        ugg = c0 * mw / r * 1e-3
        if abs(ugg) >= 1000.0:
            return f"{ugg / 1000.0:.4g} mg/g"
        return f"{ugg:.4g} µg/g"

    # ------------------------------------------------------------------ #
    #  Concentration-unit conversions                                    #
    #                                                                    #
    #  The model integrates a molar headspace concentration c_gas (µM).  #
    #  Everything the user sees/exports can be expressed in any of       #
    #  CONC_UNITS.  The factors below are exact:                         #
    #      ppbv  : c[µM] · 1000·22.414/273.15 · T_K   (T-dependent)      #
    #      ppmv  : ppbv / 1000                          (T-dependent)    #
    #      µg/m³ : c[µM] · MW · 1000                    (T-independent)  #
    #      mg/m³ : c[µM] · MW                           (T-independent)  #
    #  The pure factor/convert helpers take no Tk widgets, so they are   #
    #  safe to call from the optimiser worker thread.                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _conc_factor_uM_to(unit, T_K, MW):
        """Multiplicative factor f with  value[unit] = c[µM] · f.

        T_K may be a scalar or array (used only for volumetric units);
        MW is g/mol (used only for mass-per-volume units)."""
        if unit == 'ppmv':
            return 1000.0 * 22.414 / 273.15 * np.asarray(T_K) / 1000.0
        if unit == 'µg/m³':
            return MW * 1000.0
        if unit == 'mg/m³':
            return MW * 1.0
        # default / 'ppbv'
        return 1000.0 * 22.414 / 273.15 * np.asarray(T_K)

    def _conc_from_uM(self, c_uM, T_K, unit, MW):
        """Convert a molar concentration (µM) to `unit`."""
        return np.asarray(c_uM, dtype=float) * self._conc_factor_uM_to(unit, T_K, MW)

    def _conc_to_uM(self, val, T_K, unit, MW):
        """Convert a value expressed in `unit` back to a molar concentration (µM)."""
        return np.asarray(val, dtype=float) / self._conc_factor_uM_to(unit, T_K, MW)

    def _safe_MW(self, default=18.0):
        """Analyte MW from the entry, falling back to `default` if blank/bad."""
        try:
            mw = float(self.MW_analyte.get())
            return mw if np.isfinite(mw) and mw > 0 else default
        except (ValueError, tk.TclError, AttributeError):
            return default

    def _cgas_init_uM(self):
        """Initial headspace gas concentration, entered in the selected gas
        unit, converted to µM for the model.  Volumetric units (ppbv/ppmv) are
        interpreted at the initial temperature T0."""
        val = float(self.cgas_init.get())   # may raise; handled by caller
        unit = self._conc_plot_unit()
        try:
            T0_K = float(self.T0.get()) + 273.15
        except (ValueError, tk.TclError):
            T0_K = 298.15
        return float(self._conc_to_uM(val, T0_K, unit, self._safe_MW()))

    def _temp_K_at(self, times):
        """Absolute temperature (K) along `times`, using the GUI temperature
        profile.  Falls back to a constant T0 (or 25 °C) if the profile inputs
        are unusable, so volumetric conversions never raise."""
        times = np.asarray(times, dtype=float)
        try:
            T_C, _ = temperature_vec(times,
                                     float(self.dt1.get()),
                                     float(self.T0.get()),
                                     float(self.Tfinal.get()),
                                     float(self.RR.get()))
            return np.asarray(T_C, dtype=float) + 273.15
        except (ValueError, tk.TclError, AttributeError):
            try:
                return np.full_like(times, float(self.T0.get()) + 273.15)
            except (ValueError, tk.TclError, AttributeError):
                return np.full_like(times, 298.15)

    def _model_conc_in(self, res, unit):
        """Model effluent concentration from a results dict, expressed in `unit`."""
        T_K = np.asarray(res['T_C'], dtype=float) + 273.15
        MW = res.get('MW', self._safe_MW())
        if 'c_gas_uM' in res:
            c_uM = np.asarray(res['c_gas_uM'], dtype=float)
        else:
            # Backward-compatible fallback: recover µM from the ppbv field.
            c_uM = np.asarray(res['c_gas_ppbv'], dtype=float) / (
                1000.0 * 22.414 / 273.15 * T_K)
        return self._conc_from_uM(c_uM, T_K, unit, MW)

    def _model_feed_in(self, res, unit):
        """Model feed concentration from a results dict, expressed in `unit`.

        The feed mixing ratio is supplied/stored in ppbv; volumetric units use
        the model temperature profile, mass-per-volume units use the analyte MW."""
        T_K = np.asarray(res['T_C'], dtype=float) + 273.15
        MW = res.get('MW', self._safe_MW())
        if 'c_feed_uM' in res:
            c_uM = np.asarray(res['c_feed_uM'], dtype=float)
        else:
            c_uM = np.asarray(res['c_feed_ppbv'], dtype=float) / (
                1000.0 * 22.414 / 273.15 * T_K)
        return self._conc_from_uM(c_uM, T_K, unit, MW)

    def _exp_conc_convert(self, times, raw_vals, from_unit, to_unit):
        """Convert experimental concentrations from `from_unit` to `to_unit`.

        Volumetric (ppbv/ppmv) conversions use the model temperature profile
        sampled at the experimental times; mass-per-volume conversions are
        temperature-independent."""
        if from_unit == to_unit:
            return np.asarray(raw_vals, dtype=float)
        times = np.asarray(times, dtype=float)
        MW = self._safe_MW()
        T_K = self._temp_K_at(times)
        c_uM = self._conc_to_uM(raw_vals, T_K, from_unit, MW)
        return self._conc_from_uM(c_uM, T_K, to_unit, MW)

    def _conc_cmp_unit(self):
        """Unit used for model/experiment comparison (R², SSE).  Defined as the
        unit the experimental data were pasted in, so no data are re-scaled."""
        try:
            u = self.conc_data_unit.get()
            return u if u in CONC_UNITS else 'ppbv'
        except (AttributeError, tk.TclError):
            return 'ppbv'

    def _conc_plot_unit(self):
        """Unit selected for plotting / on-screen display of the concentration."""
        try:
            u = self.conc_unit.get()
            return u if u in CONC_UNITS else 'ppbv'
        except (AttributeError, tk.TclError):
            return 'ppbv'

    @staticmethod
    def _conc_unit_mathtext(unit):
        """Matplotlib-mathtext rendering of a concentration unit."""
        return {
            'ppbv':  'ppbv',
            'ppmv':  'ppmv',
            'µg/m³': r'$\mu$g m$^{-3}$',
            'mg/m³': r'mg m$^{-3}$',
        }.get(unit, unit)

    def _conc_axis_label(self, unit, export=False):
        """Y-axis label for the effluent-concentration plot in `unit`.

        The compact form is used for the small on-screen subplot; the verbose
        form (with the c_gas symbol) is used for the exported single panel."""
        u = self._conc_unit_mathtext(unit)
        if export:
            return r'Effluent gas concentration, $c_\mathrm{gas}$ (' + u + ')'
        return 'Concentration (' + u + ')'

    def _feed_axis_label(self, unit, export=False):
        """Y-axis label for the feed-concentration plot in `unit`."""
        u = self._conc_unit_mathtext(unit)
        if export:
            return r'Feed concentration, $c_\mathrm{feed}$ (' + u + ')'
        return 'Feed Conc (' + u + ')'

    # ------------------------------------------------------------------ #
    #  Mass-change unit conversions                                      #
    #                                                                    #
    #  The model accumulates the sample mass change in nanograms (ng).   #
    #  µg and mg are pure decimal rescalings (no temperature/MW needed). #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mass_factor_ng_to(unit):
        """Multiplicative factor f with  value[unit] = m[ng] · f."""
        return {'ng': 1.0, 'µg': 1e-3, 'mg': 1e-6}.get(unit, 1.0)

    def _mass_from_ng(self, m_ng, unit):
        """Convert a mass (ng) to `unit`."""
        return np.asarray(m_ng, dtype=float) * self._mass_factor_ng_to(unit)

    def _mass_to_ng(self, val, unit):
        """Convert a value expressed in `unit` back to ng."""
        return np.asarray(val, dtype=float) / self._mass_factor_ng_to(unit)

    def _mass_plot_unit(self):
        """Unit selected for plotting / on-screen display of the mass change."""
        try:
            u = self.mass_unit.get()
            return u if u in MASS_UNITS else 'ng'
        except (AttributeError, tk.TclError):
            return 'ng'

    def _mass_cmp_unit(self):
        """Unit used for model/experiment mass comparison (R²): the unit the
        experimental mass data were pasted in, so no data are re-scaled."""
        try:
            u = self.mass_data_unit.get()
            return u if u in MASS_UNITS else 'ng'
        except (AttributeError, tk.TclError):
            return 'ng'

    def _model_mass_in(self, res, unit):
        """Model sample mass change from a results dict, expressed in `unit`."""
        return self._mass_from_ng(res['delta_mass'], unit)

    def _exp_mass_convert(self, raw_vals, from_unit, to_unit):
        """Convert experimental mass values from `from_unit` to `to_unit`."""
        if from_unit == to_unit:
            return np.asarray(raw_vals, dtype=float)
        return self._mass_from_ng(self._mass_to_ng(raw_vals, from_unit), to_unit)

    @staticmethod
    def _mass_unit_mathtext(unit):
        """Matplotlib-mathtext rendering of a mass unit."""
        return {'ng': 'ng', 'µg': r'$\mu$g', 'mg': 'mg'}.get(unit, unit)

    def _mass_axis_label(self, unit, export=False):
        """Y-axis label for the sample-mass-change plot in `unit`."""
        u = self._mass_unit_mathtext(unit)
        if export:
            return r'Sample mass change, $\Delta m$ (' + u + ')'
        return r'$\Delta m$ (' + u + ')'

    def save_defaults(self):
        """Save current GUI values as defaults to a JSON file"""
        try:
            defaults = {
                'tfinal': self.tfinal.get(),
                'N': self.N.get(),
                'rtol': self.rtol.get(),
                'atol': self.atol.get(),
                'grid_scheme': self.grid_scheme.get(),
                'grid_stretch': self.grid_stretch.get(),
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
                'feed_tank_volume': self.feed_tank_volume.get(),
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
                'cgas_init': self.cgas_init.get(),
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
            defaults['logtime'] = self.logtime_var.get()
            defaults['logconc'] = self.logconc_var.get()
            defaults['time_min'] = self.time_min.get()
            defaults['time_max'] = self.time_max.get()

            # Save optimization settings
            defaults['opt_enabled'] = self.opt_enabled.get()
            defaults['opt_n'] = self.opt_n_entry.get()
            defaults['opt_target'] = self.opt_target.get()
            defaults['opt_nstarts'] = self.opt_nstarts.get()
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
            defaults['conc_plot_unit'] = self.conc_unit.get()
            defaults['conc_data_unit'] = self.conc_data_unit.get()
            defaults['mass_plot_unit'] = self.mass_unit.get()
            defaults['mass_data_unit'] = self.mass_data_unit.get()
            
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
            
            # Set grid scheme combobox (older parameter files lack these keys)
            gs_val = defaults.get('grid_scheme', 'uniform')
            try:
                idx = self.grid_scheme['values'].index(gs_val)
                self.grid_scheme.current(idx)
            except (ValueError, tk.TclError):
                self.grid_scheme.current(0)
            
            self.grid_stretch.delete(0, tk.END)
            self.grid_stretch.insert(0, defaults.get('grid_stretch', '2.0'))
            
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

            self.feed_tank_volume.delete(0, tk.END)
            self.feed_tank_volume.insert(0, defaults.get('feed_tank_volume', '0'))
            
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
            self.cgas_init.delete(0, tk.END)
            self.cgas_init.insert(0, defaults.get('cgas_init', '0'))
            
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

            # Restore concentration plot / experimental-data units.
            cpu = defaults.get('conc_plot_unit', 'ppbv')
            if cpu in CONC_UNITS:
                self.conc_unit.set(cpu)
            cdu = defaults.get('conc_data_unit', 'ppbv')
            if cdu in CONC_UNITS:
                self.conc_data_unit.set(cdu)
            # Restore mass plot / experimental-data units.
            mpu = defaults.get('mass_plot_unit', 'ng')
            if mpu in MASS_UNITS:
                self.mass_unit.set(mpu)
            mdu = defaults.get('mass_data_unit', 'ng')
            if mdu in MASS_UNITS:
                self.mass_data_unit.set(mdu)

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
            self.logtime_var.set(defaults.get('logtime', False))
            self.logconc_var.set(defaults.get('logconc', False))
            self.time_min.delete(0, tk.END)
            self.time_min.insert(0, defaults.get('time_min',
                                                 defaults.get('logtime_min', '1e-6')))
            self.time_max.delete(0, tk.END)
            self.time_max.insert(0, defaults.get('time_max', ''))
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
            self.opt_nstarts.config(state='normal')
            self.opt_nstarts.delete(0, tk.END)
            self.opt_nstarts.insert(0, defaults.get('opt_nstarts', '1'))
            # Rebuild rows and restore their values
            self._build_opt_rows()
            opt_rows_data = defaults.get('opt_rows', [])
            param_names = [k for k, (attr, _) in self.SWEEP_PARAMS.items()
                           if attr not in self.OPT_EXCLUDED_ATTRS]
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

            # Refresh the computed read-outs for the freshly loaded values.
            self._update_derived()

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

        # Source rows live in their own sub-frame (self.source_container) and
        # are numbered locally from 0, so the rest of the column is undisturbed.
        parent = self.source_container
        row = 0

        for i in range(n_sources):
            prev = existing_values[i] if i < len(existing_values) else {}

            ttk.Label(parent, text=f"--- Source {i+1} ---",
                     font=('Arial', 9, 'italic')).grid(row=row, column=0, columnspan=2, pady=3)
            row += 1

            c0_label = ttk.Label(parent, text=f"  c₀,{i+1} (µM):")
            c0_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            c0_entry = ttk.Entry(parent, width=12)
            c0_entry.insert(0, prev.get('c0', '100'))
            c0_entry.grid(row=row, column=1, sticky=tk.E, pady=2)
            # Recompute the mass-fraction read-out as this c0 is edited.
            c0_entry.bind("<KeyRelease>", self._update_derived, add="+")
            c0_entry.bind("<FocusOut>", self._update_derived, add="+")
            # c0 expressed as a mass fraction (µg/g or mg/g) via c0*MW/rho,
            # shown immediately to the right of the input box.
            mf_label = ttk.Label(parent, text="—", foreground="#1a5fb4",
                                 font=('Arial', 8))
            mf_label.grid(row=row, column=2, sticky=tk.W, padx=(6, 0), pady=2)
            row += 1

            A_label = ttk.Label(parent, text=f"  A{i+1} (1/min):")
            A_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            A_entry = ttk.Entry(parent, width=12)
            A_entry.insert(0, prev.get('A', '1e8'))
            A_entry.grid(row=row, column=1, sticky=tk.E, pady=2)
            row += 1

            Ea_label = ttk.Label(parent, text=f"  Ea,{i+1} (kJ/mol):")
            Ea_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            Ea_entry = ttk.Entry(parent, width=12)
            Ea_entry.insert(0, prev.get('Ea', '80'))
            Ea_entry.grid(row=row, column=1, sticky=tk.E, pady=2)
            row += 1

            self.source_entries.append([c0_label, c0_entry, A_label, A_entry,
                                        Ea_label, Ea_entry, mf_label])

        # Refresh the derived read-outs (incl. the new per-source mass fractions)
        # for the freshly built rows, once the dependent fields exist.
        if hasattr(self, 'beta_var'):
            self._update_derived()

    def create_plots(self):
        # The figure scales to fill the entire plot pane at any window or
        # monitor size, so there is never empty space around it.  (No scroll
        # region: the figure resizes with the pane instead of staying fixed.)
        plots_frame = ttk.Frame(self.plot_frame)
        plots_frame.pack(fill=tk.BOTH, expand=True)

        # constrained_layout lets matplotlib re-space the panels automatically
        # as the figure resizes, so titles and tick labels never overlap no
        # matter how tall or short the plot pane becomes.
        self.fig = Figure(figsize=(15, 11.5), dpi=100, constrained_layout=True)
        try:
            self.fig.set_constrained_layout_pads(
                w_pad=0.06, h_pad=0.06, wspace=0.04, hspace=0.06)
        except Exception:
            pass

        gs = self.fig.add_gridspec(3, 3)
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
        # Fill the pane and expand: matplotlib resizes the figure to the widget,
        # so the plots always use the full plot area at any window size.
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add experimental data inputs embedded in the plot area
        exp_frame = ttk.LabelFrame(plots_frame, text="Experimental Data", padding="10")
        exp_frame.place(relx=0.38, rely=0.68, relwidth=0.61, relheight=0.30)
        
        # Concentration data
        conc_frame = ttk.Frame(exp_frame)
        conc_frame.pack(side=tk.LEFT, padx=15, expand=True, fill=tk.BOTH)
        conc_hdr = ttk.Frame(conc_frame)
        conc_hdr.pack(fill=tk.X)
        ttk.Label(conc_hdr, text="Concentration (time, value):",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        ttk.Label(conc_hdr, text="Units:").pack(side=tk.LEFT, padx=(6, 2))
        self.conc_data_unit = ttk.Combobox(conc_hdr, width=7, state="readonly",
                                            values=CONC_UNITS)
        self.conc_data_unit.set('ppbv')
        self.conc_data_unit.pack(side=tk.LEFT)
        self.conc_data_unit.bind("<<ComboboxSelected>>", lambda e: self._refresh_plots())
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
        mass_hdr = ttk.Frame(mass_frame)
        mass_hdr.pack(fill=tk.X)
        ttk.Label(mass_hdr, text="Mass (time, value):",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        ttk.Label(mass_hdr, text="Units:").pack(side=tk.LEFT, padx=(6, 2))
        self.mass_data_unit = ttk.Combobox(mass_hdr, width=5, state="readonly",
                                           values=MASS_UNITS)
        self.mass_data_unit.set('ng')
        self.mass_data_unit.pack(side=tk.LEFT)
        self.mass_data_unit.bind("<<ComboboxSelected>>", lambda e: self._refresh_plots())
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
                    try:
                        t_val = float(parts[0])
                        v_val = float(parts[1])
                    except ValueError:
                        continue   # skip header rows / non-numeric lines
                    time_data.append(t_val)
                    value_data.append(v_val)

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
            'Feed Tank Volume (ml)':          self.feed_tank_volume.get(),
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
            f'Initial gas concentration ({self._conc_plot_unit()})': self.cgas_init.get(),
        }
        for i, sw in enumerate(self.source_entries):
            params[f'Source_{i+1}_c0 (uM)']    = sw[1].get()
            params[f'Source_{i+1}_A (1/min)']   = sw[3].get()
            params[f'Source_{i+1}_Ea (kJ/mol)'] = sw[5].get()
        return params

    def _derived_constants_dict(self):
        """Calculated constants shown in the GUI read-outs — residence times,
        characteristic diffusion time, phase ratio, no-flow equilibrium
        concentration, and the c0 mass fractions — captured exactly as
        displayed (units included in the value where they vary) so the export
        records the same numbers the user sees."""
        # Refresh first so the read-outs reflect the current input fields.
        self._update_derived()

        def _g(var):
            try:
                return var.get()
            except (AttributeError, tk.TclError):
                return "—"

        consts = {
            'Feed tank residence time, tau_f (min)':   _g(self.tau_feed_var),
            'Vessel gas residence time, tau_v (min)':  _g(self.tau_vessel_var),
            'Char. diffusion time, R^2/D (min)':       _g(self.tau_diff_var),
            'Phase ratio, beta':                       _g(self.beta_var),
            'No-flow equil. conc., c0/(K+beta)':       _g(self.noflow_var),
            'Initial mobile, mass basis (c0*MW/rho)':  _g(self.c0free_mf_var),
        }
        for i, sw in enumerate(getattr(self, 'source_entries', [])):
            if len(sw) > 6:
                try:
                    consts[f'Source_{i+1}_c0, mass basis (c0*MW/rho)'] = sw[6].cget('text')
                except (AttributeError, tk.TclError):
                    pass
        return consts

    @staticmethod
    def _results_to_df(results):
        T_K = np.asarray(results['T_C'], dtype=float) + 273.15
        MW = results.get('MW', None)
        if 'c_gas_uM' in results:
            c_uM = np.asarray(results['c_gas_uM'], dtype=float)
        else:
            c_uM = np.asarray(results['c_gas_ppbv'], dtype=float) / (
                1000.0 * 22.414 / 273.15 * T_K)
        # Feed concentration in molar units (for the alternate feed columns).
        if 'c_feed_uM' in results:
            cf_uM = np.asarray(results['c_feed_uM'], dtype=float)
        else:
            cf_uM = np.asarray(results['c_feed_ppbv'], dtype=float) / (
                1000.0 * 22.414 / 273.15 * T_K)
        cols = {
            't (min)':               results['t'],
            'T (C)':                 results['T_C'],
            'c_gas (ppbv)':          results['c_gas_ppbv'],
            'c_gas (ppmv)':          np.asarray(results['c_gas_ppbv'], dtype=float) / 1000.0,
        }
        if MW is not None:
            cols['c_gas (ug/m3)'] = c_uM * MW * 1000.0
            cols['c_gas (mg/m3)'] = c_uM * MW
        cols['c_feed (ppbv)'] = results['c_feed_ppbv']
        cols['c_feed (ppmv)'] = np.asarray(results['c_feed_ppbv'], dtype=float) / 1000.0
        if MW is not None:
            cols['c_feed (ug/m3)'] = cf_uM * MW * 1000.0
            cols['c_feed (mg/m3)'] = cf_uM * MW
        cols.update({
            'Q (ml/min)':            results['Q'],
            'D (cm^2/s)':            results['D'],
            'S (cm^3@STP/atm/cm^3)': results['S'],
            'delta_m (ng)':          results['delta_mass'],
            'delta_m (ug)':          np.asarray(results['delta_mass'], dtype=float) * 1e-3,
            'delta_m (mg)':          np.asarray(results['delta_mass'], dtype=float) * 1e-6,
        })
        return pd.DataFrame(cols)

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

    def _export_individual_plots(self, stem, labeled_results,
                                 exp_conc_time, exp_conc_data,
                                 exp_mass_time, exp_mass_data):
        """Save each plot panel as its own 300-dpi publication-styled PNG.

        Each figure gets a full black box (all four spines, width 3), inward
        major+minor ticks on all sides (no top/right tick labels), no
        gridlines, thick lines, large non-bold fonts, a tab10 colour cycle,
        and a thin-boxed legend placed inside the axes. The log-time axis
        setting (and its lower limit) match what is shown on screen.
        Returns the list of file paths written.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        log_t, left_limit, right_limit = self._time_axis_limits(labeled_results)
        tab10 = list(plt.get_cmap('tab10').colors)
        is_sweep = len(labeled_results) > 1
        short_label = (getattr(self, '_last_sweep_label', None) or '').split('–')[-1].strip()
        # Match the on-screen log-concentration toggle so exported and previewed
        # gas-concentration panels share the same y-axis scale.
        log_conc = bool(getattr(self, 'logconc_var', None) and self.logconc_var.get())

        # Concentration units: plot in the selected display unit, overlay the
        # experimental data after converting from the unit it was pasted in.
        plot_unit = self._conc_plot_unit()
        data_unit = self._conc_cmp_unit()
        mass_plot_unit = self._mass_plot_unit()
        mass_data_unit = self._mass_cmp_unit()

        # (file suffix, data key, axis title, y-log?, experimental-overlay key)
        panels = [
            ('temperature', 'T_C',         'Temperature, $T$ (°C)',                                 False, None),
            ('feed_conc',   'c_feed_ppbv', self._feed_axis_label(plot_unit, export=True), False, None),
            ('flow_rate',   'Q',           'Flow rate, $Q$ (mL min$^{-1}$)',                        False, None),
            ('diffusivity', 'D',           'Diffusivity, $D$ (cm$^2$ s$^{-1}$)',                    True,  None),
            ('solubility',  'S',           'Solubility, $S$ (cm$^3$(STP) cm$^{-3}$ atm$^{-1}$)',    False, None),
            ('mass_change', 'delta_mass',  self._mass_axis_label(mass_plot_unit, export=True),      False, 'mass'),
            ('gas_conc',    'c_gas_ppbv',  self._conc_axis_label(plot_unit, export=True),           log_conc, 'conc'),
        ]
        xlabel = 'Time, $t$ (min)'
        written = []

        for suffix, key, ylabel, ylog, exp_key in panels:
            # constrained_layout reserves room for the (large, possibly long)
            # axis labels so the y-axis title is never clipped on export.
            fig = Figure(figsize=(8, 6), constrained_layout=True)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            for i, (lbl, res) in enumerate(labeled_results):
                if suffix == 'gas_conc':
                    ydata = self._model_conc_in(res, plot_unit)
                elif suffix == 'feed_conc':
                    ydata = self._model_feed_in(res, plot_unit)
                elif suffix == 'mass_change':
                    ydata = self._model_mass_in(res, mass_plot_unit)
                else:
                    ydata = res[key]
                ax.plot(res['t'], ydata, linewidth=3,
                        color=tab10[i % len(tab10)],
                        label=(lbl if is_sweep else 'Model'))

            if suffix == 'mass_change':
                ax.axhline(0, color='black', linestyle='--', linewidth=1.5)

            has_exp = False
            if not is_sweep and exp_key == 'conc' and exp_conc_time is not None:
                exp_plot = self._exp_conc_convert(exp_conc_time, exp_conc_data,
                                                  data_unit, plot_unit)
                ax.plot(exp_conc_time, exp_plot, 'o', color='black',
                        markersize=9, label='Experimental', zorder=10)
                has_exp = True
            elif not is_sweep and exp_key == 'mass' and exp_mass_time is not None:
                exp_mplot = self._exp_mass_convert(exp_mass_data, mass_data_unit, mass_plot_unit)
                ax.plot(exp_mass_time, exp_mplot, 'o', color='black',
                        markersize=9, label='Experimental', zorder=10)
                has_exp = True

            # ── Publication styling ───────────────────────────────────────
            if ylog:
                ax.set_yscale('log')
            if log_t:
                ax.set_xscale('log')
            if left_limit is not None or right_limit is not None:
                ax.set_xlim(left=left_limit, right=right_limit)

            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(3)
                sp.set_color('black')
            ax.minorticks_on()
            ax.tick_params(which='major', direction='in', length=9, width=2,
                           top=True, bottom=True, left=True, right=True,
                           labeltop=False, labelright=False, labelsize=18)
            ax.tick_params(which='minor', direction='in', length=5, width=1.5,
                           top=True, bottom=True, left=True, right=True)
            ax.grid(False)
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)

            if is_sweep or has_exp:
                leg = ax.legend(loc='best', fontsize=15, frameon=True,
                                edgecolor='black', framealpha=1.0, fancybox=False,
                                title=(short_label if is_sweep else None),
                                title_fontsize=15,
                                ncol=2 if (is_sweep and len(labeled_results) > 8) else 1)
                leg.get_frame().set_linewidth(1.0)

            path = f"{stem}_{suffix}.png"
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.08)
            written.append(path)

        return written

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
            constants = self._derived_constants_dict()

            # Parse experimental data once
            exp_conc_time, exp_conc_data = self.parse_experimental_data(
                self.conc_data_text, self.conc_interp_var, self.conc_interp_n, self.conc_interp_method)
            exp_mass_time, exp_mass_data = self.parse_experimental_data(
                self.mass_data_text, self.mass_interp_var, self.mass_interp_n, self.mass_interp_method)

            # Concentration comparison unit = the unit the data were pasted in.
            cmp_unit = self._conc_cmp_unit()
            conc_col = f'concentration ({cmp_unit})'
            # Mass comparison unit = the unit the experimental mass was pasted in.
            mass_cmp_unit = self._mass_cmp_unit()
            mass_col = f'mass ({mass_cmp_unit})'

            def _r2_conc(res):
                """R² for concentration with the model converted to cmp_unit."""
                if exp_conc_time is None:
                    return None
                return self._compute_r_squared(
                    res['t'], self._model_conc_in(res, cmp_unit),
                    exp_conc_time, exp_conc_data)

            def _r2_mass(res):
                """R² for mass with the model converted to the mass cmp unit."""
                if exp_mass_time is None:
                    return None
                return self._compute_r_squared(
                    res['t'], self._model_mass_in(res, mass_cmp_unit),
                    exp_mass_time, exp_mass_data)

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

                f.write("# CALCULATED CONSTANTS:\n")
                for key, value in constants.items():
                    f.write(f"#   {key}: {value}\n")
                f.write("#\n")

                if not is_sweep:
                    # ── Single run ───────────────────────────────────────────
                    res = self.current_results
                    f.write(f"# Mass Balance Error (%): {res['mass_bal_error']:.6e}\n")
                    f.write(f"# Solve Time (s): {res['solve_time']:.4f}\n")
                    # R² (coefficient of determination)
                    r2_conc = _r2_conc(res)
                    r2_mass = _r2_mass(res)
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
                                          conc_col: exp_conc_data}).to_csv(f, index=False, lineterminator="\n")
                        if exp_mass_time is not None:
                            f.write("\n# Mass Data:\n")
                            pd.DataFrame({'time (min)': exp_mass_time,
                                          mass_col: exp_mass_data}).to_csv(f, index=False, lineterminator="\n")

                else:
                    # ── Sweep runs ───────────────────────────────────────────
                    # --- Sheet 1: summary table (one row per sweep point) ---
                    f.write("# SWEEP SUMMARY (one row per sweep point):\n")
                    summary_rows = []
                    for lbl, res in self.all_sweep_results:
                        c_ugm3 = self._model_conc_in(res, 'µg/m³')
                        row_dict = {
                            'sweep_value':         lbl,
                            'mass_bal_error (%)':  res['mass_bal_error'],
                            'solve_time (s)':      res['solve_time'],
                            'max_c_gas (ppbv)':    float(res['c_gas_ppbv'].max()),
                            'final_c_gas (ppbv)':  float(res['c_gas_ppbv'][-1]),
                            'max_c_gas (ug/m3)':   float(np.max(c_ugm3)),
                            'final_c_gas (ug/m3)': float(c_ugm3[-1]),
                            'max_delta_m (ng)':    float(res['delta_mass'].max()),
                            'final_delta_m (ng)':  float(res['delta_mass'][-1]),
                            'max_delta_m (ug)':    float(res['delta_mass'].max()) * 1e-3,
                            'final_delta_m (ug)':  float(res['delta_mass'][-1]) * 1e-3,
                            'max_delta_m (mg)':    float(res['delta_mass'].max()) * 1e-6,
                            'final_delta_m (mg)':  float(res['delta_mass'][-1]) * 1e-6,
                        }
                        r2c = _r2_conc(res)
                        r2m = _r2_mass(res)
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
                                          conc_col: exp_conc_data}).to_csv(f, index=False, lineterminator="\n")
                        if exp_mass_time is not None:
                            f.write("\n# Mass Data:\n")
                            pd.DataFrame({'time (min)': exp_mass_time,
                                          mass_col: exp_mass_data}).to_csv(f, index=False, lineterminator="\n")

            # Save each plot panel individually (publication-styled, 300 dpi)
            stem = filename.rsplit('.', 1)[0]
            labeled = (self.all_sweep_results if is_sweep
                       else [('Model', self.current_results)])
            plot_files = self._export_individual_plots(
                stem, labeled,
                exp_conc_time, exp_conc_data, exp_mass_time, exp_mass_data)

            n_runs = len(self.all_sweep_results) if is_sweep else 1
            plots_listed = "\n".join("  " + os.path.basename(p) for p in plot_files)
            messagebox.showinfo(
                "Export Success",
                f"{'Sweep' if is_sweep else 'Simulation'} data exported ({n_runs} run(s)):\n"
                f"{os.path.basename(filename)}\n\n"
                f"{len(plot_files)} individual plots saved alongside it:\n{plots_listed}"
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
            grid_scheme=self.grid_scheme.get(),
            grid_stretch=float(self.grid_stretch.get()),
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
            feed_tank_volume=float(self.feed_tank_volume.get()),
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
            cgas_init=self._cgas_init_uM(),
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
            'feed_tank_volume': p.get('feed_tank_volume', 0.0),
        }
        return run_simulation(
            p['tfinal'], temp_params, flow_params, cfeed_params,
            p['m'], p['R'], p['mSample'], p['rhoSample'],
            p['Vvessel'], p['MW_analyte'],
            p['src_params'], p['EaK'], p['K_ref'], p['D_ref'],
            p['EaD'], p['cD'], p['F'], p['c0free'], p.get('cgas_init', 0.0),
            p['N'], p['rtol'], p['atol'], p.get('Tref', 50),
            grid_scheme=p.get('grid_scheme', 'uniform'),
            grid_stretch=p.get('grid_stretch', 2.0)
        )

    def _toggle_opt_ui(self):
        """Enable or disable optimization widgets based on checkbox."""
        enabled = self.opt_enabled.get()
        s = 'normal' if enabled else 'disabled'
        s_ro = 'readonly' if enabled else 'disabled'
        for w in (self.opt_n_entry, self.opt_build_btn, self.opt_run_btn,
                  self.opt_maxiter, self.opt_nstarts):
            w.config(state=s)
        self.opt_stop_btn.config(state='disabled')   # only active while running
        self.opt_results_btn.config(
            state=s if (enabled and getattr(self, 'opt_multistart_results', None)) else 'disabled')
        self.opt_target.config(state=s_ro)
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

        param_names = [k for k, (attr, _) in self.SWEEP_PARAMS.items()
                       if attr not in self.OPT_EXCLUDED_ATTRS]
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

        # Concentration comparison unit (the unit the data were pasted in).
        # Snapshotted here on the main thread so the worker never touches Tk.
        conc_cmp_unit = self._conc_cmp_unit()
        mass_cmp_unit = self._mass_cmp_unit()

        has_conc = exp_conc_time is not None
        has_mass = exp_mass_time is not None

        if target == "Concentration (ppbv)" and not has_conc:
            messagebox.showerror(
                "No Experimental Data",
                "No concentration data found in the Experimental Data panel.\n\n"
                "Please enter time, value pairs (one per line) in the "
                f"Concentration box (units: {conc_cmp_unit}) before running optimization.")
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

        try:
            n_starts = max(1, int(self.opt_nstarts.get()))
        except ValueError:
            n_starts = 1

        # Snapshot the parameter mapping NOW (main thread): the worker thread
        # below must never read Tk widgets, so it works from this list instead
        # of calling combo.get() per evaluation.
        attr_list = [self.SWEEP_PARAMS[lab][0] for lab in selected_labels]

        def apply_trial(x_nat):
            """Widget-free version of _apply_opt_params for the worker thread."""
            p = dict(base)
            p['src_params'] = base['src_params'].copy()
            for val, attr_name in zip(x_nat, attr_list):
                if attr_name == 'R':
                    p['R'] = val * 1e-4
                    p['R_um'] = val
                elif attr_name.startswith('src'):
                    parts = attr_name.split('_')
                    src_idx = int(parts[0][3:]) - 1
                    field_off = {'c0': 0, 'A': 1, 'Ea': 2}[parts[1]]
                    p['src_params'][src_idx * 3 + field_off] = val
                else:
                    p[attr_name] = val
            return p

        self._opt_stop = False
        self.opt_run_btn.config(state='disabled')
        self.opt_stop_btn.config(state='normal')
        if hasattr(self, 'run_sim_btn'):
            self.run_sim_btn.config(state='disabled')
        self.opt_results_btn.config(state='disabled')
        self._opt_iter = 0
        self._opt_best_sse = np.inf
        self._opt_best_x = None
        self._opt_cur_start = 1
        self._opt_n_starts = n_starts
        self._opt_start_base = 0
        self._opt_start_best_sse = np.inf
        self._opt_start_best_x = None
        self._opt_eval_cap = maxiter          # objective raises _StopOpt at this count
        self._opt_status = ("Starting optimization...", 'orange')
        self._opt_outcome = None
        self._opt_error = None
        self._opt_done = False
        self._opt_ctx = dict(base=base, row_list=row_list, log_flags=log_flags,
                             maxiter=maxiter)
        self.opt_multistart_results = None
        self.opt_multistart_attrs = list(attr_list)

        class _StopOpt(Exception):
            pass

        def objective(x_opt):
            # Runs on the worker thread: must never touch Tk widgets.
            if self._opt_stop or self._opt_iter >= self._opt_eval_cap:
                raise _StopOpt()
            if any((xi < lo) or (xi > hi) for xi, (lo, hi) in zip(x_opt, bounds_search)):
                return 1e30

            x_nat = np.array([10 ** xi if lg else xi for xi, lg in zip(x_opt, log_flags)], dtype=float)
            try:
                p = apply_trial(x_nat)
                res = self._run_one(p)
            except _StopOpt:
                raise
            except Exception:
                return 1e30

            # Reject numerically invalid solutions: a large mass-balance error
            # means the spatial front was not resolved (common when cD is small
            # relative to the concentration scale), so the SSE is meaningless.
            mb_chk = res.get('mass_bal_error', 0.0)
            if not np.isfinite(mb_chk) or abs(mb_chk) > 5.0:
                return 1e30

            residuals = []
            if target in ("Concentration (ppbv)", "Both") and has_conc:
                sim_c = self._conc_from_uM(
                    res['c_gas_uM'], np.asarray(res['T_C']) + 273.15,
                    conc_cmp_unit, res['MW'])
                sim_interp = np.interp(exp_conc_time, res['t'], sim_c)
                norm = np.std(exp_conc_data) if np.std(exp_conc_data) > 0 else 1.0
                residuals.append(np.sum(((sim_interp - exp_conc_data) / norm) ** 2))
            if target in ("Mass (ng)", "Both") and has_mass:
                sim_m = self._mass_from_ng(res['delta_mass'], mass_cmp_unit)
                sim_interp = np.interp(exp_mass_time, res['t'], sim_m)
                norm = np.std(exp_mass_data) if np.std(exp_mass_data) > 0 else 1.0
                residuals.append(np.sum(((sim_interp - exp_mass_data) / norm) ** 2))

            self._opt_iter += 1
            sse = float(np.sum(residuals))
            if sse < self._opt_best_sse:
                self._opt_best_sse = sse
                self._opt_best_x = x_opt.copy()
            if sse < self._opt_start_best_sse:
                self._opt_start_best_sse = sse
                self._opt_start_best_x = x_opt.copy()
            ev_this = self._opt_iter - self._opt_start_base
            self._opt_status = (
                f"S{self._opt_cur_start}/{self._opt_n_starts} eval {ev_this}/{maxiter} "
                f"(tot {self._opt_iter}): SSE={sse:.4g} best={self._opt_best_sse:.4g}",
                'orange')
            return sse

        n_params = len(bounds_search)

        def to_natural(x_search):
            return np.array([10 ** xi if lg else xi
                             for xi, lg in zip(x_search, log_flags)], dtype=float)

        def r_squared(sim_t, sim_y, et, ey):
            try:
                si = np.interp(et, sim_t, sim_y)
                ss_tot = float(np.sum((ey - np.mean(ey)) ** 2))
                if ss_tot <= 0:
                    return None
                return 1.0 - float(np.sum((si - ey) ** 2)) / ss_tot
            except Exception:
                return None

        # ---- Starting points (search space) ----
        # Start 1: the current field values, clipped into the bounds, with the
        # bound midpoint substituted for any non-finite component (e.g. 'inf').
        x0_warm = []
        for attr_name, (lo, hi), lg in zip(attr_list, bounds_search, log_flags):
            if attr_name == 'R':
                val = base.get('R_um', np.nan)
            elif attr_name.startswith('src'):
                parts = attr_name.split('_')
                src_idx = int(parts[0][3:]) - 1
                field_off = {'c0': 0, 'A': 1, 'Ea': 2}[parts[1]]
                arr_idx = src_idx * 3 + field_off
                val = (base['src_params'][arr_idx]
                       if arr_idx < len(base['src_params']) else np.nan)
            else:
                val = base.get(attr_name, np.nan)
            if lg:
                val = np.log10(val) if (np.isfinite(val) and val > 0) else np.nan
            x0_warm.append(float(np.clip(val, lo, hi)) if np.isfinite(val)
                           else 0.5 * (lo + hi))
        start_points = [np.asarray(x0_warm, dtype=float)]
        if n_starts > 1:
            # Remaining starts: Latin-hypercube sample of the bounds for
            # space-filling coverage (reproducible: fixed seed).
            unit = qmc.LatinHypercube(d=n_params, seed=42).random(n_starts - 1)
            lo_arr = np.array([bd[0] for bd in bounds_search])
            hi_arr = np.array([bd[1] for bd in bounds_search])
            for u in unit:
                start_points.append(lo_arr + u * (hi_arr - lo_arr))

        def worker():
            records = []
            try:
                for k, x0 in enumerate(start_points, start=1):
                    if self._opt_stop:
                        break
                    self._opt_cur_start = k
                    self._opt_start_base = self._opt_iter
                    self._opt_start_best_sse = np.inf
                    self._opt_start_best_x = None
                    self._opt_eval_cap = self._opt_start_base + maxiter
                    status = 'converged'
                    try:
                        result = minimize(
                            objective,
                            np.asarray(x0, dtype=float),
                            method='Nelder-Mead',
                            options={'maxiter': maxiter, 'maxfev': maxiter,
                                     'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True},
                        )
                        if not result.success:
                            status = 'eval cap reached'
                    except _StopOpt:
                        status = 'terminated' if self._opt_stop else 'eval cap reached'

                    rec = dict(start=k,
                               x0_nat=to_natural(x0),
                               evals=self._opt_iter - self._opt_start_base,
                               status=status)
                    if self._opt_start_best_x is None:
                        rec['status'] = ('terminated' if self._opt_stop
                                         else 'failed (no valid simulations)')
                        rec.update(x_nat=None, sse=None, r2_conc=None,
                                   r2_mass=None, res=None)
                    else:
                        x_nat = to_natural(self._opt_start_best_x)
                        rec.update(x_nat=x_nat, sse=float(self._opt_start_best_sse))
                        # Bookkeeping re-run of this start's best fit (not
                        # counted against the budget): provides the curve for
                        # the overlay plot and the R\u00b2 values.
                        try:
                            res_k = self._run_one(apply_trial(x_nat))
                        except Exception:
                            res_k = None
                        rec['res'] = res_k
                        rec['r2_conc'] = (r_squared(
                                              res_k['t'],
                                              self._conc_from_uM(
                                                  res_k['c_gas_uM'],
                                                  np.asarray(res_k['T_C']) + 273.15,
                                                  conc_cmp_unit, res_k['MW']),
                                              exp_conc_time, exp_conc_data)
                                          if (res_k is not None and has_conc) else None)
                        rec['r2_mass'] = (r_squared(
                                              res_k['t'],
                                              self._mass_from_ng(res_k['delta_mass'], mass_cmp_unit),
                                              exp_mass_time, exp_mass_data)
                                          if (res_k is not None and has_mass) else None)
                    records.append(rec)
                self._opt_outcome = dict(records=records,
                                         terminated_early=bool(self._opt_stop),
                                         n_starts=n_starts)
            except Exception as exc:
                self._opt_error = exc
                self._opt_outcome = dict(records=records,
                                         terminated_early=True,
                                         n_starts=n_starts)
            finally:
                self._opt_done = True

        self._opt_thread = threading.Thread(target=worker, daemon=True)
        self._opt_thread.start()
        self.status_label.config(text="Starting optimization...", foreground='orange')
        self.root.after(150, self._opt_poll)

    def _opt_poll(self):
        """Refresh the status line while the optimization worker runs (main thread)."""
        try:
            text, color = self._opt_status
            self.status_label.config(text=text, foreground=color)
        except Exception:
            pass
        if not self._opt_done:
            self.root.after(150, self._opt_poll)
        else:
            self._opt_finish()

    def _opt_finish(self):
        """Apply the optimization outcome to the GUI (main thread only)."""
        self.opt_run_btn.config(state='normal')
        self.opt_stop_btn.config(state='disabled')
        if hasattr(self, 'run_sim_btn'):
            self.run_sim_btn.config(state='normal')

        outcome = self._opt_outcome or {}
        records = list(outcome.get('records', []))

        if self._opt_error is not None and not records:
            self.status_label.config(text=f"Optimization error: {self._opt_error}",
                                     foreground='red')
            messagebox.showerror("Optimization Error", str(self._opt_error))
            return

        valid = [r for r in records if r.get('sse') is not None]
        if not valid or self._opt_best_x is None:
            self.status_label.config(
                text='Optimization terminated before any valid evaluations completed.',
                foreground='blue' if self._opt_stop else 'red')
            return

        ctx = self._opt_ctx
        base, row_list = ctx['base'], ctx['row_list']
        terminated_early = bool(outcome.get('terminated_early'))

        best = min(valid, key=lambda r: r['sse'])
        x_nat_best = np.asarray(best['x_nat'], dtype=float)
        res_best = best.get('res')
        if res_best is None:
            try:
                res_best = self._run_one(self._apply_opt_params(x_nat_best, base, row_list))
            except Exception as exc:
                self.status_label.config(text=f"Best-fit re-run failed: {exc}",
                                         foreground='red')
                messagebox.showerror("Optimization Error", str(exc))
                return

        # Overlay every completed start's best-fit curve; flag the winner.
        labeled = []
        for r in valid:
            if r.get('res') is None:
                continue
            tag = "  (best)" if r is best else ""
            labeled.append((f"S{r['start']}: SSE={r['sse']:.3g}{tag}", r['res']))
        if not labeled:
            labeled = [("Best Fit", res_best)]
        self.current_results = res_best
        self.all_sweep_results = []
        self.update_plots(labeled)

        self._write_opt_results_to_gui(x_nat_best, row_list)

        self.opt_multistart_results = records
        self.opt_results_btn.config(state='normal')

        n_starts = outcome.get('n_starts', len(records))
        self.status_label.config(
            text=(f"{'Terminated' if terminated_early else 'Complete'} \u2014 "
                  f"best SSE={best['sse']:.4g} (start {best['start']}) | "
                  f"{len(valid)}/{n_starts} starts | total evals={self._opt_iter}"),
            foreground='blue' if terminated_early else 'green')
        if self._opt_error is not None:
            messagebox.showerror("Optimization Error",
                                 f"Stopped early due to an error:\n{self._opt_error}")
        self._show_multistart_window()

    def _show_multistart_window(self):
        """Show a per-start results table (initial guess, fitted parameters,
        SSE, R\u00b2) so the (non-)uniqueness of the fit can be assessed."""
        records = getattr(self, 'opt_multistart_results', None)
        if not records:
            messagebox.showinfo("Fit Results", "No optimization results available yet.")
            return
        if getattr(self, '_ms_win', None) is not None:
            try:
                if self._ms_win.winfo_exists():
                    self._ms_win.destroy()
            except Exception:
                pass

        attrs = getattr(self, 'opt_multistart_attrs', [])
        show_r2c = any(r.get('r2_conc') is not None for r in records)
        show_r2m = any(r.get('r2_mass') is not None for r in records)

        cols = ['Start', 'Status', 'Evals', 'SSE']
        if show_r2c:
            cols.append('R\u00b2 (conc)')
        if show_r2m:
            cols.append('R\u00b2 (mass)')
        for a in attrs:
            cols.append(f'{a} (init)')
            cols.append(f'{a} (fit)')

        def fmt(v, r2=False):
            if v is None:
                return '\u2014'
            return f'{v:.5f}' if r2 else f'{v:.6g}'

        def row_values(r):
            vals = [r['start'], r['status'], r['evals'], fmt(r.get('sse'))]
            if show_r2c:
                vals.append(fmt(r.get('r2_conc'), r2=True))
            if show_r2m:
                vals.append(fmt(r.get('r2_mass'), r2=True))
            for j in range(len(attrs)):
                vals.append(fmt(r['x0_nat'][j] if r.get('x0_nat') is not None else None))
                vals.append(fmt(r['x_nat'][j] if r.get('x_nat') is not None else None))
            return vals

        win = tk.Toplevel(self.root)
        win.title("Multi-Start Fit Results")
        win.geometry("1050x340")
        self._ms_win = win

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        tree = ttk.Treeview(frame, columns=cols, show='headings', height=12)
        for c in cols:
            tree.heading(c, text=c)
            w = 52 if c in ('Start', 'Evals') else (112 if '(init)' in c or '(fit)' in c else 96)
            tree.column(c, width=w, anchor=tk.W if c == 'Status' else tk.E, stretch=False)
        vsb = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        valid = [r for r in records if r.get('sse') is not None]
        best = min(valid, key=lambda r: r['sse']) if valid else None
        ordered = sorted(records,
                         key=lambda r: (r.get('sse') is None,
                                        r.get('sse') if r.get('sse') is not None else 0.0))
        tree.tag_configure('best', background='#d8f5d8')
        for r in ordered:
            tree.insert('', tk.END, values=row_values(r),
                        tags=('best',) if r is best else ())

        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, padx=6, pady=(0, 6))

        def copy_tsv():
            lines = ['\t'.join(cols)]
            for r in ordered:
                lines.append('\t'.join(str(v) for v in row_values(r)))
            self.root.clipboard_clear()
            self.root.clipboard_append('\n'.join(lines))
            self.status_label.config(text='Fit results table copied to clipboard.',
                                     foreground='green')

        ttk.Button(btns, text="Copy Table (TSV)", command=copy_tsv).pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

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

    def _fmt_sweep_val(self, attr_name, val, log_space):
        """Format a swept value for labels/status: whole numbers for integer
        attributes (e.g. grid points), else scientific for log sweeps and
        4-significant-figure for linear sweeps."""
        if attr_name in self.SWEEP_INT_ATTRS:
            return str(int(round(float(val))))
        return f"{val:.2e}" if log_space else f"{val:.4g}"

    def _mb_advisory(self, results):
        """Post-run mass-balance quality note.

        With the conservative (Kirchhoff-flux) scheme the mass-balance error
        reflects ODE-solver accuracy, not spatial resolution, so an elevated
        value points at the tolerances rather than the grid or cD.
        Returns (status_suffix, status_color_or_None).
        """
        notes = []
        color = None
        mb = abs(results.get('mass_bal_error', 0.0))
        if not np.isfinite(mb) or mb > 5.0:
            notes.append(f"MB INVALID ({mb:.2g}%) — check parameters / tighten rtol & atol")
            color = 'red'
        elif mb > 0.5:
            notes.append(f"MB {mb:.2g}% high — tighten solver tolerances (rtol/atol)")
            color = '#b36b00'
        suffix = ("  [" + "; ".join(notes) + "]") if notes else ""
        return suffix, color

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
                cmp_unit = self._conc_cmp_unit()
                mass_cmp_unit = self._mass_cmp_unit()
                model_conc_cmp = (self._model_conc_in(results, cmp_unit)
                                  if exp_conc_time is not None else None)
                r2_conc = self._compute_r_squared(results['t'], model_conc_cmp,
                                                  exp_conc_time, exp_conc_data)
                model_mass_cmp = (self._model_mass_in(results, mass_cmp_unit)
                                  if exp_mass_time is not None else None)
                r2_mass = self._compute_r_squared(results['t'], model_mass_cmp,
                                                  exp_mass_time, exp_mass_data)
                r2_parts = []
                if r2_conc is not None:
                    r2_parts.append(f"R²(conc)={r2_conc:.6f}")
                if r2_mass is not None:
                    r2_parts.append(f"R²(mass)={r2_mass:.6f}")
                r2_str = (" | " + " | ".join(r2_parts)) if r2_parts else ""

                adv_suffix, adv_color = self._mb_advisory(results)
                self.status_label.config(
                    text=(f"Complete! Solve time: {results['solve_time']:.2f}s | "
                          f"Mass balance error: {results['mass_bal_error']:.2e}%{r2_str}{adv_suffix}"),
                    foreground=adv_color or 'green')

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

                if attr_name == 'grid_stretch' and base.get('grid_scheme', 'uniform') == 'uniform':
                    raise ValueError(
                        "Grid Stretch only affects the 'tanh' and 'geometric' grids. "
                        "Select a non-uniform Grid Scheme before sweeping it (otherwise "
                        "every run would be identical)."
                    )

                if log_space:
                    sweep_vals = np.logspace(np.log10(start_val), np.log10(stop_val), n_pts)
                else:
                    sweep_vals = np.linspace(start_val, stop_val, n_pts)

                # Integer-valued sweeps (e.g. number of grid points) must be
                # whole numbers; round to integers and drop duplicates that the
                # rounding introduces, preserving sweep order. n_pts becomes the
                # number of distinct runs actually performed.
                if attr_name in self.SWEEP_INT_ATTRS:
                    seen, uniq = set(), []
                    for v in sweep_vals:
                        iv = int(round(float(v)))
                        if iv not in seen:
                            seen.add(iv)
                            uniq.append(iv)
                    sweep_vals = np.array(uniq, dtype=float)
                    n_pts = len(sweep_vals)

                cmap = plt.get_cmap(self.sweep_cmap.get())
                colors = [cmap(i / max(n_pts - 1, 1)) for i in range(n_pts)]

                labeled_results = []
                total_t0 = time.time()
                for idx, val in enumerate(sweep_vals):
                    val_lbl = self._fmt_sweep_val(attr_name, val, log_space)
                    self.status_label.config(
                        text=f"Sweep {idx + 1}/{n_pts}: {attr_name} = {val_lbl}",
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
                    elif attr_name in self.SWEEP_INT_ATTRS:
                        p[attr_name] = int(round(float(val)))
                    else:
                        p[attr_name] = val

                    res = self._run_one(p)
                    self.status_label.config(
                        text=(f"Sweep {idx + 1}/{n_pts}: {attr_name} = {val_lbl} "
                              f"done | MB {res['mass_bal_error']:.1e}%"),
                        foreground='orange')
                    self.root.update()
                    res['_sweep_color'] = colors[idx]
                    labeled_results.append((val_lbl, res))

                elapsed_total = time.time() - total_t0
                self.current_results = labeled_results[-1][1]
                self.all_sweep_results = labeled_results
                self.update_plots(labeled_results, sweep_label=param_label)
                # Mass-balance summary across the sweep
                mb_abs = np.array([abs(r['mass_bal_error']) for _, r in labeled_results])
                finite_mb = np.isfinite(mb_abs)
                i_worst = int(np.argmax(np.where(finite_mb, mb_abs, np.inf))) \
                    if np.any(finite_mb) else int(np.argmax(~finite_mb))
                mb_max = mb_abs[i_worst]
                mb_txt = (f" | max |MB|: {mb_max:.2e}% "
                          f"({attr_name} = {labeled_results[i_worst][0]})")
                if not np.isfinite(mb_max) or mb_max > 5.0:
                    mb_txt += " — INVALID, check parameters / tighten rtol & atol"
                    sweep_color = 'red'
                elif mb_max > 0.5:
                    mb_txt += " — tighten solver tolerances (rtol/atol)"
                    sweep_color = '#b36b00'
                else:
                    sweep_color = 'green'
                self.status_label.config(
                    text=f"Sweep complete! {n_pts} runs in {elapsed_total:.1f}s{mb_txt}",
                    foreground=sweep_color)

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground='red')
            messagebox.showerror('Simulation Error', str(e))

    def _time_axis_limits(self, labeled_results):
        """Return (use_log_time, left_limit, right_limit) for the time axis.

        The user-entered min/max apply to BOTH linear and log scales; either
        may be blank (-> autoscale that side). On a log axis a non-positive or
        blank min falls back to the smallest positive output time so the t = 0
        point is excluded cleanly, and a max is honoured only if it lies above
        the resolved lower limit. Inconsistent entries (min >= max) drop the
        max override rather than inverting the axis.
        """
        log_t = bool(getattr(self, 'logtime_var', None) and self.logtime_var.get())

        def _parse(entry):
            try:
                return float(entry.get().strip())
            except (ValueError, AttributeError):
                return None
        vmin = _parse(getattr(self, 'time_min', None))
        vmax = _parse(getattr(self, 'time_max', None))

        allt = (np.concatenate([r['t'] for _, r in labeled_results])
                if labeled_results else np.array([]))
        tmax_data = float(allt.max()) if allt.size else None

        left = right = None
        if log_t:
            pos = allt[allt > 0]
            left = float(pos.min()) if pos.size else None
            if vmin is not None and vmin > 0 and (tmax_data is None or vmin < tmax_data):
                left = vmin
            if vmax is not None and vmax > 0 and (left is None or vmax > left):
                right = vmax
        else:
            left = vmin
            right = vmax
            if left is not None and right is not None and left >= right:
                right = None
        return log_t, left, right

    def _refresh_plots(self):
        """Re-render the most recent plot (e.g. after toggling the log-time
        axis) without re-running the simulation."""
        last = getattr(self, '_last_labeled_results', None)
        if last:
            self.update_plots(last, getattr(self, '_last_sweep_label', None))

    def update_plots(self, labeled_results, sweep_label=None):
        """
        Plot one or more simulation results.
        labeled_results : list of (label_str, results_dict)
        sweep_label     : human-readable name of the swept parameter (or None)
        """
        # Remember inputs so display-only toggles (log-time axis) can re-render.
        self._last_labeled_results = labeled_results
        self._last_sweep_label = sweep_label

        for ax in [self.ax1, self.ax2, self.ax3,
                   self.ax4, self.ax5, self.ax6, self.ax7]:
            ax.clear()

        is_sweep = len(labeled_results) > 1

        # Concentration display unit (plot) and comparison unit (R²).
        plot_unit = self._conc_plot_unit()
        cmp_unit = self._conc_cmp_unit()
        data_unit = cmp_unit   # experimental concentrations are pasted in this unit

        # Mass-change display unit (plot) and comparison unit (R²).
        mass_plot_unit = self._mass_plot_unit()
        mass_cmp_unit = self._mass_cmp_unit()

        # Experimental data (shown only once, on top)
        exp_conc_time, exp_conc_data = self.parse_experimental_data(
            self.conc_data_text, self.conc_interp_var, self.conc_interp_n, self.conc_interp_method)
        exp_mass_time, exp_mass_data = self.parse_experimental_data(
            self.mass_data_text, self.mass_interp_var, self.mass_interp_n, self.mass_interp_method)

        # Experimental concentrations converted to the plot unit for overlay.
        exp_conc_plot = (self._exp_conc_convert(exp_conc_time, exp_conc_data,
                                                data_unit, plot_unit)
                         if exp_conc_time is not None else None)
        # Experimental masses converted to the mass plot unit for overlay.
        exp_mass_plot = (self._exp_mass_convert(exp_mass_data, mass_cmp_unit, mass_plot_unit)
                         if exp_mass_time is not None else None)

        # Colour / linewidth logic
        def _colour(res, default):
            return res.get('_sweep_color', default)

        lw_main = 1.5 if is_sweep else 2.0

        for label, res in labeled_results:
            t    = res['t']
            col  = _colour(res, None)   # None → use matplotlib default cycle
            kw   = dict(linewidth=lw_main, label=label, color=col) if col else dict(linewidth=lw_main, label=label)

            self.ax1.plot(t, res['T_C'],          **kw)
            self.ax2.plot(t, self._model_feed_in(res, plot_unit), **kw)
            self.ax3.plot(t, res['Q'],             **kw)
            self.ax4.semilogy(t, res['D'],   **kw)
            self.ax5.plot(t, res['S'],             **kw)
            self.ax6.plot(t, self._model_mass_in(res, mass_plot_unit), **kw)
            self.ax7.plot(t, self._model_conc_in(res, plot_unit), **kw)

        # Time-axis scale and user-set limits (apply to linear and log).
        log_t, left_limit, right_limit = self._time_axis_limits(labeled_results)

        # ── Axis labels / titles ──────────────────────────────────────────────
        for ax, xlabel, ylabel, title in [
            (self.ax1, 'Time (min)', 'Temperature (°C)',           'Temperature'),
            (self.ax2, 'Time (min)', self._feed_axis_label(plot_unit),  'Feed Concentration'),
            (self.ax3, 'Time (min)', 'Flow Rate (ml/min)',         'Flow Rate'),
            (self.ax4, 'Time (min)', 'Diffusivity (cm²/s)',        'Diffusivity at Surface'),
            (self.ax5, 'Time (min)', 'S (cm³@STP/atm/cm³)',        'Solubility'),
            (self.ax6, 'Time (min)', self._mass_axis_label(mass_plot_unit), 'Sample Mass Change'),
            (self.ax7, 'Time (min)', self._conc_axis_label(plot_unit), 'Headspace Gas Concentration'),
        ]:
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)
            if log_t:
                ax.set_xscale('log')
                ax.grid(True, which='both', alpha=0.3)
            if left_limit is not None or right_limit is not None:
                ax.set_xlim(left=left_limit, right=right_limit)

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

        # Optional log scale for the headspace gas concentration (display toggle).
        # c_gas can be 0 (e.g. at t = 0); a log axis masks non-positive values,
        # so anchor the lower limit to the smallest positive concentration.
        if getattr(self, 'logconc_var', None) and self.logconc_var.get():
            self.ax7.set_yscale('log')
            self.ax7.grid(True, which="both", ls="-", alpha=0.3)
            all_c = np.concatenate([self._model_conc_in(r, plot_unit)
                                    for _, r in labeled_results])
            positive_c = all_c[np.isfinite(all_c) & (all_c > 0)]
            if positive_c.size:
                cmin, cmax = float(positive_c.min()), float(positive_c.max())
                self.ax7.set_ylim(bottom=cmin if cmax > cmin else cmin / 10)

        # Horizontal zero line on mass plot
        self.ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)

        # Experimental data overlays
        if exp_mass_time is not None:
            self.ax6.plot(exp_mass_time, exp_mass_plot, 'ko',
                          markersize=5, label='Experimental', zorder=10)
        if exp_conc_time is not None:
            self.ax7.plot(exp_conc_time, exp_conc_plot, 'ko',
                          markersize=5, label='Experimental', zorder=10)

        # R² annotations (single-run only); comparison done in the data unit.
        if not is_sweep and len(labeled_results) == 1:
            _, res0 = labeled_results[0]
            model_cmp = (self._model_conc_in(res0, cmp_unit)
                         if exp_conc_time is not None else None)
            r2_conc = self._compute_r_squared(
                res0['t'], model_cmp, exp_conc_time, exp_conc_data)
            model_mass_cmp = (self._model_mass_in(res0, mass_cmp_unit)
                              if exp_mass_time is not None else None)
            r2_mass = self._compute_r_squared(
                res0['t'], model_mass_cmp, exp_mass_time, exp_mass_data)
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