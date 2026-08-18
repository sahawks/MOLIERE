# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Lawrence Livermore National Security, LLC
# LLNL-CODE-2017385
# CP 2025-187
# Author: Steven A. Hawks

"""Tkinter GUI for the MOLIERE outgassing model."""

import copy
import json
import os
import platform
import tempfile
import threading
import time
import traceback
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timezone
from tkinter import filedialog, messagebox, ttk
from types import MappingProxyType

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix
from scipy.stats import qmc

LLNL_RELEASE_ID = "LLNL-CODE-2017385"
APP_TITLE = "Outgassing Model Simulator"
APP_VERSION = "12.0"
CONFIG_SCHEMA_VERSION = 2
MAX_UI_SOURCES = 20
MAX_GRID_POINTS = 10000
MAX_SWEEP_POINTS = 200
MAX_RESAMPLE_POINTS = 20000
MAX_OPT_STARTS = 50
MAX_OPT_EVALS = 5000
MAX_OPT_TOTAL_EVALS = 20000
MAX_SOLUTION_ELEMENTS = 50000000
MAX_JACOBIAN_CONDITION = 1e12
MAX_FORCING_BREAKPOINTS = 20000


@dataclass(frozen=True)
class ParameterSpec:
    """Single source of truth for fit defaults and transformations."""

    lower: float
    upper: float
    log_default: bool


PARAMETER_SPECS = {
    'T0': ParameterSpec(-100.0, 400.0, False),
    'Tfinal': ParameterSpec(-100.0, 500.0, False),
    'RR': ParameterSpec(0.01, 100.0, True),
    'Q': ParameterSpec(0.001, 1000.0, True),
    'feed_tank_volume': ParameterSpec(0.0, 10000.0, False),
    'R': ParameterSpec(1.0, 10000.0, True),
    'mSample': ParameterSpec(0.001, 10000.0, True),
    'rhoSample': ParameterSpec(0.05, 20.0, True),
    'Vvessel': ParameterSpec(0.01, 100000.0, True),
    'K_ref': ParameterSpec(1e-4, 1e8, True),
    'EaK': ParameterSpec(-250.0, 250.0, False),
    'D_ref': ParameterSpec(1e-15, 1e-3, True),
    'EaD': ParameterSpec(0.0, 300.0, False),
    'cD': ParameterSpec(1e-8, 1e8, True),
    'F': ParameterSpec(1e-12, 10.0, True),
    'c0free': ParameterSpec(1e-12, 1e9, True),
    'base_conc': ParameterSpec(0.0, 1e9, False),
    'src_c0': ParameterSpec(1e-12, 1e9, True),
    'src_A': ParameterSpec(1e-12, 1e18, True),
    'src_Ea': ParameterSpec(0.0, 400.0, False),
}


@dataclass(frozen=True)
class ExperimentalSeries:
    raw_time: object = None
    raw_values: object = None
    raw_sigma: object = None
    plot_time: object = None
    plot_values: object = None
    unit: str = ''
    raw_text: str = ''
    diagnostics: tuple = ()
    duplicate_policy: str = 'Retain replicates'


@dataclass(frozen=True)
class ExperimentalSnapshot:
    concentration: ExperimentalSeries
    mass: ExperimentalSeries


@dataclass(frozen=True)
class FitParameterSnapshot:
    label: str
    attribute: str
    lower: float
    upper: float
    use_log: bool


@dataclass(frozen=True)
class RunRecord:
    mode: str
    label: str
    params: dict
    results: dict
    experimental: ExperimentalSnapshot
    started_at_utc: str
    completed_at_utc: str
    sweep_parameter: str = None
    sweep_attribute: str = None
    sweep_value: float = None
    versions: dict = None


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _runtime_versions():
    return {
        'application': APP_VERSION,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'pandas': pd.__version__,
        'matplotlib': matplotlib.__version__,
    }

RESAMPLE_TOOLTIP = (
    "Resample the pasted data onto N equally spaced times for display only.\n\n"
    "The optimizer always evaluates residuals at the original measurement "
    "times, so interpolation does not manufacture observations, change target "
    "weighting, or inflate the apparent sample size used for uncertainty "
    "estimates.\n\n"
    "Use this option only to draw a smoother guide through ragged plot data."
)

# Selectable units for the headspace gas concentration (plotting / experimental
# input).  Volumetric mixing ratios (ppbv, ppmv) depend on gas temperature;
# mass concentrations (µg/m³, mg/m³) do not.  Conversions go through the model's
# fundamental molar concentration c_gas (µM); see OutgassingGUI._conc_from_uM.
CONC_UNITS = ['ppbv', 'ppmv', 'µg/m³', 'mg/m³']

# Sample mass-change display/input units.  The model computes the cumulative
# mass change in nanograms (ng); µg and mg are simple decimal rescalings
# (1 µg = 1e3 ng, 1 mg = 1e6 ng).  See OutgassingGUI._mass_from_ng.
MASS_UNITS = ['ng', 'µg', 'mg']
DUPLICATE_POLICIES = ('Retain replicates', 'Average replicates', 'Reject duplicates')

# ---- Plot-pane layout -------------------------------------------------------
# Height of the gas-concentration row relative to one row of supporting panels.
# 1.0 would make all three rows equal; 2.4 gives concentration ~55 % of the
# figure height against ~23 % for each of the two supporting rows.
CONC_PANEL_HEIGHT_RATIO = 2.4
# The experimental-data panel is docked to the right of the canvas.  It keeps a
# fixed width so that widening the window grows the plots, not the text boxes.
EXP_PANEL_MIN_WIDTH = 300
EXP_TEXT_WIDTH = 30
EXP_TEXT_MIN_LINES = 8
# Floor on the plot canvas, both when the controls/plots divider is first placed
# and when the window is shrunk.  Below EXP_PANEL_MIN_WIDTH + this, the data
# panel gives up width first (down to EXP_PANEL_FLOOR_WIDTH) so the canvas is
# never crushed to nothing.
PLOT_CANVAS_MIN_WIDTH = 620
EXP_PANEL_FLOOR_WIDTH = 170
# Hard floor on the canvas width.  The controls/plots divider is pushed left
# rather than letting the canvas fall below this.
PLOT_CANVAS_HARD_MIN = 420

# ============================================================================
# Core simulation functions (MOLIERE v13 -- conservative Kirchhoff-flux
# finite-volume scheme; exactly mass-conservative for any cD)
#
# v13 changes (identical to notebook v13; the two solvers are verified to agree
# bit-for-bit on the default, short-pulse, finite-F, small-cD and feed-staircase
# cases):
#   * Breakpoint-split time integration replaces max_step = tfinal/10.  The
#     forcing is piecewise smooth but not smooth, and BDF's error estimate is
#     not valid across a discontinuity -- it can fail optimistically, returning
#     success = True on a badly wrong trajectory (measured: 647 % too high on a
#     0.5 min purge pulse, while reporting a 7e-4 % mass-balance error).
#   * Net and absolute boundary-flux integrals are carried as ODE states instead
#     of being recovered by quadrature over the output grid.  This gives an
#     exact conservation residual and a stable normalization even when large
#     opposing inflow/outflow nearly cancel.
#   * Analytical Jacobian added for the finite-F constant-D case; the per-call
#     COO -> CSC rebuild is replaced by an O(nnz) update.
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
        # np.minimum guards the floating-point edge case where t_act/step_time
        # rounds up to exactly 2*n_steps at the end of the active window, which
        # would make the falling branch return base_conc - delta.
        cs = np.minimum((t_act / step_time).astype(int), 2 * n_steps - 1)
        rising = cs < n_steps
        result[active] = np.where(rising, base_conc + (cs + 1) * delta,
                                  base_conc + (2 * n_steps - cs - 1) * delta)
    return float(result[0]) if scalar else result



def feed_tank_segments(tfinal, Q_val, V_feed, n_steps, delta, step_time,
                       base_conc, hold_time_initial, hold_time_final,
                       max_segments=MAX_FORCING_BREAKPOINTS):
    """Precompute the independent CSTR roll-over of the staircase feed.

    The feed tank intentionally uses its own continuously operating carrier
    flow ``Q_val``.  It is therefore independent of the chamber's on/off flow
    schedule.  A finite volume smooths the ideal staircase with
    ``tau = V_feed / Q_val``.
    """
    if V_feed <= 0 or Q_val <= 0 or n_steps <= 0 or delta == 0:
        return None
    total = hold_time_initial + 2 * n_steps * step_time + hold_time_final
    if total <= 0:
        return None
    n_cycles = int(np.floor(float(tfinal) / total)) + 1
    estimated = n_cycles * (2 * int(n_steps) + 2) + 2
    if estimated > max_segments:
        raise ValueError(
            "The requested feed profile would require approximately "
            f"{estimated:,} feed-tank segments, exceeding the supported limit "
            f"of {max_segments:,}. Increase the step time, shorten the "
            "simulation, reduce the number of steps, or use an averaged feed."
        )

    edges = [0.0, float(tfinal)]
    for p in range(n_cycles):
        b = p * total
        edges.append(b)
        for j in range(0, 2 * n_steps + 1):
            edges.append(b + hold_time_initial + j * step_time)
        edges.append(b + total)
    edges = np.unique(np.clip(np.asarray(edges, dtype=float), 0.0, float(tfinal)))
    if edges.size > max_segments:
        raise ValueError(
            f"The feed profile contains {edges.size:,} unique segment edges, "
            f"exceeding the supported limit of {max_segments:,}."
        )

    mids = 0.5 * (edges[:-1] + edges[1:])
    level = np.atleast_1d(feed_conc_vec(
        mids, n_steps, delta, step_time, base_conc,
        hold_time_initial, hold_time_final))
    rate = Q_val / V_feed
    seg_left = edges[:-1]
    y_start = np.empty_like(level)
    y = float(base_conc)
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
    if int(N) != N or int(N) < 3:
        raise ValueError("Grid Points (N) must be an integer of at least 3.")
    if int(N) > MAX_GRID_POINTS:
        raise ValueError(
            f"Grid Points (N) cannot exceed {MAX_GRID_POINTS:,}; use a grid "
            "convergence study rather than an unbounded grid."
        )
    if grid_scheme not in GRID_SCHEMES:
        raise ValueError("Grid Scheme must be one of: " + ", ".join(GRID_SCHEMES) + ".")
    if grid_scheme in ('tanh', 'geometric'):
        if not np.isfinite(grid_stretch) or grid_stretch <= 0:
            raise ValueError(
                "Grid Stretch must be a finite value > 0 for the "
                f"'{grid_scheme}' scheme. For a geometric grid, values > 1 "
                "refine the sample surface and values < 1 refine the origin."
            )

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
    sample_volume = mSample * 1e-3 / rhoSample
    if Vvessel <= sample_volume:
        raise ValueError(
            "Vessel volume must exceed the sample volume so the headspace is positive.")
    if MW_analyte <= 0:
        raise ValueError("MW_analyte must be > 0.")
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
    for label, value in [('T0', temp_params['T0']),
                         ('Tfinal', temp_params['Tfinal']), ('Tref', Tref)]:
        if value <= -273.15:
            raise ValueError(f"{label} must be greater than -273.15 °C.")
    if (not np.isclose(temp_params['Tfinal'], temp_params['T0'])
            and temp_params['RR'] <= 0):
        raise ValueError("Ramp rate RR must be > 0 when T0 and Tfinal differ.")

    ns = cfeed_params['n_steps']
    if int(ns) != ns or int(ns) < 0:
        raise ValueError("Number of feed-concentration steps must be a nonnegative integer.")
    if ns > 0 and cfeed_params['step_time'] <= 0:
        raise ValueError("Step Time must be > 0 when Number of Steps is positive.")
    if cfeed_params['hold_time_initial'] < 0 or cfeed_params['hold_time_final'] < 0:
        raise ValueError("Feed hold times must be >= 0.")
    total_feed_time = (cfeed_params['hold_time_initial']
                       + 2 * int(ns) * cfeed_params['step_time']
                       + cfeed_params['hold_time_final'])
    if total_feed_time <= 0:
        raise ValueError(
            "The total feed profile duration (initial hold + 2 x steps x step "
            "time + final hold) must be > 0."
        )
    if cfeed_params.get('feed_tank_volume', 0.0) < 0:
        raise ValueError("Feed Tank Volume must be >= 0.")
    feed_minimum = min(
        cfeed_params['base_conc'],
        cfeed_params['base_conc'] + int(ns) * cfeed_params['delta'])
    if feed_minimum < 0:
        raise ValueError(
            "The feed staircase would contain a negative concentration; adjust "
            "base_conc, delta, or n_steps.")

    if np.isneginf(cD):
        raise ValueError("Plasticizer power cD may be finite positive or +inf, not -inf.")
    if np.isneginf(F):
        raise ValueError("Surface mass-transfer coefficient F may be finite or +inf, not -inf.")
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
        'c0free': c0free,
        'cgas_init': cgas_init,
        'Tref': Tref,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if not (np.isfinite(cD) or np.isposinf(cD)):
        raise ValueError("cD must be finite or +inf.")
    if not (np.isfinite(F) or np.isposinf(F)):
        raise ValueError("F must be finite or +inf.")

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

def append_quadrature_row(sp_csc, is_inf_F, N):
    """Extend sparsity with net and absolute boundary-flux integrals.

    q_net integrates Q(c_gas-c_feed)/vSample and q_abs integrates its absolute
    value.  The latter supplies a stable conservation normalization even when
    large opposing inflow and outflow nearly cancel.  Both rows couple only to
    the gas concentration and to their own diagnostic state.
    """
    k = sp_csc.shape[0]
    gas_col = N - 1 if is_inf_F else N
    sp = lil_matrix((k + 2, k + 2), dtype=np.int8)
    sp[:k, :k] = sp_csc
    for row in (k, k + 1):
        sp[row, row] = 1
        sp[row, gas_col] = 1
    return csc_matrix(sp)


def build_csc_updater(rows, cols, shape):
    """Return (indices, indptr, perm) so a CSC matrix with a fixed sparsity
    pattern can be rebuilt in O(nnz) from a value array, without the COO sort
    that csc_matrix(coo_matrix(...)) performs on every call.

    Usage:  csc_matrix((vals[perm], indices, indptr), shape=shape)
    """
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    if len(set(zip(rows.tolist(), cols.tolist()))) != rows.size:
        raise ValueError('duplicate (row, col) entries would be summed by COO -> CSC')
    probe = coo_matrix((np.arange(rows.size, dtype=np.float64), (rows, cols)),
                       shape=shape).tocsc()
    return probe.indices.copy(), probe.indptr.copy(), probe.data.astype(np.int64)


# ============================================================================
# Breakpoint-split time integration
# ============================================================================
#
# The imposed forcing is piecewise smooth but NOT smooth: the flow square wave
# and the feed staircase step, and dT/dt jumps at the start and end of the ramp
# (which enters the right-hand side through the dK/dt repartitioning term).
# BDF builds its error estimate on a polynomial history that assumes smoothness
# over the step, so a discontinuity inside a step makes the estimate unreliable
# -- and unreliable in the optimistic direction, i.e. the solver can return
# success = True on a badly wrong trajectory.  Capping max_step only reduces the
# probability of that; it does not remove it, and it forces small steps through
# the long quiescent stretches too.  Instead the integration is performed one
# smooth segment at a time.  Inside a segment the right-hand side is analytic,
# BDF's error control is valid, and no max_step is required.



def forcing_breakpoints(tfinal, temp_params, flow_params, cfeed_params,
                        extra=None, max_breaks=MAX_FORCING_BREAKPOINTS):
    """Return every forcing discontinuity in ``[0, tfinal]``.

    Breakpoints are never silently discarded.  A request exceeding the
    supported limit fails with an actionable message instead of allowing BDF
    to integrate across unresolved square-wave discontinuities.
    """
    tfinal = float(tfinal)
    candidates = [np.array([0.0, tfinal], dtype=float)]

    tEq = float(flow_params['tEq'])
    tFlush = float(flow_params['tFlush'])
    period = tEq + tFlush
    if float(flow_params['Q']) != 0.0 and tEq > 0.0 and tFlush > 0.0:
        n_cyc = int(np.floor(tfinal / period)) + 1
        estimated = 2 * (n_cyc + 1)
        if estimated > max_breaks:
            raise ValueError(
                "The requested flow square wave contains approximately "
                f"{estimated:,} discontinuities, exceeding the supported limit "
                f"of {max_breaks:,}. Increase the flow period, shorten the "
                "simulation, or use an averaged-flow model."
            )
        k = np.arange(n_cyc + 1, dtype=float)
        candidates.extend((k * period, k * period + tEq))

    ns = int(cfeed_params['n_steps'])
    st = float(cfeed_params['step_time'])
    hti = float(cfeed_params['hold_time_initial'])
    htf = float(cfeed_params['hold_time_final'])
    total = hti + 2 * ns * st + htf
    if ns > 0 and float(cfeed_params['delta']) != 0.0 and st > 0.0 and total > 0.0:
        n_cyc = int(np.floor(tfinal / total)) + 1
        estimated = (n_cyc + 1) * (2 * ns + 2)
        if estimated > max_breaks:
            raise ValueError(
                "The requested feed staircase contains approximately "
                f"{estimated:,} discontinuities, exceeding the supported limit "
                f"of {max_breaks:,}. Increase the step time, shorten the "
                "simulation, reduce the number of steps, or use an averaged feed."
            )
        j = np.arange(2 * ns + 1, dtype=float)
        feed_parts = []
        for k in range(n_cyc + 1):
            base = k * total
            feed_parts.append(base + hti + j * st)
            feed_parts.append(np.array([base + total]))
        candidates.append(np.concatenate(feed_parts))

    temp_edges = []
    dt1 = float(temp_params['dt1'])
    T0 = float(temp_params['T0'])
    T1 = float(temp_params['Tfinal'])
    RR = float(temp_params['RR'])
    if dt1 > 0.0:
        temp_edges.append(dt1)
    if not np.isclose(T1, T0) and RR != 0.0:
        temp_edges.append(dt1 + abs(T1 - T0) / abs(RR))
    if temp_edges:
        candidates.append(np.asarray(temp_edges, dtype=float))

    if extra is not None:
        extra_arr = np.asarray(extra, dtype=float).ravel()
        if not np.all(np.isfinite(extra_arr)):
            raise ValueError("Additional forcing breakpoints must be finite.")
        if extra_arr.size > max_breaks:
            raise ValueError(
                f"The additional breakpoint table contains {extra_arr.size:,} "
                f"entries, exceeding the supported limit of {max_breaks:,}."
            )
        candidates.append(extra_arr)

    b = np.concatenate(candidates)
    b = b[(b >= 0.0) & (b <= tfinal)]
    b.sort()
    if b.size:
        scale = np.maximum(1.0, np.maximum(np.abs(b[:-1]), np.abs(b[1:])))
        tol = 32.0 * np.finfo(float).eps * scale
        keep = np.concatenate(([True], np.diff(b) > tol))
        b = b[keep]
    if b.size == 0 or b[0] != 0.0:
        b = np.insert(b, 0, 0.0)
    else:
        b[0] = 0.0
    if b[-1] != tfinal:
        b = np.append(b, tfinal)
    else:
        b[-1] = tfinal
    if b.size > max_breaks:
        raise ValueError(
            f"The forcing profile contains {b.size:,} unique breakpoints, "
            f"exceeding the supported limit of {max_breaks:,}. Increase the "
            "forcing periods, shorten the run, or use an averaged model."
        )
    return b


def solve_forced_ivp(rhs, tfinal, y0, breaks, rtol, atol,
                     min_out_per_segment=8, method='BDF',
                     max_solution_elements=MAX_SOLUTION_ELEMENTS, **solver_kw):
    """Integrate 0 -> tfinal one smooth segment at a time.

    Returns (t, Y, stats).  `t` is strictly increasing; `Y` has shape
    (n_states, len(t)); `stats` carries nfev/njev/nlu, n_segments and seg_edges
    (the index of each breakpoint in `t`, so a discontinuous integrand can be
    integrated segment by segment rather than straddling a jump).
    """
    y = np.asarray(y0, dtype=float)
    t_out, y_out = [], []
    nfev = njev = nlu = 0
    stored_points = 0
    for a, b in zip(breaks[:-1], breaks[1:]):
        if not (b > a):
            continue
        res = solve_ivp(rhs, (a, b), y, method=method, rtol=rtol, atol=atol,
                        dense_output=True, **solver_kw)
        if not res.success:
            raise RuntimeError(
                f"solve_ivp failed on segment [{a:g}, {b:g}] min: {res.message}")
        nfev += res.nfev
        njev += res.njev
        nlu += res.nlu

        # Output grid: the solver's own accepted steps, minus the
        # near-coincident points a BDF restart leaves at the segment start.
        seg_tol = 1e-10 * max(b - a, 1e-30)
        ts = res.t[np.concatenate((np.diff(res.t) > seg_tol, [True]))]
        if ts.size < min_out_per_segment:
            ts = np.unique(np.concatenate(
                (ts, np.linspace(a, b, min_out_per_segment))))
        ts[0], ts[-1] = a, b
        projected_points = stored_points + int(ts.size)
        if y.size * projected_points > int(max_solution_elements):
            estimated_mb = y.size * projected_points * 8.0 / (1024.0 ** 2)
            raise RuntimeError(
                'The adaptive solver output grew beyond the supported '
                f'workload (about {estimated_mb:.0f} MiB). Reduce Grid Points '
                'or forcing events, shorten the simulation, or relax '
                'unnecessarily strict solver tolerances.')
        t_out.append(ts)
        y_out.append(res.sol(ts))
        stored_points = projected_points
        y = res.y[:, -1]

    t = np.concatenate(t_out)
    Y = np.hstack(y_out)
    keep = np.concatenate((np.diff(t) > 0.0, [True]))
    t, Y = t[keep], Y[:, keep]
    return t, Y, dict(nfev=nfev, njev=njev, nlu=nlu,
                      n_segments=len(breaks) - 1,
                      seg_edges=np.searchsorted(t, breaks))


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
    is_inf_F = np.isposinf(F)
    is_inf_cD = np.isposinf(cD)

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

    # Two diagnostic states carry the net and absolute purge integrals:
    #   dq_net/dt = Q(t) * (c_gas - c_feed) / vSample
    #   dq_abs/dt = abs(dq_net/dt)
    # The absolute integral stabilizes the relative conservation metric when
    # substantial inflow and outflow nearly cancel over the full experiment.
    q_idx = (N + num_srcs) if is_inf_F else (N + 1 + num_srcs)
    qa_idx = q_idx + 1
    n_total = qa_idx + 1
    inv_vSample = 1.0 / vSample

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
        cs = min(int((tc - hold_time_initial) / step_time), 2 * n_steps - 1)
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
            boundary_rate = Q * (cs[-1] / K - cfeed) * inv_vSample
            dcdt[q_idx] = boundary_rate
            dcdt[qa_idx] = abs(boundary_rate)
            return dcdt

        if not is_inf_cD:
            sparsity = append_quadrature_row(
                generate_jacobian_sparsity(N, num_srcs, True), True, N)
            return rhs, dict(jac_sparsity=sparsity)

        # ---- Analytical Jacobian for the constant-D case ----
        idx_int = np.arange(1, N - 1)
        jac_rows = np.concatenate([
            [0, 0],                                          # center
            idx_int, idx_int, idx_int,                        # interior sub/diag/super
            [N - 1, N - 1],                                  # surface (2-pt flux)
            np.arange(src_off, src_off + num_srcs),          # source self
            np.tile(np.arange(N), num_srcs),                 # source -> all solid rows
            [q_idx, q_idx],                                  # net purge integral
            [qa_idx, qa_idx],                                # absolute purge integral
        ]).astype(np.int32)
        jac_cols = np.concatenate([
            [0, 1],
            idx_int - 1, idx_int, idx_int + 1,
            [N - 2, N - 1],
            np.arange(src_off, src_off + num_srcs),
            np.repeat(np.arange(src_off, src_off + num_srcs), N),
            [N - 1, q_idx],
            [N - 1, qa_idx],
        ]).astype(np.int32)
        n_entries = len(jac_rows)
        _jac_vals = np.empty(n_entries)
        _idx, _ptr, _perm = build_csc_updater(jac_rows, jac_cols,
                                              (n_total, n_total))

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

            boundary_rate = Q * (c[N - 1] / K - _feed(t) * 0.001
                                 / 22.414 * 273.15 * inv_T) * inv_vSample
            derivative = Q / K * inv_vSample
            vals[-4] = derivative               # d(q_net)/dc_R
            vals[-3] = 0.0                      # d(q_net)/dq_net
            vals[-2] = np.sign(boundary_rate) * derivative  # d(q_abs)/dc_R
            vals[-1] = 0.0                      # d(q_abs)/dq_abs

            return csc_matrix((vals[_perm], _idx, _ptr), shape=(n_total, n_total))

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
            boundary_rate = Q * (c_gas - cfeed) * inv_vSample
            dcdt[q_idx] = boundary_rate
            dcdt[qa_idx] = abs(boundary_rate)
            return dcdt

        sparsity = append_quadrature_row(
            generate_jacobian_sparsity(N, num_srcs, False), False, N)
        if not is_inf_cD:
            return rhs, dict(jac_sparsity=sparsity)

        # ---- Analytical Jacobian for the constant-D, finite-F case ----
        # This branch previously always fell back to numerical differencing.
        idx_int = np.arange(1, N - 1)
        jac_rows = np.concatenate([
            [0, 0],                                          # centre
            idx_int, idx_int, idx_int,                        # interior tridiagonal
            [N - 1, N - 1, N - 1],                           # surface cell
            [N, N],                                          # headspace
            np.arange(src_off, src_off + num_srcs),          # source self-decay
            np.tile(np.arange(N), num_srcs),                 # source -> solid rows
            [q_idx, q_idx],                                  # net purge integral
            [qa_idx, qa_idx],                                # absolute purge integral
        ]).astype(np.int32)
        jac_cols = np.concatenate([
            [0, 1],
            idx_int - 1, idx_int, idx_int + 1,
            [N - 2, N - 1, N],
            [N - 1, N],
            np.arange(src_off, src_off + num_srcs),
            np.repeat(np.arange(src_off, src_off + num_srcs), N),
            [N, q_idx],
            [N, qa_idx],
        ]).astype(np.int32)
        _jac_vals = np.empty(len(jac_rows))
        _idx, _ptr, _perm = build_csc_updater(jac_rows, jac_cols,
                                              (n_total, n_total))
        F60 = 60.0 * F
        invV_last = invV[N - 1]

        def jac(t, c):
            TC, _ = _temp(t)
            T = TC + 273.15
            inv_T = 1.0 / T
            Q = _flow(t)
            K = K0 * np.exp(EaK_Rgas * inv_T)
            D = D0 * np.exp(neg_EaD_Rgas * inv_T)

            vals = _jac_vals
            v = orgC * D
            vals[0] = -v
            vals[1] = v

            base = 2
            n_int = N - 2
            vals[base:base + n_int] = D * Csub
            vals[base + n_int:base + 2 * n_int] = D * Cdia
            vals[base + 2 * n_int:base + 3 * n_int] = D * Csup

            b = base + 3 * n_int
            aD = Af_dx[-1] * D * invV_last
            vals[b] = aD                                        # dc_{N-1}/dc_{N-2}
            vals[b + 1] = -aD - Rm * F60 * invV_last            # dc_{N-1}/dc_{N-1}
            vals[b + 2] = Rm * F60 * K * invV_last              # dc_{N-1}/dc_gas
            vals[b + 3] = mp1 * F60 * inv_R_beta                # dc_gas/dc_{N-1}
            vals[b + 4] = -mp1 * F60 * K * inv_R_beta - Q * inv_Vhs

            b_src = b + 5
            if num_srcs > 0:
                k_src = src_A * np.exp(-src_Ea / (Rgas * T))
                vals[b_src:b_src + num_srcs] = -k_src
                b_coup = b_src + num_srcs
                for si in range(num_srcs):
                    vals[b_coup + si * N:b_coup + (si + 1) * N] = k_src[si]

            cfeed = _feed(t) * 0.001 / 22.414 * 273.15 * inv_T
            boundary_rate = Q * (c[N] - cfeed) * inv_vSample
            derivative = Q * inv_vSample
            vals[-4] = derivative               # d(q_net)/dc_gas
            vals[-3] = 0.0                      # d(q_net)/dq_net
            vals[-2] = np.sign(boundary_rate) * derivative  # d(q_abs)/dc_gas
            vals[-1] = 0.0                      # d(q_abs)/dq_abs

            return csc_matrix((vals[_perm], _idx, _ptr), shape=(n_total, n_total))

        return rhs, dict(jac=jac)

# ============================================================================
# Main simulation
# ============================================================================


def run_simulation(tfinal, temp_params, flow_params, cfeed_params,
                   m, R, mSample, rhoSample, Vvessel, MW_analyte,
                   src_params, EaK, K_ref, D_ref, EaD, cD, F, c0free, cgas_init, N,
                   rtol, atol, Tref=50, grid_scheme='uniform', grid_stretch=2.0):
    """Run the conservative MOLIERE outgassing simulation."""
    src_params = _as_float_array(src_params, 'src_params')
    _validate_simulation_inputs(
        tfinal, temp_params, flow_params, cfeed_params,
        m, R, mSample, rhoSample, Vvessel, MW_analyte,
        src_params, EaK, K_ref, D_ref, EaD, cD, F,
        c0free, cgas_init, N, rtol, atol, Tref,
        grid_scheme, grid_stretch,
    )

    num_srcs = len(src_params) // 3
    is_inf_F = np.isposinf(F)
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

    x = make_grid(R, int(N), scheme=grid_scheme, stretch=grid_stretch)
    K_t0 = K0 * np.exp(-EaK / Rgas / (temp_params['T0'] + 273.15))
    _, _, Vfv = fv_geometry(x, m)
    gam = vSample * (m + 1) / R ** (m + 1)

    c_init = np.full(int(N), c0free, dtype=float)
    if is_inf_F:
        c_init[-1] = ((gam * Vfv[-1] * c0free + Vheadspace * cgas_init)
                      / (gam * Vfv[-1] + Vheadspace / K_t0))
        ic = np.hstack((c_init, src_params[::3])) if num_srcs > 0 else c_init
    else:
        ic = (np.hstack((c_init, [cgas_init], src_params[::3]))
              if num_srcs > 0 else np.hstack((c_init, [cgas_init])))
    ic = np.hstack((ic, [0.0, 0.0]))

    dt1 = temp_params['dt1']
    T0 = temp_params['T0']
    Tfinal = temp_params['Tfinal']
    RR = temp_params['RR']
    tEq = flow_params['tEq']
    tFlush = flow_params['tFlush']
    Q_val = flow_params['Q']
    ns = int(cfeed_params['n_steps'])
    dl = cfeed_params['delta']
    st = cfeed_params['step_time']
    bc = cfeed_params['base_conc']
    hti = cfeed_params['hold_time_initial']
    htf = cfeed_params['hold_time_final']
    V_feed = cfeed_params.get('feed_tank_volume', 0.0)

    # The feed tank intentionally remains independent of the chamber flow
    # schedule and is purged continuously at Q_val.
    feed_tank_table = feed_tank_segments(
        tfinal, Q_val, V_feed, ns, dl, st, bc, hti, htf)

    rhs, solver_kw = make_ode_system(
        int(N), x, R, cD, beta, dt1, T0, Tfinal, RR, tEq, tFlush, Q_val,
        ns, dl, st, bc, hti, htf, src_params, Vheadspace, F, Rgas, D0, EaD,
        K0, EaK, m, feed_tank_table=feed_tank_table)

    tank_edges = feed_tank_table[0] if feed_tank_table is not None else None
    breaks = forcing_breakpoints(
        tfinal, temp_params, flow_params, cfeed_params, extra=tank_edges)
    minimum_output_points = 1 + max(len(breaks) - 1, 0) * 7
    minimum_solution_elements = int(ic.size) * int(minimum_output_points)
    if minimum_solution_elements > MAX_SOLUTION_ELEMENTS:
        estimated_mb = minimum_solution_elements * 8.0 / (1024.0 ** 2)
        raise ValueError(
            'The requested grid/forcing combination has a minimum output '
            f'array of about {estimated_mb:.0f} MiB, exceeding the supported '
            'workload. Reduce Grid Points, reduce forcing events, or shorten '
            'the simulation.')
    t_start = time.time()
    try:
        t, Y, solve_stats = solve_forced_ivp(
            rhs, tfinal, ic, breaks, rtol=rtol, atol=atol, **solver_kw)
    except RuntimeError:
        raise
    except (ValueError, FloatingPointError, ArithmeticError) as exc:
        raise RuntimeError(f"SciPy solve_ivp failed: {exc}") from exc
    elapsed = time.time() - t_start
    if not np.all(np.isfinite(Y)):
        raise RuntimeError('Solver returned non-finite values.')

    # Concentration states should be nonnegative.  Remove only solver-scale
    # roundoff; materially negative values indicate an invalid trajectory.
    concentration_states = Y[:-2, :]
    state_scale = max(1.0, float(np.max(np.abs(concentration_states))))
    negative_tolerance = max(10.0 * float(atol), 1e-10 * state_scale, 1e-14)
    min_concentration = float(np.min(concentration_states))
    clipped_negative_count = 0
    if min_concentration < -negative_tolerance:
        raise RuntimeError(
            "Solver produced a materially negative concentration "
            f"({min_concentration:.6g}); tighten tolerances or inspect the parameters."
        )
    if min_concentration < 0:
        mask = concentration_states < 0
        clipped_negative_count = int(np.count_nonzero(mask))
        concentration_states[mask] = 0.0

    y = Y.T
    cR = y[:, int(N) - 1]
    T_C, _ = temperature_vec(t, dt1, T0, Tfinal, RR)
    Q = flow_rate_vec(t, tEq, tFlush, Q_val)
    T_K = T_C + 273.15
    K = K0 * np.exp(-EaK / Rgas / T_K)
    S = K * (273.15 / T_K)
    DR = D0 * np.exp(-EaD / Rgas / T_K
                     + cR * (0.0 if np.isposinf(cD) else 1.0 / cD))
    DR_cm2s = DR / 60.0

    c_feed_ppbv = (feed_tank_eval(t, *feed_tank_table)
                   if feed_tank_table is not None
                   else feed_conc_vec(t, ns, dl, st, bc, hti, htf))
    c_feed_uM = c_feed_ppbv / 1000 / 22.414 * 273.15 / T_K
    c_gas = cR / K if is_inf_F else y[:, int(N)]
    cGasPPBV = c_gas * 1000 * 22.414 / 273.15 * T_K

    initial_integral = y[0, :int(N)] @ Vfv
    time_integrals = y[:, :int(N)] @ Vfv
    delta_mass_free = (-(m + 1) / R ** (m + 1) * vSample * MW_analyte
                       * (initial_integral - time_integrals))

    src_off = int(N) if is_inf_F else int(N) + 1
    if num_srcs > 0:
        src_remaining = y[:, src_off:src_off + num_srcs]
        src_initial = src_params[::3]
        delta_mass_src = (-vSample * MW_analyte
                          * np.sum(src_initial - src_remaining, axis=1))
    else:
        delta_mass_src = np.zeros_like(t)
    delta_mass_total = delta_mass_free + delta_mass_src

    c0_end = (m + 1) / R ** (m + 1) * (y[-1, :int(N)] @ Vfv)
    c_Gas_end = c_gas[-1] * beta
    initial_total = c0free + beta * cgas_init + float(np.sum(src_params[::3]))
    final_source_total = (float(np.sum(y[-1, src_off:src_off + num_srcs]))
                          if num_srcs > 0 else 0.0)
    final_total = c0_end + c_Gas_end + final_source_total
    purge_integral = float(y[-1, -2])
    absolute_purge_integral = float(y[-1, -1])
    flux_scale = max(abs(purge_integral), 1.0)
    if absolute_purge_integral < -max(10.0 * float(atol), 1e-12 * flux_scale):
        raise RuntimeError(
            'Solver produced a materially negative absolute boundary-flux integral.')
    absolute_purge_integral = max(0.0, absolute_purge_integral)
    balance_residual = float(initial_total - final_total - purge_integral)
    balance_scale = max(abs(initial_total), abs(final_total),
                        abs(purge_integral), abs(absolute_purge_integral), 1e-12)
    mass_bal_error = 100.0 * balance_residual / balance_scale

    return {
        't': t,
        'T_C': T_C,
        'c_gas_ppbv': cGasPPBV,
        'c_gas_uM': c_gas,
        'MW': MW_analyte,
        'c_feed_ppbv': c_feed_ppbv,
        'c_feed_uM': c_feed_uM,
        'Q': Q,
        'D': DR_cm2s,
        'S': S,
        'delta_mass': delta_mass_total,
        'mass_bal_error': mass_bal_error,
        'mass_balance_residual': balance_residual,
        'mass_balance_scale': balance_scale,
        'initial_total_inventory': float(initial_total),
        'final_total_inventory': float(final_total),
        'purge_integral': purge_integral,
        'net_boundary_flux_integral': purge_integral,
        'absolute_purge_integral': absolute_purge_integral,
        'absolute_boundary_flux_integral': absolute_purge_integral,
        'minimum_concentration_before_clip': min_concentration,
        'clipped_negative_count': clipped_negative_count,
        'solve_time': elapsed,
        'n_segments': solve_stats['n_segments'],
        'nfev': solve_stats['nfev'],
        'njev': solve_stats['njev'],
        'nlu': solve_stats['nlu'],
    }


class _Tooltip:
    """Minimal hover tooltip (ttk has none built in)."""

    def __init__(self, widget, text, wraplength=380, delay=450):
        self.widget, self.text = widget, text
        self.wraplength, self.delay = wraplength, delay
        self._after_id = self._win = None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        if self._win is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self._win = tk.Toplevel(self.widget)
            self._win.wm_overrideredirect(True)
            self._win.wm_geometry(f"+{x}+{y}")
            tk.Label(self._win, text=self.text, justify=tk.LEFT,
                     background='#ffffe0', relief=tk.SOLID, borderwidth=1,
                     wraplength=self.wraplength, font=('TkDefaultFont', 8)
                     ).pack(ipadx=4, ipady=2)
        except tk.TclError:
            self._win = None

    def _hide(self, _event=None):
        self._cancel()
        if self._win is not None:
            try:
                self._win.destroy()
            except tk.TclError:
                pass
            self._win = None


# ============================================================================
# GUI Application
# ============================================================================

class OutgassingGUI:

    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_TITLE} v{APP_VERSION}")
        try:
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            # The plot pane carries the canvas plus the docked data-entry panel,
            # so the old 1850x1000 cap left the canvas narrow on a large display.
            win_w = max(1280, min(2200, int(sw * 0.95)))
            win_h = max(720, min(1250, int(sh * 0.92)))
            self.root.geometry(f"{win_w}x{win_h}")
        except tk.TclError:
            self.root.geometry("1700x950")

        try:
            self.root.minsize(1120, 700)
        except tk.TclError:
            pass

        self.action_frame = ttk.Frame(root, padding=(10, 6))
        self.action_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_pane.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.control_frame = ttk.Frame(self.main_pane, padding="10")
        self.plot_frame = ttk.Frame(self.main_pane, padding="10")
        self.main_pane.add(self.control_frame, weight=0)
        self.main_pane.add(self.plot_frame, weight=1)
        self.main_pane.bind('<Configure>', self._clamp_sash, add='+')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        # Persistent files live in the platform's per-user application-data
        # directory rather than whichever working directory launched the GUI.
        if os.name == 'nt':
            config_root = os.environ.get('APPDATA', os.path.expanduser('~'))
        else:
            config_root = os.environ.get(
                'XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config'))
        self.app_dir = os.path.join(config_root, 'OutgassingModel')
        os.makedirs(self.app_dir, exist_ok=True)
        self.config_file = os.path.join(self.app_dir, 'config.json')
        self.legacy_config_file = os.path.join(
            os.path.expanduser('~'), '.outgassing_gui_config.json')
        self.log_file = os.path.join(self.app_dir, 'outgassing_gui.log')

        self.source_entries = []
        self.current_results = None
        self.all_sweep_results = []
        self.current_run_record = None
        self.sweep_run_records = []
        self._last_experimental_snapshot = None
        self._last_result_mode = None
        self._busy = False
        self._busy_state_cache = []
        self._active_job_id = 0
        self.defaults_file = self._load_config_defaults_path()

        self.sweep_enabled = tk.BooleanVar(value=False)
        self.opt_enabled = tk.BooleanVar(value=False)
        self.opt_param_rows = []
        self.opt_n_params = 1
        self._opt_stop = False

        self.SWEEP_PARAMS = {
            "T0 – Initial Temp (°C)":                       ("T0",        False),
            "Tfinal – Final Temp (°C)":                     ("Tfinal",    False),
            "RR – Ramp Rate (°C/min)":                      ("RR",        True),
            "Q – Flow Rate (ml/min)":                       ("Q",         True),
            "V_feed – Feed Tank Volume (ml)":               ("feed_tank_volume", False),
            "R – Sample Radius / Half-thickness (µm)":      ("R",         True),
            "mSample – Sample Mass (mg)":                   ("mSample",   True),
            "rhoSample – Density (g/ml)":                   ("rhoSample", True),
            "Vvessel – Vessel Volume (ml)":                 ("Vvessel",   True),
            "N – Number of Grid Points":                    ("N",            True),
            "Grid Stretch – β / ratio (non-uniform grids)": ("grid_stretch", False),
            "K – Partition Coeff @ Tref":                    ("K_ref",       True),
            "EaK – Sorption Enthalpy (kJ/mol)":             ("EaK",       False),
            "D – Diffusivity @ Tref,c=0 (cm²/s)":            ("D_ref",       True),
            "EaD – Diffusivity Ea (kJ/mol)":                ("EaD",       False),
            "cD – Plasticization Power (µM)":               ("cD",        True),
            "F – Surface Mass-Transfer Coeff (cm/s)":       ("F",         True),
            "c0free – Initial Mobile Conc (µM)":            ("c0free",    True),
            "base_conc – Feed Base Conc (ppbv)":            ("base_conc", False),
            "Src 1 – c₀ Initial Conc (µM)":                 ("src1_c0",   True),
            "Src 1 – A Pre-exponential (1/min)":            ("src1_A",    True),
            "Src 1 – Ea Activation Energy (kJ/mol)":        ("src1_Ea",   False),
            "Src 2 – c₀ Initial Conc (µM)":                 ("src2_c0",   True),
            "Src 2 – A Pre-exponential (1/min)":            ("src2_A",    True),
            "Src 2 – Ea Activation Energy (kJ/mol)":        ("src2_Ea",   False),
            "Src 3 – c₀ Initial Conc (µM)":                 ("src3_c0",   True),
            "Src 3 – A Pre-exponential (1/min)":            ("src3_A",    True),
            "Src 3 – Ea Activation Energy (kJ/mol)":        ("src3_Ea",   False),
        }
        self.SWEEP_INT_ATTRS = {"N"}
        self.OPT_EXCLUDED_ATTRS = {"N", "grid_stretch"}

        self.create_action_bar()
        self.create_controls()
        self.create_plots()
        self.load_defaults()
        self.root.after_idle(self._init_sash_position)

    def _init_sash_position(self):
        """Set the initial controls/plots divider to the controls' natural width."""
        try:
            self.root.update_idletasks()
            want = self.control_frame.winfo_reqwidth()
            total = self.main_pane.winfo_width()
            # The plot pane now holds the canvas AND the docked data-entry panel,
            # so reserve room for both before giving the controls their natural
            # width -- otherwise the canvas is the only thing that gets squeezed.
            reserve = EXP_PANEL_MIN_WIDTH + PLOT_CANVAS_MIN_WIDTH
            if total > 0:
                want = min(want, max(360, total - reserve))
            self.main_pane.sashpos(0, max(1, int(want)))
        except Exception:
            pass


    def _load_config_defaults_path(self):
        """Load the remembered parameter-file path, including legacy migration."""
        fallback = os.path.join(self.app_dir, 'outgassing_defaults.json')
        for cfg_path in (self.config_file, self.legacy_config_file):
            try:
                if not os.path.exists(cfg_path):
                    continue
                with open(cfg_path, 'r', encoding='utf-8') as handle:
                    cfg = json.load(handle)
                path = os.path.abspath(os.path.expanduser(
                    str(cfg.get('defaults_file', fallback))))
                if os.path.exists(path):
                    return path
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
        return fallback


    def _save_config(self):
        """Persist the active parameter-file path atomically."""
        try:
            self._atomic_write_json(
                self.config_file,
                {'schema_version': CONFIG_SCHEMA_VERSION,
                 'defaults_file': os.path.abspath(self.defaults_file)})
        except OSError as exc:
            self._log_exception('saving application config', exc)

    def _update_defaults_label(self):
        """Update the status label showing the active parameter file."""
        display = os.path.basename(self.defaults_file)
        self.defaults_path_label.config(text=f"Param file: {display}")


    def save_defaults_as(self):
        """Save current GUI values to a user-chosen JSON file."""
        if self._busy:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Parameters As",
            initialdir=os.path.dirname(self.defaults_file) or self.app_dir,
            initialfile=os.path.basename(self.defaults_file),
        )
        if not filename:
            return
        filename = os.path.abspath(filename)
        if self.save_defaults(path=filename):
            self.defaults_file = filename
            self._save_config()
            self._update_defaults_label()


    def load_defaults_from_file(self):
        """Browse for a validated parameter file and apply it atomically."""
        if self._busy:
            return
        filename = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Parameters",
            initialdir=os.path.dirname(self.defaults_file) or self.app_dir,
        )
        if not filename:
            return
        filename = os.path.abspath(filename)
        if self.load_defaults(path=filename):
            self.defaults_file = filename
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
        # Simulations now run on a worker thread, so the UI stays responsive and
        # a long sweep can be interrupted between points.
        self.stop_sim_btn = ttk.Button(row1, text="Stop", state='disabled',
                                       command=self._stop_simulation)
        self.stop_sim_btn.pack(side=tk.LEFT, padx=(0, 4))
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

        # Mouse-wheel scrolling is bound to this toplevel only and activated
        # only when the pointer is over the controls.  This avoids bind_all /
        # unbind_all removing wheel handlers installed by unrelated widgets.
        def _over_controls(event):
            try:
                w = self.root.winfo_containing(event.x_root, event.y_root)
                while w is not None:
                    if w is self.control_frame:
                        return True
                    w = getattr(w, 'master', None)
            except tk.TclError:
                return False
            return False

        def _ctrl_wheel(event):
            if _over_controls(event):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return 'break'
            return None

        def _ctrl_shift_wheel(event):
            if _over_controls(event):
                canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
                return 'break'
            return None

        self.root.bind("<MouseWheel>", _ctrl_wheel, add='+')
        self.root.bind("<Shift-MouseWheel>", _ctrl_shift_wheel, add='+')


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
        self.grid_stretch = self.create_entry(sim, "Grid Stretch (β; ratio >1 surface, <1 origin):", "2.0", r); r += 1

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
        self.feed_cycle_var = tk.StringVar(value="—")
        self.create_readout(feed, "Repeating feed-cycle period (min):",
                            r, self.feed_cycle_var); r += 1

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
        # Initial headspace concentration has its own input unit.  Plot-unit
        # changes are display-only and can no longer reinterpret this value.
        self.cgas_init_label = ttk.Label(trans, text="Initial gas concentration:")
        self.cgas_init_label.grid(row=r, column=0, sticky=tk.W, pady=2)
        cgas_frame = ttk.Frame(trans)
        cgas_frame.grid(row=r, column=1, sticky=tk.E, pady=2)
        self.cgas_init = ttk.Entry(cgas_frame, width=8)
        self.cgas_init.insert(0, "0")
        self.cgas_init.pack(side=tk.LEFT)
        self.cgas_init_unit = ttk.Combobox(
            cgas_frame, width=7, state='readonly', values=CONC_UNITS)
        self.cgas_init_unit.set('ppbv')
        self._cgas_input_unit_last = 'ppbv'
        self.cgas_init_unit.pack(side=tk.LEFT, padx=(3, 0))
        self.cgas_init_unit.bind(
            '<<ComboboxSelected>>', self._on_cgas_input_unit_change)
        r += 1
        # Characteristic diffusion time ~ R^2 / D, at Tref and c = 0.
        self.tau_diff_var = tk.StringVar(value="—")
        self.create_readout(trans, "Char. diffusion time, R²/D (min):",
                            r, self.tau_diff_var); r += 1
        # Mass-transfer Biot number Bi = F·R/D, at Tref and c = 0.  Ratio of the
        # internal diffusion resistance to the interfacial transfer resistance:
        # Bi >> 1 is diffusion-controlled, Bi << 1 is surface-controlled, and
        # F = inf is the equilibrium-surface limit (Bi → ∞).
        self.biot_var = tk.StringVar(value="—")
        self.create_readout(trans, "Biot number, Bi = F·R/D:",
                            r, self.biot_var); r += 1
        # Purge Péclet number Pe = Q·R²/(V_head·D) = τ_D/τ_v, at Tref and c = 0.
        # Convective removal from the headspace vs. diffusive supply from the
        # sample: Pe >> 1 keeps the headspace swept toward the feed value, while
        # Pe << 1 lets it sit near partition equilibrium with the sample.
        self.peclet_var = tk.StringVar(value="—")
        self.create_readout(trans, "Péclet number, Pe = QR²/(V_h·D):",
                            r, self.peclet_var); r += 1
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
        self.sweep_start = ttk.Entry(sweep, width=12)
        self.sweep_start.insert(0, "1e-8")
        self.sweep_start.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.sweep_start.config(state='disabled')

        ttk.Label(sweep, text="Stop Value:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.sweep_stop = ttk.Entry(sweep, width=12)
        self.sweep_stop.insert(0, "1e-6")
        self.sweep_stop.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.sweep_stop.config(state='disabled')

        ttk.Label(sweep, text="Number of Points:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.sweep_n = ttk.Entry(sweep, width=12)
        self.sweep_n.insert(0, "5")
        self.sweep_n.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.sweep_n.config(state='disabled')

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
        self.opt_n_entry = ttk.Entry(opt, width=8)
        self.opt_n_entry.insert(0, "1")
        self.opt_n_entry.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.opt_n_entry.config(state='disabled')

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
            opt, values=["Concentration", "Mass", "Both"],
            width=20, state='disabled')
        self.opt_target.set("Concentration")
        self.opt_target.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1

        ttk.Label(opt, text="Concentration block weight:").grid(
            row=r, column=0, sticky=tk.W, pady=2)
        self.opt_conc_weight = ttk.Entry(opt, width=8)
        self.opt_conc_weight.insert(0, "1.0")
        self.opt_conc_weight.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.opt_conc_weight.config(state='disabled')

        ttk.Label(opt, text="Mass block weight:").grid(
            row=r, column=0, sticky=tk.W, pady=2)
        self.opt_mass_weight = ttk.Entry(opt, width=8)
        self.opt_mass_weight.insert(0, "1.0")
        self.opt_mass_weight.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.opt_mass_weight.config(state='disabled')

        ttk.Label(opt, text="Multi-Starts:").grid(row=r, column=0, sticky=tk.W, pady=2)
        # Number of Nelder-Mead fits to run: start 1 uses the current field
        # values; additional starts are drawn from a Latin-hypercube sample of
        # the bounds. The per-start results table reveals fit (non-)uniqueness.
        self.opt_nstarts = ttk.Entry(opt, width=8)
        self.opt_nstarts.insert(0, "1")
        self.opt_nstarts.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.opt_nstarts.config(state='disabled')

        ttk.Label(opt, text="Max Evals per Start:").grid(row=r, column=0, sticky=tk.W, pady=2)
        self.opt_maxiter = ttk.Entry(opt, width=8)
        self.opt_maxiter.insert(0, "200")
        self.opt_maxiter.grid(row=r, column=1, sticky=tk.E, pady=2); r += 1
        self.opt_maxiter.config(state='disabled')

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
        """Enable or disable sweep widgets without overriding a busy lock."""
        if getattr(self, '_busy', False):
            return
        enabled = self.sweep_enabled.get()
        state = 'normal' if enabled else 'disabled'
        for widget in (self.sweep_start, self.sweep_stop, self.sweep_n,
                       self.sweep_cmap):
            widget.config(state=state)
        self.sweep_param_combo.config(state='readonly' if enabled else 'disabled')
        self.sweep_log_check.config(state=state)

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
        for w in (self.feed_tank_volume, self.Q, self.n_steps, self.delta,
                  self.step_time, self.hold_time_initial, self.hold_time_final,
                  self.Vvessel, self.mSample, self.rhoSample, self.R, self.D_ref,
                  self.MW_analyte, self.c0free, self.K_ref, self.Tref, self.F):
            w.bind("<KeyRelease>", self._update_derived, add="+")
            w.bind("<FocusOut>", self._update_derived, add="+")

    def _update_derived(self, *_event):
        """Update the computed read-outs (residence times, diffusion time).

        Each value is computed independently; a bad/blank field only blanks its
        own read-out (shown as '—') rather than raising.
        """
        def _f(widget):
            return float(widget.get())

        # Feed-tank residence time, tau_f = V_feed / Q  (min)
        try:
            V_feed = _f(self.feed_tank_volume)
            Q = _f(self.Q)
            if V_feed <= 0:
                self.tau_feed_var.set("0 (ideal)")
            elif Q <= 0:
                self.tau_feed_var.set("∞")
            else:
                self.tau_feed_var.set(f"{V_feed / Q:.4g}")
        except (ValueError, tk.TclError):
            self.tau_feed_var.set("—")

        # The ideal staircase is periodic through feed_conc_vec's modulo cycle.
        # Show that period explicitly so repetition is never a hidden assumption.
        try:
            n_steps = int(self.n_steps.get())
            delta = _f(self.delta)
            step_time = _f(self.step_time)
            hold_initial = _f(self.hold_time_initial)
            hold_final = _f(self.hold_time_final)
            period = hold_initial + 2 * n_steps * step_time + hold_final
            if n_steps <= 0 or delta == 0:
                self.feed_cycle_var.set("constant")
            elif period <= 0:
                self.feed_cycle_var.set("—")
            else:
                self.feed_cycle_var.set(f"{period:.4g}")
        except (ValueError, tk.TclError):
            self.feed_cycle_var.set("—")

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

        # Mass-transfer Biot number, Bi = F·R/D  (dimensionless, at Tref, c = 0)
        # with F [cm/s], R [cm] and D [cm²/s].  F = 'inf' is the instantaneous-
        # equilibrium surface used by the F = inf branch of the model.
        try:
            R_cm = _f(self.R) * 1e-4              # µm -> cm
            D = _f(self.D_ref)                   # cm^2/s
            F_val = _f(self.F)                   # cm/s ('inf' parses to inf)
            if R_cm <= 0 or D <= 0 or F_val < 0:
                self.biot_var.set("—")
            elif np.isposinf(F_val):
                self.biot_var.set("∞ (equil.)")
            else:
                self.biot_var.set(f"{F_val * R_cm / D:.4g}")
        except (ValueError, ZeroDivisionError, tk.TclError):
            self.biot_var.set("—")

        # Purge Péclet number, Pe = Q·R²/(V_head·D) = τ_D / τ_v (dimensionless,
        # at Tref, c = 0).  Headspace flush rate Q/V_head [1/min] over the
        # diffusive release rate D/R² [1/min].
        try:
            R_cm = _f(self.R) * 1e-4              # µm -> cm
            D = _f(self.D_ref)                   # cm^2/s
            Q = _f(self.Q)                       # ml/min
            vSample = _f(self.mSample) * 1e-3 / _f(self.rhoSample)   # mg, g/ml -> ml
            Vhead = _f(self.Vvessel) - vSample
            if R_cm <= 0 or D <= 0:
                self.peclet_var.set("—")
            elif Vhead <= 0:
                self.peclet_var.set("— (no headspace)")
            elif Q <= 0:
                self.peclet_var.set("0 (Q = 0)")
            else:
                tau_D_min = (R_cm * R_cm / D) / 60.0
                self.peclet_var.set(f"{tau_D_min * Q / Vhead:.4g}")
        except (ValueError, ZeroDivisionError, tk.TclError):
            self.peclet_var.set("—")

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


    def _safe_MW(self, default=float('nan')):
        """Return a strictly positive analyte molecular weight or ``default``."""
        try:
            mw = float(self.MW_analyte.get())
        except (ValueError, tk.TclError, AttributeError):
            return default
        return mw if np.isfinite(mw) and mw > 0 else default


    def _cgas_input_unit(self):
        try:
            unit = self.cgas_init_unit.get()
        except (AttributeError, tk.TclError):
            unit = 'ppbv'
        return unit if unit in CONC_UNITS else 'ppbv'

    def _on_cgas_input_unit_change(self, _event=None):
        """Convert the visible initial value when its input unit changes."""
        new_unit = self._cgas_input_unit()
        old_unit = getattr(self, '_cgas_input_unit_last', new_unit)
        if new_unit == old_unit:
            return
        try:
            val = float(self.cgas_init.get())
            T0_K = float(self.T0.get()) + 273.15
            mw = self._safe_MW()
            if not np.isfinite(T0_K) or T0_K <= 0:
                raise ValueError('Initial temperature must be greater than -273.15 °C.')
            if not np.isfinite(mw) or mw <= 0:
                raise ValueError('Molecular weight must be a finite value > 0.')
            c_uM = float(self._conc_to_uM(val, T0_K, old_unit, mw))
            converted = float(self._conc_from_uM(c_uM, T0_K, new_unit, mw))
            if not np.isfinite(converted):
                raise ValueError('Conversion requires a valid positive molecular weight.')
            self.cgas_init.delete(0, tk.END)
            self.cgas_init.insert(0, f'{converted:.10g}')
        except (ValueError, tk.TclError, FloatingPointError) as exc:
            self.cgas_init_unit.set(old_unit)
            messagebox.showerror('Unit Conversion Error', str(exc))
            return
        self._cgas_input_unit_last = new_unit
        self._update_derived()

    def _cgas_init_uM(self):
        """Convert the independently unit-tagged initial gas value to µM."""
        val = float(self.cgas_init.get())
        unit = self._cgas_input_unit()
        T0_K = float(self.T0.get()) + 273.15
        mw = self._safe_MW()
        if not np.isfinite(T0_K) or T0_K <= 0:
            raise ValueError('Initial temperature must be greater than -273.15 °C.')
        if not np.isfinite(mw) or mw <= 0:
            raise ValueError('Molecular weight must be a finite value > 0.')
        out = float(self._conc_to_uM(val, T0_K, unit, mw))
        if not np.isfinite(out):
            raise ValueError(
                'Initial gas concentration could not be converted; check the '
                'value, input unit, temperature, and molecular weight.')
        return out

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

    def _exp_conc_convert(self, times, raw_vals, from_unit, to_unit,
                          reference_results=None):
        """Convert experimental concentration using the associated run state.

        When ``reference_results`` is supplied, temperature and molecular
        weight come from that frozen result rather than mutable GUI widgets.
        This keeps overlays and exports reproducible after later GUI edits.
        """
        if from_unit == to_unit:
            return np.asarray(raw_vals, dtype=float)
        times = np.asarray(times, dtype=float)
        if reference_results is not None:
            model_time = np.asarray(reference_results['t'], dtype=float)
            model_temp = np.asarray(reference_results['T_C'], dtype=float)
            T_K = np.interp(times, model_time, model_temp) + 273.15
            MW = float(reference_results.get('MW', self._safe_MW()))
        else:
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


    def _log_exception(self, context, exc):
        """Append a full traceback to the per-user diagnostic log."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as handle:
                handle.write(f"\n[{_utc_now_iso()}] {context}: {exc}\n")
                handle.write(traceback.format_exc())
        except OSError:
            pass

    @staticmethod
    def _atomic_write_json(path, payload):
        """Write JSON through a same-directory temporary file and os.replace."""
        path = os.path.abspath(path)
        directory = os.path.dirname(path) or '.'
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix='.outgassing-', suffix='.tmp', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _set_entry_value(widget, value):
        """Write `value` into an Entry, temporarily lifting a disabled state.

        widget.cget('state') returns a Tcl index object, not a str: comparing it
        to 'disabled' with == is always False.  That made this helper a silent
        no-op on every disabled widget, so loading a parameter file quietly
        discarded the sweep block and every optimizer row while reporting
        success.  str() before comparing, and restore whatever state was there
        rather than assuming it was 'normal'.
        """
        old_state = None
        try:
            old_state = str(widget.cget('state'))
        except tk.TclError:
            old_state = None
        try:
            if old_state in ('disabled', 'readonly'):
                widget.config(state='normal')
            widget.delete(0, tk.END)
            widget.insert(0, str(value))
        finally:
            if old_state in ('disabled', 'readonly'):
                try:
                    widget.config(state=old_state)
                except tk.TclError:
                    pass

    def _defaults_dict_from_gui(self):
        """Serialize the complete editable GUI state."""
        defaults = {
            'schema_version': CONFIG_SCHEMA_VERSION,
            'tfinal': self.tfinal.get(), 'N': self.N.get(),
            'rtol': self.rtol.get(), 'atol': self.atol.get(),
            'grid_scheme': self.grid_scheme.get(),
            'grid_stretch': self.grid_stretch.get(),
            'dt1': self.dt1.get(), 'T0': self.T0.get(),
            'Tfinal': self.Tfinal.get(), 'RR': self.RR.get(),
            'tEq': self.tEq.get(), 'Q': self.Q.get(),
            'n_steps': self.n_steps.get(), 'delta': self.delta.get(),
            'step_time': self.step_time.get(), 'base_conc': self.base_conc.get(),
            'hold_time_initial': self.hold_time_initial.get(),
            'hold_time_final': self.hold_time_final.get(),
            'feed_tank_volume': self.feed_tank_volume.get(),
            'm': self.m.get(), 'R': self.R.get(),
            'mSample': self.mSample.get(), 'rhoSample': self.rhoSample.get(),
            'Vvessel': self.Vvessel.get(), 'MW_analyte': self.MW_analyte.get(),
            'K_ref': self.K_ref.get(), 'EaK': self.EaK.get(),
            'D_ref': self.D_ref.get(), 'EaD': self.EaD.get(),
            'Tref': self.Tref.get(), 'cD': self.cD.get(), 'F': self.F.get(),
            'c0free': self.c0free.get(), 'cgas_init': self.cgas_init.get(),
            'cgas_init_unit': self._cgas_input_unit(),
            'num_sources': self.num_sources.get(),
            'sources': [
                {'c0': sw[1].get(), 'A': sw[3].get(), 'Ea': sw[5].get()}
                for sw in self.source_entries
            ],
            'sweep_enabled': self.sweep_enabled.get(),
            'sweep_param': self.sweep_param_var.get(),
            'sweep_start': self.sweep_start.get(),
            'sweep_stop': self.sweep_stop.get(),
            'sweep_n': self.sweep_n.get(),
            'sweep_log': self.sweep_log.get(),
            'sweep_cmap': self.sweep_cmap.get(),
            'logtime': self.logtime_var.get(),
            'logconc': self.logconc_var.get(),
            'time_min': self.time_min.get(),
            'time_max': self.time_max.get(),
            'opt_enabled': self.opt_enabled.get(),
            'opt_n': self.opt_n_entry.get(),
            'opt_target': self.opt_target.get(),
            'opt_conc_weight': self.opt_conc_weight.get(),
            'opt_mass_weight': self.opt_mass_weight.get(),
            'opt_nstarts': self.opt_nstarts.get(),
            'opt_maxiter': self.opt_maxiter.get(),
            'opt_rows': [
                {'param': rd['combo'].get(), 'min': rd['min_entry'].get(),
                 'max': rd['max_entry'].get(), 'log': rd['log_var'].get()}
                for rd in self.opt_param_rows
            ],
            'exp_conc_data': self.conc_data_text.get('1.0', tk.END).strip(),
            'exp_mass_data': self.mass_data_text.get('1.0', tk.END).strip(),
            'conc_interp': self.conc_interp_var.get(),
            'conc_interp_n': self.conc_interp_n.get(),
            'conc_interp_method': self.conc_interp_method.get(),
            'mass_interp': self.mass_interp_var.get(),
            'mass_interp_n': self.mass_interp_n.get(),
            'mass_interp_method': self.mass_interp_method.get(),
            'conc_plot_unit': self.conc_unit.get(),
            'conc_data_unit': self.conc_data_unit.get(),
            'mass_plot_unit': self.mass_unit.get(),
            'mass_data_unit': self.mass_data_unit.get(),
            'duplicate_policy': self.duplicate_policy.get(),
        }
        return defaults

    def _validate_defaults_payload(self, defaults):
        """Validate a complete parameter file before changing any widget."""
        self._last_payload_repairs = []
        if not isinstance(defaults, dict):
            raise ValueError('Parameter file must contain a JSON object.')
        d = copy.deepcopy(defaults)
        try:
            schema_version = int(d.get('schema_version', 1))
        except (TypeError, ValueError) as exc:
            raise ValueError('schema_version must be an integer.') from exc
        if schema_version < 1 or schema_version > CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f'Unsupported parameter-file schema {schema_version}; this '
                f'application supports versions 1 through {CONFIG_SCHEMA_VERSION}.')
        # Older files are normalized by the fallbacks/migrations below and are
        # written back using the current schema on the next save.
        d['schema_version'] = CONFIG_SCHEMA_VERSION

        # A blank string is not a value.  The GUI serializes raw Entry contents,
        # so any field the user never filled in -- and every field belonging to a
        # disabled feature -- arrives here as ''.  Treat those as absent so the
        # documented fallbacks apply, instead of letting int('') abort the load.
        # BLANK_IS_MEANINGFUL lists the keys where '' is a real setting: the two
        # axis limits mean "auto", and the data boxes mean "no data".
        BLANK_IS_MEANINGFUL = {'time_min', 'time_max',
                               'exp_conc_data', 'exp_mass_data', 'sweep_param'}

        def _is_blank(value):
            return isinstance(value, str) and not value.strip()

        # Physically significant fields get their documented default when blank,
        # but that is never done silently -- a blank geometry becoming a sphere
        # would otherwise change the model without telling anyone.  Purely
        # presentational or feature-toggle fields are filled quietly.
        QUIET_IF_BLANK = {
            'sweep_start', 'sweep_stop', 'sweep_n', 'sweep_log', 'sweep_cmap',
            'opt_n', 'opt_nstarts', 'opt_maxiter', 'opt_target', 'opt_rows',
            'opt_conc_weight', 'opt_mass_weight', 'logtime', 'logconc',
            'conc_interp', 'conc_interp_n', 'conc_interp_method',
            'mass_interp', 'mass_interp_n', 'mass_interp_method',
            'conc_plot_unit', 'conc_data_unit', 'mass_plot_unit',
            'mass_data_unit', 'cgas_init_unit', 'duplicate_policy',
            'schema_version', 'sweep_enabled', 'opt_enabled',
        }
        blank_defaulted = []
        for key in [k for k, v in d.items()
                    if k not in BLANK_IS_MEANINGFUL and _is_blank(v)]:
            if key not in QUIET_IF_BLANK:
                blank_defaulted.append(key)
            del d[key]
        for list_key in ('opt_rows', 'sources'):
            entries = d.get(list_key)
            if isinstance(entries, list):
                d[list_key] = [
                    {k: v for k, v in entry.items() if not _is_blank(v)}
                    if isinstance(entry, dict) else entry
                    for entry in entries
                ]

        # Settings that belong to a switched-off feature must never block a load.
        # They are repaired to something usable and reported to the caller.
        repairs = []
        if blank_defaulted:
            repairs.append(
                'blank in the file, so the built-in default was used for: '
                + ', '.join(sorted(blank_defaulted)))

        def val(key, fallback):
            return d.get(key, fallback)

        def f(key, fallback):
            try:
                out = float(val(key, fallback))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'{key} must be a number (got {val(key, fallback)!r}).') from exc
            if not np.isfinite(out):
                raise ValueError(f'{key} must be finite.')
            return out

        def as_int(key, fallback):
            try:
                return int(float(val(key, fallback)))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'{key} must be a whole number '
                    f'(got {val(key, fallback)!r}).') from exc

        def inf_float(key, fallback):
            raw = str(val(key, fallback)).strip().lower()
            if raw in ('inf', '+inf', 'infinity', '+infinity'):
                return np.inf
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{key} must be a number or 'inf' (got {raw!r}).") from exc

        def clamp(key, fallback, lo, hi, enabled, as_int=True):
            """Range-check a setting; repair rather than reject when disabled."""
            raw = val(key, fallback)
            try:
                out = int(float(raw)) if as_int else float(raw)
            except (TypeError, ValueError):
                if enabled:
                    raise ValueError(f'{key} must be a number (got {raw!r}).')
                out = int(float(fallback)) if as_int else float(fallback)
                repairs.append(f'{key} was not a number; using {out}')
            if not lo <= out <= hi:
                if enabled:
                    raise ValueError(f'{key} must be between {lo} and {hi}.')
                fixed = min(max(out, lo), hi)
                repairs.append(f'{key}={out} is outside {lo}-{hi}; using {fixed}')
                out = fixed
            d[key] = str(out)
            return out

        # Schema 1 stored the initial gas concentration in µM (the field was
        # labelled "Initial gas concentration (µM)").  Schema 2 added a unit
        # selector whose default is ppbv, so an unmigrated schema-1 file would
        # have its initial gas silently reinterpreted.  Convert it instead.
        if schema_version < 2 and 'cgas_init_unit' not in d:
            try:
                legacy_uM = float(val('cgas_init', '0'))
            except (TypeError, ValueError):
                legacy_uM = 0.0
            d['cgas_init_unit'] = 'ppbv'
            if legacy_uM != 0.0 and np.isfinite(legacy_uM):
                try:
                    converted = float(self._conc_from_uM(
                        legacy_uM, f('T0', '50') + 273.15, 'ppbv',
                        f('MW_analyte', '18')))
                    d['cgas_init'] = f'{converted:.10g}'
                    repairs.append(
                        f'cgas_init {legacy_uM:g} µM converted to '
                        f'{converted:.6g} ppbv (schema 1 -> 2)')
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    repairs.append(
                        'cgas_init could not be converted from µM; left as-is')

        n_sources = as_int('num_sources', '1')
        if not 0 <= n_sources <= MAX_UI_SOURCES:
            raise ValueError(f'num_sources must be between 0 and {MAX_UI_SOURCES}.')
        sources = list(val('sources', []))
        src_values = []
        for src_index in range(n_sources):
            src = (sources[src_index]
                   if src_index < len(sources) and isinstance(sources[src_index], dict)
                   else {})
            src_values.extend((float(src.get('c0', '100')),
                               float(src.get('A', '1e8')),
                               float(src.get('Ea', '80'))))
        src_array = _as_float_array(src_values, 'sources')

        # "2 - Sphere" -> 2.  An empty or malformed string used to raise
        # IndexError, which load_defaults does not catch, crashing the app
        # instead of reporting an unusable file.
        m_text = str(val('m', '2 - Sphere')).strip()
        m_head = m_text.split()[0] if m_text.split() else ''
        try:
            m_value = int(m_head)
        except ValueError as exc:
            raise ValueError(
                f'Unknown geometry {m_text!r}; expected a value like '
                "'0 - Slab', '1 - Cylinder' or '2 - Sphere'.") from exc
        if m_value not in (0, 1, 2):
            raise ValueError(f'Geometry index must be 0, 1 or 2 (got {m_value}).')
        mw = f('MW_analyte', '18')
        if mw <= 0:
            raise ValueError('MW_analyte must be > 0.')
        cgas_unit = str(val('cgas_init_unit', val('conc_plot_unit', 'ppbv')))
        if cgas_unit not in CONC_UNITS:
            raise ValueError(f'Unknown initial gas concentration unit: {cgas_unit}')
        T0 = f('T0', '50')
        if T0 <= -273.15:
            raise ValueError('T0 must be greater than -273.15 °C.')
        cgas_input = f('cgas_init', '0')
        cgas_uM = float(self._conc_to_uM(
            cgas_input, T0 + 273.15, cgas_unit, mw))

        temp = {'dt1': f('dt1', '60'), 'T0': T0,
                'Tfinal': f('Tfinal', '250'), 'RR': f('RR', '5')}
        tfinal = f('tfinal', '150')
        flow = {'tEq': f('tEq', '30'), 'tFlush': tfinal, 'Q': f('Q', '20')}
        feed = {'n_steps': as_int('n_steps', '0'),
                'delta': f('delta', '1000'),
                'step_time': f('step_time', '240'),
                'base_conc': f('base_conc', '0'),
                'hold_time_initial': f('hold_time_initial', '60'),
                'hold_time_final': f('hold_time_final', '3000'),
                'feed_tank_volume': f('feed_tank_volume', '0')}
        grid_scheme = str(val('grid_scheme', 'uniform'))
        grid_stretch = f('grid_stretch', '2.0')
        _validate_simulation_inputs(
            tfinal, temp, flow, feed, m_value, f('R', '200') * 1e-4,
            f('mSample', '50'), f('rhoSample', '1'), f('Vvessel', '10'),
            mw, src_array, f('EaK', '-35'), f('K_ref', '150'),
            f('D_ref', '1e-7'), f('EaD', '15'), inf_float('cD', 'inf'),
            inf_float('F', 'inf'), f('c0free', '10'), cgas_uM,
            as_int('N', '201'), f('rtol', '1e-7'), f('atol', '1e-8'),
            f('Tref', '50'), grid_scheme, grid_stretch)

        for key, allowed, fallback in [
            ('conc_plot_unit', CONC_UNITS, 'ppbv'),
            ('conc_data_unit', CONC_UNITS, 'ppbv'),
            ('mass_plot_unit', MASS_UNITS, 'ng'),
            ('mass_data_unit', MASS_UNITS, 'ng'),
            ('conc_interp_method', ('linear','cubic','pchip','akima'), 'linear'),
            ('mass_interp_method', ('linear','cubic','pchip','akima'), 'linear'),
        ]:
            if str(val(key, fallback)) not in allowed:
                raise ValueError(f'Invalid {key}: {val(key, fallback)}')
        duplicate_policy = str(val('duplicate_policy', DUPLICATE_POLICIES[0]))
        if duplicate_policy not in DUPLICATE_POLICIES:
            raise ValueError(
                'duplicate_policy must be one of: ' + ', '.join(DUPLICATE_POLICIES))

        for key, resample_on in (('conc_interp_n', bool(val('conc_interp', False))),
                                 ('mass_interp_n', bool(val('mass_interp', False)))):
            clamp(key, '100', 2, MAX_RESAMPLE_POINTS, resample_on)

        sweep_on = bool(val('sweep_enabled', False))
        clamp('sweep_n', '5', 2, MAX_SWEEP_POINTS, sweep_on)
        if sweep_on:
            sweep_param = str(val('sweep_param', ''))
            if sweep_param not in self.SWEEP_PARAMS:
                raise ValueError(
                    'The sweep is enabled but its parameter '
                    f'{sweep_param!r} is not one this version offers.')
            s0, s1 = f('sweep_start', '1e-8'), f('sweep_stop', '1e-6')
            if bool(val('sweep_log', False)) and (s0 <= 0 or s1 <= 0):
                raise ValueError('Log sweep endpoints must be positive.')
            if s0 == s1:
                raise ValueError('Sweep start and stop must differ.')

        opt_target = {'Concentration (ppbv)': 'Concentration',
                      'Mass (ng)': 'Mass'}.get(
                          str(val('opt_target', 'Concentration')),
                          str(val('opt_target', 'Concentration')))
        if opt_target not in ('Concentration', 'Mass', 'Both'):
            raise ValueError('opt_target must be Concentration, Mass, or Both.')
        d['opt_target'] = opt_target

        opt_on = bool(val('opt_enabled', False))
        opt_n = clamp('opt_n', '1', 1, 6, opt_on)
        n_starts = clamp('opt_nstarts', '1', 1, MAX_OPT_STARTS, opt_on)
        max_evals = clamp('opt_maxiter', '200', 1, MAX_OPT_EVALS, opt_on)
        if n_starts * max_evals > MAX_OPT_TOTAL_EVALS:
            if opt_on:
                raise ValueError(
                    f'opt_nstarts × opt_maxiter cannot exceed '
                    f'{MAX_OPT_TOTAL_EVALS:,}.')
            max_evals = max(1, MAX_OPT_TOTAL_EVALS // max(n_starts, 1))
            d['opt_maxiter'] = str(max_evals)
            repairs.append(
                f'opt_nstarts × opt_maxiter exceeded {MAX_OPT_TOTAL_EVALS:,}; '
                f'opt_maxiter reduced to {max_evals}')
        for key in ('opt_conc_weight', 'opt_mass_weight'):
            if f(key, '1.0') <= 0:
                raise ValueError(f'{key} must be > 0.')
        valid_params = {k for k, (attr, _) in self.SWEEP_PARAMS.items()
                        if attr not in self.OPT_EXCLUDED_ATTRS}
        for index, row in enumerate(list(val('opt_rows', []))[:opt_n]):
            if not isinstance(row, dict):
                raise ValueError('Each optimization row must be an object.')
            label = row.get('param', '')
            if label and label not in valid_params:
                if opt_on:
                    raise ValueError(
                        f'Fit parameter {label!r} is not one this version can '
                        'optimize.')
                repairs.append(f'opt_rows[{index}] parameter {label!r} is not '
                               'available; left unset')
                row.pop('param', None)
            try:
                lo = float(row.get('min', '1e-9'))
                hi = float(row.get('max', '1e-5'))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'Fit bounds for row {index + 1} must be numbers.') from exc
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(
                    f'Fit bounds for row {index + 1} must be finite with '
                    'min < max.')
            if bool(row.get('log', True)) and lo <= 0:
                raise ValueError(
                    f'Log-scaled fit bounds for row {index + 1} must be '
                    'positive.')

        for key in ('time_min', 'time_max'):
            raw = str(val(key, '')).strip()
            if raw:
                try:
                    if not np.isfinite(float(raw)):
                        raise ValueError
                except ValueError as exc:
                    raise ValueError(
                        f'{key} must be blank or a finite number '
                        f'(got {raw!r}).') from exc

        self._last_payload_repairs = repairs
        return d

    def _apply_defaults_payload(self, d):
        """Apply an already validated parameter payload."""
        entry_defaults = {
            'tfinal':'150','N':'201','rtol':'1e-7','atol':'1e-8',
            'grid_stretch':'2.0','dt1':'60','T0':'50','Tfinal':'250','RR':'5',
            'tEq':'30','Q':'20','n_steps':'0','delta':'1000','step_time':'240',
            'base_conc':'0','hold_time_initial':'60','hold_time_final':'3000',
            'feed_tank_volume':'0','R':'200','mSample':'50','rhoSample':'1',
            'Vvessel':'10','MW_analyte':'18','K_ref':'150','EaK':'-35',
            'D_ref':'1e-7','EaD':'15','Tref':'50','cD':'inf','F':'inf',
            'c0free':'10','cgas_init':'0',
        }
        for key, fallback in entry_defaults.items():
            self._set_entry_value(getattr(self, key), d.get(key, fallback))

        gs = d.get('grid_scheme', 'uniform')
        self.grid_scheme.set(gs if gs in GRID_SCHEMES else 'uniform')
        geom = d.get('m', '2 - Sphere')
        self.m.set(geom if geom in self.m['values'] else '2 - Sphere')
        cgas_unit = d.get('cgas_init_unit', d.get('conc_plot_unit', 'ppbv'))
        self.cgas_init_unit.set(cgas_unit if cgas_unit in CONC_UNITS else 'ppbv')
        self._cgas_input_unit_last = self.cgas_init_unit.get()

        self._set_entry_value(self.num_sources, d.get('num_sources', '1'))
        self.update_source_fields()
        for i, src in enumerate(d.get('sources', [])):
            if i >= len(self.source_entries) or not isinstance(src, dict):
                break
            self._set_entry_value(self.source_entries[i][1], src.get('c0', '100'))
            self._set_entry_value(self.source_entries[i][3], src.get('A', '1e8'))
            self._set_entry_value(self.source_entries[i][5], src.get('Ea', '80'))

        self.conc_data_text.delete('1.0', tk.END)
        self.conc_data_text.insert('1.0', d.get('exp_conc_data', ''))
        self.mass_data_text.delete('1.0', tk.END)
        self.mass_data_text.insert('1.0', d.get('exp_mass_data', ''))
        self.conc_interp_var.set(bool(d.get('conc_interp', False)))
        self.mass_interp_var.set(bool(d.get('mass_interp', False)))
        self._set_entry_value(self.conc_interp_n, d.get('conc_interp_n', '100'))
        self._set_entry_value(self.mass_interp_n, d.get('mass_interp_n', '100'))
        self.conc_interp_method.set(d.get('conc_interp_method', 'linear'))
        self.mass_interp_method.set(d.get('mass_interp_method', 'linear'))
        self.conc_unit.set(d.get('conc_plot_unit', 'ppbv'))
        self.conc_data_unit.set(d.get('conc_data_unit', 'ppbv'))
        self.mass_unit.set(d.get('mass_plot_unit', 'ng'))
        self.mass_data_unit.set(d.get('mass_data_unit', 'ng'))
        duplicate_policy = d.get('duplicate_policy', DUPLICATE_POLICIES[0])
        self.duplicate_policy.set(
            duplicate_policy if duplicate_policy in DUPLICATE_POLICIES
            else DUPLICATE_POLICIES[0])

        self.sweep_enabled.set(bool(d.get('sweep_enabled', False)))
        sweep_param = d.get('sweep_param', '')
        if sweep_param in self.SWEEP_PARAMS:
            self.sweep_param_var.set(sweep_param)
        self._set_entry_value(self.sweep_start, d.get('sweep_start', '1e-8'))
        self._set_entry_value(self.sweep_stop, d.get('sweep_stop', '1e-6'))
        self._set_entry_value(self.sweep_n, d.get('sweep_n', '5'))
        self.sweep_log.set(bool(d.get('sweep_log', False)))
        self.sweep_cmap.set(d.get('sweep_cmap', 'viridis'))
        self.logtime_var.set(bool(d.get('logtime', False)))
        self.logconc_var.set(bool(d.get('logconc', False)))
        self._set_entry_value(self.time_min, d.get('time_min', ''))
        self._set_entry_value(self.time_max, d.get('time_max', ''))
        self._toggle_sweep_ui()

        self.opt_enabled.set(bool(d.get('opt_enabled', False)))
        self._set_entry_value(self.opt_n_entry, d.get('opt_n', '1'))
        self.opt_target.set({'Concentration (ppbv)': 'Concentration', 'Mass (ng)': 'Mass'}.get(d.get('opt_target', 'Concentration'), d.get('opt_target', 'Concentration')))
        self._set_entry_value(self.opt_conc_weight, d.get('opt_conc_weight', '1.0'))
        self._set_entry_value(self.opt_mass_weight, d.get('opt_mass_weight', '1.0'))
        self._set_entry_value(self.opt_nstarts, d.get('opt_nstarts', '1'))
        self._set_entry_value(self.opt_maxiter, d.get('opt_maxiter', '200'))
        self._build_opt_rows()
        for rd, row in zip(self.opt_param_rows, d.get('opt_rows', [])):
            if not isinstance(row, dict):
                continue
            if row.get('param') in rd['combo']['values']:
                rd['combo'].set(row['param'])
            self._set_entry_value(rd['min_entry'], row.get('min', '1e-9'))
            self._set_entry_value(rd['max_entry'], row.get('max', '1e-5'))
            rd['log_var'].set(bool(row.get('log', True)))
        self._toggle_opt_ui()
        self._update_derived()

    def save_defaults(self, path=None):
        """Validate and atomically save the complete GUI state."""
        if self._busy:
            return False
        target = os.path.abspath(path or self.defaults_file)
        try:
            payload = self._validate_defaults_payload(self._defaults_dict_from_gui())
            payload['schema_version'] = CONFIG_SCHEMA_VERSION
            self._atomic_write_json(target, payload)
            self.status_label.config(
                text=f"Saved to {os.path.basename(target)}", foreground='green')
            if path is None:
                self._update_defaults_label()
            messagebox.showinfo('Save Parameters', f'Parameters saved to:\n{target}')
            return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._log_exception('saving parameters', exc)
            self.status_label.config(text=f'Error saving parameters: {exc}', foreground='red')
            messagebox.showerror('Save Error', f'Failed to save parameters:\n{exc}')
            return False
    

    def load_defaults(self, path=None):
        """Validate an entire JSON file, then apply it as one transaction."""
        target = os.path.abspath(path or self.defaults_file)
        if not os.path.exists(target):
            return False
        backup = None
        try:
            backup = self._defaults_dict_from_gui()
            with open(target, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            validated = self._validate_defaults_payload(payload)
            repairs = list(getattr(self, '_last_payload_repairs', ()))
            self._apply_defaults_payload(validated)
            # Report anything that had to be adjusted, rather than changing the
            # user's settings silently.  These are only ever settings belonging
            # to a switched-off feature, or a schema migration.
            note = f"  ({len(repairs)} setting(s) adjusted)" if repairs else ''
            self.status_label.config(
                text=f"Loaded: {os.path.basename(target)}{note}",
                foreground='#b36b00' if repairs else 'green')
            if repairs and path is not None:
                messagebox.showinfo(
                    'Parameter file adjusted',
                    'The file loaded, but these settings had to be adjusted:\n\n'
                    + '\n'.join(f'  \u2022 {r}' for r in repairs[:12])
                    + ('\n  ...' if len(repairs) > 12 else '')
                    + '\n\nSave the file to store the corrected values.')
            return True
        except (OSError, json.JSONDecodeError, ValueError, TypeError, tk.TclError) as exc:
            self._log_exception('loading parameters', exc)
            if backup is not None:
                try:
                    self._apply_defaults_payload(backup)
                except (ValueError, tk.TclError):
                    pass
            self.status_label.config(text='Parameter file was not applied.', foreground='red')
            if path is not None:
                messagebox.showerror('Load Error', f'Failed to load parameters:\n{exc}')
            return False
    
    def update_source_fields(self):
        if getattr(self, '_busy', False):
            return
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


    def _plot_pane_floor(self):
        """Minimum width the plot pane needs: docked data panel + a usable canvas.

        The data panel's requested width is what grid actually gives it -- a
        column `minsize` is a floor, not a cap, so it cannot be used to make the
        panel narrower than its contents ask for.  The floor therefore has to be
        read from the panel itself.
        """
        panel = getattr(self, 'exp_frame', None)
        width = EXP_PANEL_MIN_WIDTH
        if panel is not None:
            try:
                width = max(EXP_PANEL_FLOOR_WIDTH, panel.winfo_reqwidth())
            except tk.TclError:
                pass
        return width + PLOT_CANVAS_HARD_MIN

    def _clamp_sash(self, event=None):
        """Stop the controls pane from squeezing the canvas out of existence.

        Without this, shrinking the window took every pixel from the canvas --
        it reached 56 px wide and matplotlib warned that the axes had collapsed
        to zero.  The controls pane is inside a scrolled canvas with both
        scrollbars, so narrowing it only means scrolling, never losing access.
        """
        try:
            total = self.main_pane.winfo_width()
            if total <= 1:
                return
            floor = self._plot_pane_floor()
            if total - self.main_pane.sashpos(0) < floor:
                self.main_pane.sashpos(0, max(1, total - floor))
        except (tk.TclError, IndexError):
            pass

    def create_plots(self):
        """Create the plot grid with the data-entry panel docked on the right."""
        plots_frame = ttk.Frame(self.plot_frame)
        plots_frame.pack(fill=tk.BOTH, expand=True)
        # The data-entry panel sits to the RIGHT of the canvas rather than below
        # it, so the full window height is available to the plots and the gas
        # concentration panel can be given the majority of it.
        plots_frame.rowconfigure(0, weight=1)
        plots_frame.columnconfigure(0, weight=1)      # canvas: takes the slack
        plots_frame.columnconfigure(1, weight=0, minsize=EXP_PANEL_MIN_WIDTH)

        self.fig = Figure(figsize=(12.5, 11.5), dpi=100, constrained_layout=True)
        try:
            self.fig.set_constrained_layout_pads(
                w_pad=0.06, h_pad=0.06, wspace=0.04, hspace=0.06)
        except (AttributeError, TypeError):
            pass
        # Gas concentration (ax7) is the primary result, so it gets its own band
        # roughly as tall as the two rows of supporting panels combined.  Tune
        # CONC_PANEL_HEIGHT_RATIO to change the split without touching anything
        # else: the value is the height of the ax7 row relative to one small row.
        gs = self.fig.add_gridspec(
            3, 3, height_ratios=[1.0, 1.0, CONC_PANEL_HEIGHT_RATIO])
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[0, 2])
        self.ax4 = self.fig.add_subplot(gs[1, 0])
        self.ax5 = self.fig.add_subplot(gs[1, 1])
        self.ax6 = self.fig.add_subplot(gs[1, 2])
        self.ax7 = self.fig.add_subplot(gs[2, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.exp_frame = ttk.LabelFrame(
            plots_frame, text="Experimental Data", padding="8")
        self.exp_frame.grid(row=0, column=1, sticky='nsew', padx=(8, 0))
        # A fixed minsize on the data column would let a narrowing window crush
        # the canvas to nothing, so the column's floor tracks the space actually
        # available: the data panel yields width before the plots do.
        self._plots_frame = plots_frame
        # Re-clamp once the panel exists and its requested width is known.
        self.root.after_idle(self._clamp_sash)
        # One column, three rows: shared options on top, then the concentration
        # and mass blocks stacked.  Rows 1 and 2 share the vertical slack so the
        # two text boxes grow equally as the window gets taller.
        self.exp_frame.columnconfigure(0, weight=1)
        self.exp_frame.rowconfigure(0, weight=0)
        self.exp_frame.rowconfigure(1, weight=1)
        self.exp_frame.rowconfigure(2, weight=1)
        data_options = ttk.Frame(self.exp_frame)
        data_options.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(data_options, text='Duplicate times:').pack(side=tk.LEFT)
        self.duplicate_policy = ttk.Combobox(
            data_options, width=18, state='readonly', values=DUPLICATE_POLICIES)
        self.duplicate_policy.set(DUPLICATE_POLICIES[0])
        self.duplicate_policy.pack(side=tk.LEFT, padx=(4, 0))
        _Tooltip(
            self.duplicate_policy,
            'Retain replicates keeps every original observation for fitting. '
            'Average replicates combines equal-time observations before fitting. '
            'Reject duplicates requires every observation time to be unique.')

        conc_frame = ttk.Frame(self.exp_frame)
        conc_frame.grid(row=1, column=0, sticky='nsew', pady=(0, 8))
        conc_hdr = ttk.Frame(conc_frame)
        conc_hdr.pack(fill=tk.X)
        conc_hdr.columnconfigure(0, weight=1)
        ttk.Label(conc_hdr, text="Conc.: t, y[, σ]",
                  font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(conc_hdr, text="Units:").grid(
            row=0, column=1, sticky=tk.E, padx=(6, 2))
        self.conc_data_unit = ttk.Combobox(
            conc_hdr, width=7, state="readonly", values=CONC_UNITS)
        self.conc_data_unit.set('ppbv')
        self.conc_data_unit.grid(row=0, column=2, sticky=tk.E)
        self.conc_data_unit.bind(
            "<<ComboboxSelected>>", lambda e: self._refresh_plots())
        self.conc_data_text = tk.Text(
            conc_frame, height=EXP_TEXT_MIN_LINES, width=EXP_TEXT_WIDTH,
            font=('Courier', 9), undo=True)
        self.conc_data_text.pack(expand=True, fill=tk.BOTH)
        conc_interp_frame = ttk.Frame(conc_frame)
        conc_interp_frame.pack(fill=tk.X, pady=(2, 0))
        self.conc_interp_var = tk.BooleanVar(value=False)
        conc_chk = ttk.Checkbutton(
            conc_interp_frame, text="Resample",
            variable=self.conc_interp_var, command=self._refresh_plots)
        conc_chk.pack(side=tk.LEFT)
        _Tooltip(conc_chk, RESAMPLE_TOOLTIP)
        ttk.Label(conc_interp_frame, text="N:").pack(side=tk.LEFT, padx=(4, 0))
        self.conc_interp_n = ttk.Entry(conc_interp_frame, width=5)
        self.conc_interp_n.insert(0, "100")
        self.conc_interp_n.pack(side=tk.LEFT, padx=2)
        self.conc_interp_method = ttk.Combobox(
            conc_interp_frame, width=7, state='readonly',
            values=["linear", "cubic", "pchip", "akima"])
        self.conc_interp_method.set("linear")
        self.conc_interp_method.pack(side=tk.LEFT, padx=2)

        mass_frame = ttk.Frame(self.exp_frame)
        mass_frame.grid(row=2, column=0, sticky='nsew')
        mass_hdr = ttk.Frame(mass_frame)
        mass_hdr.pack(fill=tk.X)
        mass_hdr.columnconfigure(0, weight=1)
        ttk.Label(mass_hdr, text="Mass: t, y[, σ]",
                  font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(mass_hdr, text="Units:").grid(
            row=0, column=1, sticky=tk.E, padx=(6, 2))
        self.mass_data_unit = ttk.Combobox(
            mass_hdr, width=5, state="readonly", values=MASS_UNITS)
        self.mass_data_unit.set('ng')
        self.mass_data_unit.grid(row=0, column=2, sticky=tk.E)
        self.mass_data_unit.bind(
            "<<ComboboxSelected>>", lambda e: self._refresh_plots())
        self.mass_data_text = tk.Text(
            mass_frame, height=EXP_TEXT_MIN_LINES, width=EXP_TEXT_WIDTH,
            font=('Courier', 9), undo=True)
        self.mass_data_text.pack(expand=True, fill=tk.BOTH)
        mass_interp_frame = ttk.Frame(mass_frame)
        mass_interp_frame.pack(fill=tk.X, pady=(2, 0))
        self.mass_interp_var = tk.BooleanVar(value=False)
        mass_chk = ttk.Checkbutton(
            mass_interp_frame, text="Resample",
            variable=self.mass_interp_var, command=self._refresh_plots)
        mass_chk.pack(side=tk.LEFT)
        _Tooltip(mass_chk, RESAMPLE_TOOLTIP)
        ttk.Label(mass_interp_frame, text="N:").pack(side=tk.LEFT, padx=(4, 0))
        self.mass_interp_n = ttk.Entry(mass_interp_frame, width=5)
        self.mass_interp_n.insert(0, "100")
        self.mass_interp_n.pack(side=tk.LEFT, padx=2)
        self.mass_interp_method = ttk.Combobox(
            mass_interp_frame, width=7, state='readonly',
            values=["linear", "cubic", "pchip", "akima"])
        self.mass_interp_method.set("linear")
        self.mass_interp_method.pack(side=tk.LEFT, padx=2)
    

    def parse_experimental_data(self, text_widget, interp_var=None,
                                interp_n_widget=None, interp_method_widget=None,
                                apply_resampling=True, return_details=False,
                                duplicate_policy='Retain replicates'):
        """Parse time, value[, 1σ] data with explicit diagnostics.

        Original observations, including replicate times, are retained for
        fitting.  Resampling is display-only and uses a duplicate-averaged copy.
        """
        data_str = text_widget.get("1.0", tk.END).strip()
        policy = (duplicate_policy if duplicate_policy in DUPLICATE_POLICIES
                  else DUPLICATE_POLICIES[0])
        empty = dict(raw_time=None, raw_values=None, raw_sigma=None,
                     plot_time=None, plot_values=None, diagnostics=(),
                     raw_text=data_str, duplicate_policy=policy)
        if not data_str:
            return empty if return_details else (None, None)

        times, values, sigmas = [], [], []
        diagnostics = []
        for line_no, raw_line in enumerate(data_str.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if ',' in line:
                parts = [part.strip() for part in line.split(',')]
            elif '\t' in line:
                parts = [part.strip() for part in line.split('\t')]
            else:
                parts = line.split()
            if len(parts) < 2:
                diagnostics.append(f"Line {line_no}: expected at least two columns.")
                continue
            if len(parts) > 3:
                diagnostics.append(
                    f"Line {line_no}: ignored {len(parts) - 3} extra column(s).")
            try:
                t_val = float(parts[0])
                v_val = float(parts[1])
            except ValueError:
                # A single header row is common; report it rather than silently
                # treating arbitrary malformed rows as valid input.
                diagnostics.append(
                    f"Line {line_no}: nonnumeric time/value ({parts[0]!r}, {parts[1]!r}).")
                continue
            if not (np.isfinite(t_val) and np.isfinite(v_val)):
                diagnostics.append(f"Line {line_no}: time and value must be finite.")
                continue
            sigma = np.nan
            if len(parts) >= 3 and parts[2] != '':
                try:
                    sigma = float(parts[2])
                except ValueError:
                    diagnostics.append(
                        f"Line {line_no}: uncertainty {parts[2]!r} is not numeric; ignored.")
                    sigma = np.nan
                if np.isfinite(sigma) and sigma <= 0:
                    diagnostics.append(
                        f"Line {line_no}: uncertainty must be > 0; ignored.")
                    sigma = np.nan
                elif not np.isfinite(sigma):
                    diagnostics.append(
                        f"Line {line_no}: uncertainty must be finite; ignored.")
                    sigma = np.nan
            times.append(t_val)
            values.append(v_val)
            sigmas.append(sigma)

        if not times:
            details = dict(empty)
            details['diagnostics'] = tuple(diagnostics)
            return details if return_details else (None, None)

        t_arr = np.asarray(times, dtype=float)
        v_arr = np.asarray(values, dtype=float)
        s_arr = np.asarray(sigmas, dtype=float)
        order = np.argsort(t_arr, kind='mergesort')
        t_arr, v_arr, s_arr = t_arr[order], v_arr[order], s_arr[order]

        unique_times, inverse, counts = np.unique(
            t_arr, return_inverse=True, return_counts=True)
        n_duplicates = int(t_arr.size - unique_times.size)
        if n_duplicates and policy == 'Reject duplicates':
            repeated = unique_times[counts > 1]
            preview = ', '.join(f'{value:.10g}' for value in repeated[:8])
            if repeated.size > 8:
                preview += f', ... ({repeated.size} repeated times)'
            raise ValueError(
                'Duplicate observation times are disallowed by the selected '
                f'policy. Repeated time(s): {preview}')
        if n_duplicates and policy == 'Average replicates':
            averaged_values = np.empty(unique_times.size, dtype=float)
            averaged_sigma = np.full(unique_times.size, np.nan, dtype=float)
            weighted_groups = 0
            for group in range(unique_times.size):
                mask = inverse == group
                group_values = v_arr[mask]
                group_sigma = s_arr[mask]
                valid_sigma = np.isfinite(group_sigma) & (group_sigma > 0)
                if group_values.size > 1 and np.all(valid_sigma):
                    weights = 1.0 / group_sigma ** 2
                    averaged_values[group] = float(
                        np.sum(weights * group_values) / np.sum(weights))
                    averaged_sigma[group] = float(np.sqrt(1.0 / np.sum(weights)))
                    weighted_groups += 1
                else:
                    averaged_values[group] = float(np.mean(group_values))
                    if group_values.size == 1 and valid_sigma[0]:
                        averaged_sigma[group] = float(group_sigma[0])
                    elif group_values.size > 1:
                        standard_error = float(
                            np.std(group_values, ddof=1) / np.sqrt(group_values.size))
                        if np.isfinite(standard_error) and standard_error > 0:
                            averaged_sigma[group] = standard_error
            t_arr, v_arr, s_arr = unique_times, averaged_values, averaged_sigma
            diagnostics.append(
                f"Averaged {n_duplicates} replicate observation(s) into "
                f"{unique_times.size} unique times; {weighted_groups} replicate "
                "group(s) used inverse-variance weighting.")
        elif n_duplicates:
            diagnostics.append(
                f"Retained {n_duplicates} replicate observation(s) at duplicate times; "
                "display resampling averages replicates, fitting does not.")

        plot_t, plot_v = t_arr.copy(), v_arr.copy()
        do_interp = bool(apply_resampling and interp_var is not None and interp_var.get())
        if do_interp and t_arr.size >= 2:
            try:
                n_pts = int(interp_n_widget.get())
            except (ValueError, tk.TclError, AttributeError):
                n_pts = 100
                diagnostics.append("Invalid resample count; used 100 display points.")
            if n_pts < 2 or n_pts > MAX_RESAMPLE_POINTS:
                raise ValueError(
                    f"Display resampling must use 2 to {MAX_RESAMPLE_POINTS:,} points.")

            unique_t, inverse = np.unique(t_arr, return_inverse=True)
            counts = np.bincount(inverse)
            sums = np.bincount(inverse, weights=v_arr)
            unique_v = sums / counts
            if unique_t.size >= 2:
                plot_t = np.linspace(unique_t.min(), unique_t.max(), n_pts)
                method = (interp_method_widget.get()
                          if interp_method_widget is not None else 'linear')
                if method == 'akima' and unique_t.size < 5:
                    diagnostics.append(
                        "Akima requires at least five unique times; used linear interpolation.")
                    method = 'linear'
                try:
                    if method == 'linear':
                        plot_v = np.interp(plot_t, unique_t, unique_v)
                    elif method == 'cubic':
                        plot_v = CubicSpline(unique_t, unique_v)(plot_t)
                    elif method == 'pchip':
                        plot_v = PchipInterpolator(unique_t, unique_v)(plot_t)
                    elif method == 'akima':
                        plot_v = Akima1DInterpolator(unique_t, unique_v)(plot_t)
                    else:
                        plot_v = np.interp(plot_t, unique_t, unique_v)
                except (ValueError, FloatingPointError) as exc:
                    diagnostics.append(
                        f"{method} interpolation failed ({exc}); used linear interpolation.")
                    plot_v = np.interp(plot_t, unique_t, unique_v)

                # Cubic/Akima can overshoot nonnegative physical data.  Replace
                # that display curve with shape-preserving PCHIP, never the raw
                # observations used by the optimizer.
                if (np.all(unique_v >= 0) and np.any(plot_v < 0)
                        and unique_t.size >= 2):
                    diagnostics.append(
                        f"{method} interpolation went negative for nonnegative data; "
                        "used shape-preserving PCHIP for display.")
                    plot_v = PchipInterpolator(unique_t, unique_v)(plot_t)

        details = dict(raw_time=t_arr, raw_values=v_arr, raw_sigma=s_arr,
                       plot_time=plot_t, plot_values=np.asarray(plot_v, dtype=float),
                       diagnostics=tuple(diagnostics), raw_text=data_str,
                       duplicate_policy=policy)
        return details if return_details else (details['plot_time'], details['plot_values'])

    @staticmethod
    def _freeze_value(value):
        """Deep-copy model state and make NumPy arrays read-only."""
        if isinstance(value, np.ndarray):
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            return copied
        if isinstance(value, dict):
            return MappingProxyType({
                key: OutgassingGUI._freeze_value(val)
                for key, val in value.items()})
        if isinstance(value, list):
            return tuple(OutgassingGUI._freeze_value(val) for val in value)
        if isinstance(value, tuple):
            return tuple(OutgassingGUI._freeze_value(val) for val in value)
        return copy.deepcopy(value)

    def _make_run_record(self, mode, label, params, results, experimental,
                         started_at, sweep_parameter=None,
                         sweep_attribute=None, sweep_value=None):
        """Create a self-contained, immutable provenance record."""
        return RunRecord(
            mode=str(mode), label=str(label),
            params=self._freeze_value(params),
            results=self._freeze_value(results),
            experimental=experimental,
            started_at_utc=str(started_at),
            completed_at_utc=_utc_now_iso(),
            sweep_parameter=sweep_parameter,
            sweep_attribute=sweep_attribute,
            sweep_value=(None if sweep_value is None else float(sweep_value)),
            versions=self._freeze_value(_runtime_versions()),
        )

    @staticmethod
    def _labeled_from_records(records):
        return [(record.label, record.results) for record in records]

    def _base_params_dict(self, p=None):
        """Return export metadata from a run snapshot, never live widgets."""
        if p is None:
            p = self._collect_params()
        geometry = p.get('geometry_label', {
            0: '0 - Slab', 1: '1 - Cylinder', 2: '2 - Sphere'
        }.get(int(p['m']), str(p['m'])))
        cD = 'inf' if np.isposinf(p['cD']) else p['cD']
        F = 'inf' if np.isposinf(p['F']) else p['F']
        params = {
            'Application Version': APP_VERSION,
            'Final Time (min)': p['tfinal'],
            'Grid Points': p['N'],
            'Grid Scheme': p.get('grid_scheme', 'uniform'),
            'Grid Stretch': p.get('grid_stretch', 2.0),
            'Relative Tolerance': p['rtol'],
            'Absolute Tolerance': p['atol'],
            'Time at Initial Temp (min)': p['dt1'],
            'Initial Temp (C)': p['T0'],
            'Final Temp (C)': p['Tfinal'],
            'Ramp Rate (C/min)': p['RR'],
            'Initial time at Q=0 (min)': p['tEq'],
            'Flow Rate (ml/min)': p['Q'],
            'Feed n_steps': p['n_steps'],
            'Feed delta (ppbv)': p['delta'],
            'Feed step_time (min)': p['step_time'],
            'Feed base_conc (ppbv)': p['base_conc'],
            'Feed hold_time_initial (min)': p['hold_time_initial'],
            'Feed hold_time_final (min)': p['hold_time_final'],
            'Feed Profile Interpretation':
                'Periodic staircase; cycle repeats for the full simulation',
            'Feed cycle period (min)': (
                p['hold_time_initial'] + 2 * p['n_steps'] * p['step_time']
                + p['hold_time_final']),
            'Feed Tank Volume (ml)': p['feed_tank_volume'],
            'Feed Tank Flow Interpretation':
                'Independent continuous carrier flow (not chamber flow schedule)',
            'Geometry': geometry,
            'Sample Radius / Half-thickness (um)': p.get('R_um', p['R'] * 1e4),
            'Total sample mass (mg)': p['mSample'],
            'Density (g/ml)': p['rhoSample'],
            'Vessel Volume (ml)': p['Vvessel'],
            'MW (g/mol)': p['MW_analyte'],
            'Reference Temp (C)': p['Tref'],
            'K @ Tref': p['K_ref'],
            'Sorption Enthalpy (kJ/mol)': p['EaK'],
            'D @ Tref,c=0 (cm^2/s)': p['D_ref'],
            'Ea_D (kJ/mol)': p['EaD'],
            'Plasticizer power cD (uM)': cD,
            'Surface mass-transfer coeff F (cm/s)': F,
            'Initial mobile concentration (uM)': p['c0free'],
            'Initial gas concentration input': p.get('cgas_init_input'),
            'Initial gas concentration input unit': p.get('cgas_init_unit', 'uM'),
            'Initial gas concentration canonical (uM)': p['cgas_init'],
            'Concentration plot unit at run': p.get('conc_plot_unit', 'ppbv'),
            'Concentration data unit at run': p.get('conc_data_unit', 'ppbv'),
            'Mass plot unit at run': p.get('mass_plot_unit', 'ng'),
            'Mass data unit at run': p.get('mass_data_unit', 'ng'),
            'Log time at run': p.get('log_time', False),
            'Log concentration at run': p.get('log_concentration', False),
            'Time-axis minimum at run': p.get('time_min'),
            'Time-axis maximum at run': p.get('time_max'),
        }
        src = np.asarray(p.get('src_params', []), dtype=float)
        for i in range(src.size // 3):
            params[f'Source_{i+1}_c0 (uM)'] = src[3 * i]
            params[f'Source_{i+1}_A (1/min)'] = src[3 * i + 1]
            params[f'Source_{i+1}_Ea (kJ/mol)'] = src[3 * i + 2]
        return params

    def _derived_constants_dict(self, p=None):
        """Calculate exported derived quantities from the frozen run inputs."""
        if p is None:
            p = self._collect_params()

        def fmt(value):
            if value is None:
                return '—'
            if np.isposinf(value):
                return 'inf'
            return f'{float(value):.10g}'

        Q = float(p['Q'])
        feed_volume = float(p['feed_tank_volume'])
        tau_feed = (0.0 if feed_volume <= 0 else
                    (np.inf if Q <= 0 else feed_volume / Q))
        feed_cycle = (float(p['hold_time_initial'])
                      + 2 * int(p['n_steps']) * float(p['step_time'])
                      + float(p['hold_time_final']))
        sample_volume = float(p['mSample']) * 1e-3 / float(p['rhoSample'])
        headspace = float(p['Vvessel']) - sample_volume
        tau_vessel = (None if headspace <= 0 else
                      (np.inf if Q <= 0 else headspace / Q))
        tau_diff = ((float(p['R']) ** 2 / float(p['D_ref'])) / 60.0
                    if p['R'] > 0 and p['D_ref'] > 0 else None)
        # Mass-transfer Biot number Bi = F*R/D and purge Peclet number
        # Pe = Q*R^2/(V_head*D) = tau_diff/tau_vessel, both at Tref and c = 0.
        # p['R'] is already in cm, D_ref in cm^2/s and F in cm/s.
        biot = None
        if p['R'] > 0 and p['D_ref'] > 0 and p['F'] >= 0:
            biot = (np.inf if np.isposinf(p['F'])
                    else float(p['F']) * float(p['R']) / float(p['D_ref']))
        peclet = None
        if tau_diff is not None and headspace > 0:
            peclet = 0.0 if Q <= 0 else tau_diff * Q / headspace
        beta = headspace / sample_volume if sample_volume > 0 and headspace > 0 else None
        c_eq = (float(p['c0free']) / (float(p['K_ref']) + beta)
                if beta is not None and float(p['K_ref']) + beta > 0 else None)
        noflow = None
        noflow_unit = p.get('conc_plot_unit', 'ppbv')
        if c_eq is not None:
            noflow = float(self._conc_from_uM(
                c_eq, float(p['Tref']) + 273.15, noflow_unit,
                float(p['MW_analyte'])))
        constants = {
            'Feed tank residence time, tau_f (min)': fmt(tau_feed),
            'Repeating feed-cycle period (min)': fmt(feed_cycle),
            'Vessel gas residence time, tau_v (min)': fmt(tau_vessel),
            'Headspace volume (ml)': fmt(headspace),
            'Sample volume (ml)': fmt(sample_volume),
            'Characteristic diffusion time, R^2/D (min)': fmt(tau_diff),
            'Biot number, Bi = F*R/D': fmt(biot),
            'Peclet number, Pe = Q*R^2/(V_head*D)': fmt(peclet),
            'Phase ratio, beta': fmt(beta),
            f'No-flow equilibrium concentration ({noflow_unit})': fmt(noflow),
            'Initial mobile, mass basis': self._mass_fraction_str(
                p['c0free'], p['MW_analyte'], p['rhoSample']),
        }
        src = np.asarray(p.get('src_params', []), dtype=float)
        for i in range(src.size // 3):
            constants[f'Source_{i+1}_c0, mass basis'] = self._mass_fraction_str(
                src[3 * i], p['MW_analyte'], p['rhoSample'])
        return constants

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
        """R² at the original finite observations within the model time range."""
        if exp_time is None or exp_data is None:
            return None
        sim_time = np.asarray(sim_time, dtype=float)
        sim_data = np.asarray(sim_data, dtype=float)
        exp_time = np.asarray(exp_time, dtype=float)
        exp_data = np.asarray(exp_data, dtype=float)
        if sim_time.size < 2 or exp_time.size < 2:
            return None
        valid_sim = np.isfinite(sim_time) & np.isfinite(sim_data)
        if np.count_nonzero(valid_sim) < 2:
            return None
        sim_time = sim_time[valid_sim]
        sim_data = sim_data[valid_sim]
        order = np.argsort(sim_time, kind='mergesort')
        sim_time, sim_data = sim_time[order], sim_data[order]
        valid_exp = (np.isfinite(exp_time) & np.isfinite(exp_data)
                     & (exp_time >= sim_time[0]) & (exp_time <= sim_time[-1]))
        t_exp, y_exp = exp_time[valid_exp], exp_data[valid_exp]
        if t_exp.size < 2:
            return None
        y_sim = np.interp(t_exp, sim_time, sim_data)
        ss_res = float(np.sum((y_exp - y_sim) ** 2))
        ss_tot = float(np.sum((y_exp - np.mean(y_exp)) ** 2))
        scale = max(float(np.sum(y_exp ** 2)), 1.0)
        if ss_tot <= np.finfo(float).eps * scale:
            return None
        return 1.0 - ss_res / ss_tot

    def _export_individual_plots(self, stem, labeled_results,
                                 experimental=None, result_mode='single',
                                 sweep_label=None, plot_unit=None,
                                 mass_plot_unit=None, plot_settings=None):
        """Save seven provenance-consistent, 300-dpi plot panels."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        experimental = experimental or self._last_experimental_snapshot
        plot_unit = plot_unit or self._conc_plot_unit()
        mass_plot_unit = mass_plot_unit or self._mass_plot_unit()
        is_sweep = result_mode == 'sweep'
        is_multi = len(labeled_results) > 1
        short_label = (sweep_label or '').split('–')[-1].strip()
        log_t, left_limit, right_limit = self._time_axis_limits(
            labeled_results, settings=plot_settings)
        log_conc = (bool(plot_settings.get('log_concentration', False))
                    if plot_settings is not None else
                    bool(getattr(self, 'logconc_var', None)
                         and self.logconc_var.get()))

        conc_series = (experimental.concentration if experimental else ExperimentalSeries())
        mass_series = (experimental.mass if experimental else ExperimentalSeries())
        reference_results = labeled_results[0][1] if labeled_results else None
        exp_conc_plot = (self._exp_conc_convert(
            conc_series.raw_time, conc_series.raw_values,
            conc_series.unit, plot_unit, reference_results)
            if conc_series.raw_time is not None else None)
        exp_mass_plot = (self._exp_mass_convert(
            mass_series.raw_values, mass_series.unit, mass_plot_unit)
            if mass_series.raw_time is not None else None)

        panels = [
            ('temperature', 'T_C', 'Temperature, $T$ (°C)', False, None),
            ('feed_conc', 'c_feed_ppbv',
             self._feed_axis_label(plot_unit, export=True), False, None),
            ('flow_rate', 'Q', 'Flow rate, $Q$ (mL min$^{-1}$)', False, None),
            ('diffusivity', 'D', 'Diffusivity, $D$ (cm$^2$ s$^{-1}$)', True, None),
            ('solubility', 'S',
             'Solubility, $S$ (cm$^3$(STP) cm$^{-3}$ atm$^{-1}$)', False, None),
            ('mass_change', 'delta_mass',
             self._mass_axis_label(mass_plot_unit, export=True), False, 'mass'),
            ('gas_conc', 'c_gas_ppbv',
             self._conc_axis_label(plot_unit, export=True), log_conc, 'conc'),
        ]
        written = []
        for suffix, key, ylabel, ylog, exp_key in panels:
            fig = Figure(figsize=(8, 6), constrained_layout=True)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            for label, results in labeled_results:
                if suffix == 'gas_conc':
                    y_data = self._model_conc_in(results, plot_unit)
                elif suffix == 'feed_conc':
                    y_data = self._model_feed_in(results, plot_unit)
                elif suffix == 'mass_change':
                    y_data = self._model_mass_in(results, mass_plot_unit)
                else:
                    y_data = results[key]
                kwargs = dict(linewidth=3,
                              label=(label if (is_sweep or is_multi) else 'Model'))
                color = results.get('_sweep_color')
                if color is not None:
                    kwargs['color'] = color
                ax.plot(results['t'], y_data, **kwargs)

            if suffix == 'mass_change':
                ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
            has_experimental = False
            if (not is_sweep and exp_key == 'conc'
                    and conc_series.raw_time is not None):
                ax.plot(conc_series.raw_time, exp_conc_plot, 'o', color='black',
                        markersize=9, label='Experimental', zorder=10)
                has_experimental = True
            elif (not is_sweep and exp_key == 'mass'
                  and mass_series.raw_time is not None):
                ax.plot(mass_series.raw_time, exp_mass_plot, 'o', color='black',
                        markersize=9, label='Experimental', zorder=10)
                has_experimental = True

            if ylog:
                ax.set_yscale('log')
            if log_t:
                ax.set_xscale('log')
            if left_limit is not None or right_limit is not None:
                ax.set_xlim(left=left_limit, right=right_limit)

            if suffix == 'gas_conc' and ylog:
                positive = []
                for _, results in labeled_results:
                    values = np.asarray(self._model_conc_in(results, plot_unit), dtype=float)
                    positive.extend(values[np.isfinite(values) & (values > 0)])
                if exp_conc_plot is not None:
                    values = np.asarray(exp_conc_plot, dtype=float)
                    positive.extend(values[np.isfinite(values) & (values > 0)])
                if positive:
                    minimum = float(np.min(positive))
                    maximum = float(np.max(positive))
                    ax.set_ylim(bottom=(minimum / 1.2 if maximum > minimum else minimum / 10))

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)
                spine.set_color('black')
            ax.minorticks_on()
            ax.tick_params(which='major', direction='in', length=9, width=2,
                           top=True, bottom=True, left=True, right=True,
                           labeltop=False, labelright=False, labelsize=18)
            ax.tick_params(which='minor', direction='in', length=5, width=1.5,
                           top=True, bottom=True, left=True, right=True)
            ax.grid(False)
            ax.set_xlabel('Time, $t$ (min)', fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            if is_sweep or is_multi or has_experimental:
                legend = ax.legend(
                    loc='best', fontsize=15, frameon=True, edgecolor='black',
                    framealpha=1.0, fancybox=False,
                    title=(short_label if is_sweep else None), title_fontsize=15,
                    ncol=2 if ((is_sweep or is_multi)
                               and len(labeled_results) > 8) else 1)
                legend.get_frame().set_linewidth(1.0)
            output = f'{stem}_{suffix}.png'
            fig.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.08)
            written.append(output)
            fig.clear()
        return written

    def export_data(self):
        """Export the exact completed run records rather than current GUI state."""
        if getattr(self, '_busy', False):
            return
        is_sweep = (self._last_result_mode == 'sweep'
                    and bool(self.sweep_run_records))
        records = (list(self.sweep_run_records) if is_sweep else
                   ([self.current_run_record] if self.current_run_record else []))
        if not records:
            messagebox.showwarning('No Data',
                                   'Please run a simulation or fit before exporting data.')
            return
        filename = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='Export Simulation Results')
        if not filename:
            return

        try:
            first = records[0]
            experimental = first.experimental
            params = self._base_params_dict(first.params)
            constants = self._derived_constants_dict(first.params)
            conc_series = experimental.concentration
            mass_series = experimental.mass
            conc_unit = conc_series.unit or first.params.get('conc_data_unit', 'ppbv')
            mass_unit = mass_series.unit or first.params.get('mass_data_unit', 'ng')

            def r2_concentration(record):
                if conc_series.raw_time is None:
                    return None
                return self._compute_r_squared(
                    record.results['t'],
                    self._model_conc_in(record.results, conc_unit),
                    conc_series.raw_time, conc_series.raw_values)

            def r2_mass(record):
                if mass_series.raw_time is None:
                    return None
                return self._compute_r_squared(
                    record.results['t'],
                    self._model_mass_in(record.results, mass_unit),
                    mass_series.raw_time, mass_series.raw_values)

            def write_experimental(handle):
                if (conc_series.raw_time is None and mass_series.raw_time is None):
                    return
                handle.write(
                    '\n# FIT DATA (after duplicate-time policy; no resampled pseudo-points):\n')
                handle.write(
                    f'# Duplicate-time policy: {conc_series.duplicate_policy}\n')
                if conc_series.raw_time is not None:
                    data = {'time (min)': conc_series.raw_time,
                            f'concentration ({conc_unit})': conc_series.raw_values}
                    if (conc_series.raw_sigma is not None
                            and np.any(np.isfinite(conc_series.raw_sigma))):
                        data[f'concentration 1sigma ({conc_unit})'] = conc_series.raw_sigma
                    handle.write('# Concentration Data:\n')
                    pd.DataFrame(data).to_csv(handle, index=False, lineterminator='\n')
                if mass_series.raw_time is not None:
                    data = {'time (min)': mass_series.raw_time,
                            f'mass ({mass_unit})': mass_series.raw_values}
                    if (mass_series.raw_sigma is not None
                            and np.any(np.isfinite(mass_series.raw_sigma))):
                        data[f'mass 1sigma ({mass_unit})'] = mass_series.raw_sigma
                    handle.write('# Mass Data:\n')
                    pd.DataFrame(data).to_csv(handle, index=False, lineterminator='\n')

            with open(filename, 'w', encoding='utf-8', newline='') as handle:
                handle.write('# Outgassing Model Simulation Results\n')
                handle.write(f'# Release: {LLNL_RELEASE_ID}\n')
                handle.write(f'# Application Version: {APP_VERSION}\n')
                handle.write(f'# Result Mode: {"sweep" if is_sweep else first.mode}\n')
                handle.write(f'# Started UTC: {first.started_at_utc}\n')
                handle.write(f'# Completed UTC: {records[-1].completed_at_utc}\n')
                for key, value in (first.versions or {}).items():
                    handle.write(f'# Runtime {key}: {value}\n')
                if is_sweep:
                    handle.write(f'# Sweep Parameter: {first.sweep_parameter}\n')
                    handle.write(f'# Successful Sweep Points: {len(records)}\n')
                handle.write('#\n# RUN PARAMETERS (captured at launch):\n')
                for key, value in params.items():
                    handle.write(f'#   {key}: {value}\n')
                handle.write('#\n# CALCULATED CONSTANTS (from captured inputs):\n')
                for key, value in constants.items():
                    handle.write(f'#   {key}: {value}\n')

                if not is_sweep:
                    record = first
                    result = record.results
                    handle.write(f"#\n# Mass Balance Residual Absolute: "
                                 f"{result.get('mass_balance_residual', np.nan):.12g}\n")
                    handle.write(f"# Net Boundary Flux Integral: "
                                 f"{result.get('purge_integral', np.nan):.12g}\n")
                    handle.write(f"# Absolute Boundary Flux Integral: "
                                 f"{result.get('absolute_purge_integral', np.nan):.12g}\n")
                    handle.write(f"# Mass Balance Scale: "
                                 f"{result.get('mass_balance_scale', np.nan):.12g}\n")
                    handle.write(f"# Mass Balance Error (%): "
                                 f"{result['mass_bal_error']:.12g}\n")
                    handle.write(f"# Solve Time (s): {result['solve_time']:.8g}\n")
                    handle.write(f"# Solver Segments: {result.get('n_segments', '')}\n")
                    handle.write(f"# Solver Function Evaluations: {result.get('nfev', '')}\n")
                    handle.write(f"# Solver Jacobian Evaluations: {result.get('njev', '')}\n")
                    handle.write(f"# Solver LU Factorizations: {result.get('nlu', '')}\n")
                    handle.write(f"# Minimum Concentration Before Clipping: "
                                 f"{result.get('minimum_concentration_before_clip', np.nan):.12g}\n")
                    handle.write(f"# Clipped Negative State Count: "
                                 f"{result.get('clipped_negative_count', 0)}\n")
                    r2c, r2m = r2_concentration(record), r2_mass(record)
                    if r2c is not None:
                        handle.write(f'# R-squared Concentration: {r2c:.10g}\n')
                    if r2m is not None:
                        handle.write(f'# R-squared Mass: {r2m:.10g}\n')
                    handle.write('#\n# DATA:\n')
                    self._results_to_df(result).to_csv(
                        handle, index=False, lineterminator='\n')
                    write_experimental(handle)
                else:
                    handle.write('#\n# SWEEP SUMMARY (exact numeric values):\n')
                    summary = []
                    for record in records:
                        result = record.results
                        mass = np.asarray(result['delta_mass'], dtype=float)
                        gas_ugm3 = np.asarray(
                            self._model_conc_in(result, 'µg/m³'), dtype=float)
                        row = {
                            'sweep_label': record.label,
                            'sweep_value_exact': record.sweep_value,
                            'sweep_attribute': record.sweep_attribute,
                            'mass_balance_residual_absolute': result.get(
                                'mass_balance_residual', np.nan),
                            'net_boundary_flux_integral': result.get(
                                'purge_integral', np.nan),
                            'absolute_boundary_flux_integral': result.get(
                                'absolute_purge_integral', np.nan),
                            'mass_balance_scale': result.get(
                                'mass_balance_scale', np.nan),
                            'mass_balance_error (%)': result['mass_bal_error'],
                            'solve_time (s)': result['solve_time'],
                            'max_c_gas (ppbv)': float(np.nanmax(result['c_gas_ppbv'])),
                            'final_c_gas (ppbv)': float(result['c_gas_ppbv'][-1]),
                            'max_c_gas (ug/m3)': float(np.nanmax(gas_ugm3)),
                            'final_c_gas (ug/m3)': float(gas_ugm3[-1]),
                            'min_delta_m (ng)': float(np.nanmin(mass)),
                            'max_delta_m (ng)': float(np.nanmax(mass)),
                            'max_abs_delta_m (ng)': float(np.nanmax(np.abs(mass))),
                            'peak_mass_lost (ng)': max(0.0, -float(np.nanmin(mass))),
                            'peak_mass_gained (ng)': max(0.0, float(np.nanmax(mass))),
                            'final_delta_m (ng)': float(mass[-1]),
                        }
                        r2c, r2m = r2_concentration(record), r2_mass(record)
                        if r2c is not None:
                            row['R2_concentration'] = r2c
                        if r2m is not None:
                            row['R2_mass'] = r2m
                        summary.append(row)
                    pd.DataFrame(summary).to_csv(
                        handle, index=False, lineterminator='\n')
                    handle.write('\n# PER-RUN DATA:\n')
                    for record in records:
                        result = record.results
                        handle.write(
                            f"# --- {record.sweep_attribute} exact="
                            f"{record.sweep_value:.17g}; label={record.label}; "
                            f"residual={result['mass_bal_error']:.6e}%; "
                            f"solve={result['solve_time']:.6g}s ---\n")
                        self._results_to_df(result).to_csv(
                            handle, index=False, lineterminator='\n')
                    write_experimental(handle)

            stem = filename.rsplit('.', 1)[0]
            labeled = self._labeled_from_records(records)
            plot_files = self._export_individual_plots(
                stem, labeled, experimental=experimental,
                result_mode=('sweep' if is_sweep else first.mode),
                sweep_label=(first.sweep_parameter if is_sweep else None),
                plot_unit=first.params.get('conc_plot_unit', 'ppbv'),
                mass_plot_unit=first.params.get('mass_plot_unit', 'ng'),
                plot_settings=first.params)
            listing = '\n'.join('  ' + os.path.basename(item) for item in plot_files)
            messagebox.showinfo(
                'Export Success',
                f"{'Sweep' if is_sweep else 'Run'} data exported "
                f"({len(records)} run(s)):\n{os.path.basename(filename)}\n\n"
                f'{len(plot_files)} plots saved alongside it:\n{listing}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._log_exception('exporting results', exc)
            messagebox.showerror('Export Error', f'Failed to export data:\n{exc}')

    def _collect_params(self):
        """Return a validated, widget-free snapshot of all run inputs."""
        def finite(widget, label):
            try:
                value = float(widget.get())
            except (ValueError, tk.TclError) as exc:
                raise ValueError(f'{label} must be numeric.') from exc
            if not np.isfinite(value):
                raise ValueError(f'{label} must be finite.')
            return value

        def integer(widget, label):
            raw = widget.get().strip()
            try:
                value = int(raw)
            except (ValueError, tk.TclError) as exc:
                raise ValueError(f'{label} must be an integer.') from exc
            return value

        def positive_infinity_or_finite(widget, label):
            raw = widget.get().strip().lower()
            if raw in ('inf', '+inf', 'infinity', '+infinity'):
                return np.inf
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f'{label} must be finite or +inf.') from exc
            if not np.isfinite(value):
                raise ValueError(f'{label} must be finite or +inf.')
            return value

        tfinal = finite(self.tfinal, 'Final Time')
        geometry_label = self.m.get()
        try:
            geometry = int(geometry_label.split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError('Select a valid geometry.') from exc
        mw = finite(self.MW_analyte, 'Molecular weight')
        if mw <= 0:
            raise ValueError('Molecular weight must be > 0.')
        T0 = finite(self.T0, 'Initial temperature')
        if T0 <= -273.15:
            raise ValueError('Initial temperature must be greater than -273.15 °C.')
        cgas_input = finite(self.cgas_init, 'Initial gas concentration')
        cgas_unit = self._cgas_input_unit()
        cgas_uM = float(self._conc_to_uM(
            cgas_input, T0 + 273.15, cgas_unit, mw))

        src_values = []
        for i, source in enumerate(self.source_entries, start=1):
            src_values.extend((
                finite(source[1], f'Source {i} c0'),
                finite(source[3], f'Source {i} A'),
                finite(source[5], f'Source {i} Ea'),
            ))
        src_params = np.asarray(src_values, dtype=float)

        def optional_axis(widget, label):
            raw = widget.get().strip()
            if not raw:
                return None
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f'{label} must be blank or finite.') from exc
            if not np.isfinite(value):
                raise ValueError(f'{label} must be blank or finite.')
            return value

        p = dict(
            tfinal=tfinal,
            N=integer(self.N, 'Grid Points'),
            rtol=finite(self.rtol, 'Relative tolerance'),
            atol=finite(self.atol, 'Absolute tolerance'),
            grid_scheme=self.grid_scheme.get(),
            grid_stretch=finite(self.grid_stretch, 'Grid stretch'),
            dt1=finite(self.dt1, 'Time at initial temperature'),
            T0=T0,
            Tfinal=finite(self.Tfinal, 'Final temperature'),
            RR=finite(self.RR, 'Ramp rate'),
            tEq=finite(self.tEq, 'Initial zero-flow time'),
            tFlush=tfinal,
            Q=finite(self.Q, 'Flow rate'),
            n_steps=integer(self.n_steps, 'Feed steps'),
            delta=finite(self.delta, 'Feed increment'),
            step_time=finite(self.step_time, 'Feed step time'),
            base_conc=finite(self.base_conc, 'Feed base concentration'),
            hold_time_initial=finite(self.hold_time_initial, 'Initial feed hold'),
            hold_time_final=finite(self.hold_time_final, 'Final feed hold'),
            feed_tank_volume=finite(self.feed_tank_volume, 'Feed tank volume'),
            m=geometry, geometry_label=geometry_label,
            R=finite(self.R, 'Sample radius / half-thickness') * 1e-4,
            R_um=finite(self.R, 'Sample radius / half-thickness'),
            mSample=finite(self.mSample, 'Sample mass'),
            rhoSample=finite(self.rhoSample, 'Sample density'),
            Vvessel=finite(self.Vvessel, 'Vessel volume'),
            MW_analyte=mw,
            EaK=finite(self.EaK, 'Sorption enthalpy'),
            K_ref=finite(self.K_ref, 'Partition coefficient'),
            D_ref=finite(self.D_ref, 'Reference diffusivity'),
            EaD=finite(self.EaD, 'Diffusion activation energy'),
            Tref=finite(self.Tref, 'Reference temperature'),
            c0free=finite(self.c0free, 'Initial mobile concentration'),
            cgas_init=cgas_uM,
            cgas_init_input=cgas_input,
            cgas_init_unit=cgas_unit,
            cD=positive_infinity_or_finite(self.cD, 'Plasticizer power cD'),
            F=positive_infinity_or_finite(self.F, 'Mass-transfer coefficient F'),
            src_params=src_params,
            conc_plot_unit=self._conc_plot_unit(),
            conc_data_unit=self._conc_cmp_unit(),
            mass_plot_unit=self._mass_plot_unit(),
            mass_data_unit=self._mass_cmp_unit(),
            log_time=bool(self.logtime_var.get()),
            log_concentration=bool(self.logconc_var.get()),
            time_min=optional_axis(self.time_min, 'Time-axis minimum'),
            time_max=optional_axis(self.time_max, 'Time-axis maximum'),
            sweep_colormap=self.sweep_cmap.get(),
        )
        temp = {'dt1': p['dt1'], 'T0': p['T0'],
                'Tfinal': p['Tfinal'], 'RR': p['RR']}
        flow = {'tEq': p['tEq'], 'tFlush': p['tFlush'], 'Q': p['Q']}
        feed = {key: p[key] for key in (
            'n_steps', 'delta', 'step_time', 'base_conc',
            'hold_time_initial', 'hold_time_final', 'feed_tank_volume')}
        _validate_simulation_inputs(
            p['tfinal'], temp, flow, feed, p['m'], p['R'], p['mSample'],
            p['rhoSample'], p['Vvessel'], p['MW_analyte'], p['src_params'],
            p['EaK'], p['K_ref'], p['D_ref'], p['EaD'], p['cD'], p['F'],
            p['c0free'], p['cgas_init'], p['N'], p['rtol'], p['atol'],
            p['Tref'], p['grid_scheme'], p['grid_stretch'])
        return p

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
        """Enable or disable optimization widgets without overriding busy state."""
        if getattr(self, '_busy', False):
            return
        enabled = self.opt_enabled.get()
        state = 'normal' if enabled else 'disabled'
        readonly = 'readonly' if enabled else 'disabled'
        for widget in (self.opt_n_entry, self.opt_build_btn, self.opt_run_btn,
                       self.opt_maxiter, self.opt_nstarts,
                       self.opt_conc_weight, self.opt_mass_weight):
            widget.config(state=state)
        self.opt_stop_btn.config(state='disabled')
        self.opt_results_btn.config(
            state='normal' if (enabled and getattr(
                self, 'opt_multistart_results', None)) else 'disabled')
        self.opt_target.config(state=readonly)
        for row in self.opt_param_rows:
            row['combo'].config(state=readonly)
            row['min_entry'].config(state=state)
            row['max_entry'].config(state=state)
            row['log_check'].config(state=state)


    @staticmethod
    def _parameter_spec(attribute):
        if attribute.startswith('src'):
            field = attribute.split('_', 1)[1]
            return PARAMETER_SPECS[f'src_{field}']
        return PARAMETER_SPECS.get(attribute, ParameterSpec(1e-9, 1e-5, True))

    def _apply_opt_row_defaults(self, row):
        label = row['combo'].get()
        if label not in self.SWEEP_PARAMS:
            return
        attribute = self.SWEEP_PARAMS[label][0]
        spec = self._parameter_spec(attribute)
        self._set_entry_value(row['min_entry'], f'{spec.lower:.10g}')
        self._set_entry_value(row['max_entry'], f'{spec.upper:.10g}')
        row['log_var'].set(spec.log_default)

    def _build_opt_rows(self):
        """Build fit rows with parameter-specific bounds and transformations."""
        if getattr(self, '_busy', False):
            return
        old_data = [
            {'param': rd['combo'].get(), 'min': rd['min_entry'].get(),
             'max': rd['max_entry'].get(), 'log': rd['log_var'].get()}
            for rd in self.opt_param_rows
        ]
        for widget in self.opt_rows_frame.winfo_children():
            widget.destroy()
        self.opt_param_rows = []

        try:
            n = int(self.opt_n_entry.get())
        except (ValueError, tk.TclError):
            n = 1
        n = max(1, min(6, n))
        self._set_entry_value(self.opt_n_entry, n)

        param_names = [
            label for label, (attribute, _) in self.SWEEP_PARAMS.items()
            if attribute not in self.OPT_EXCLUDED_ATTRS
        ]
        preferred = ['D_ref', 'K_ref', 'EaD', 'EaK', 'c0free', 'F']
        enabled = self.opt_enabled.get()
        state = 'normal' if enabled else 'disabled'
        readonly = 'readonly' if enabled else 'disabled'

        for i in range(n):
            frame = ttk.Frame(self.opt_rows_frame)
            frame.pack(fill=tk.X, pady=1)
            combo = ttk.Combobox(frame, values=param_names, width=28,
                                 state=readonly)
            restored = i < len(old_data) and old_data[i]['param'] in param_names
            if restored:
                combo.set(old_data[i]['param'])
            else:
                wanted = preferred[min(i, len(preferred) - 1)]
                default_label = next(
                    (label for label in param_names
                     if self.SWEEP_PARAMS[label][0] == wanted), param_names[0])
                combo.set(default_label)
            combo.pack(side=tk.LEFT, padx=1)

            min_entry = ttk.Entry(frame, width=10)
            max_entry = ttk.Entry(frame, width=10)
            min_entry.pack(side=tk.LEFT, padx=1)
            max_entry.pack(side=tk.LEFT, padx=1)
            log_var = tk.BooleanVar(value=False)
            log_check = ttk.Checkbutton(frame, variable=log_var)
            log_check.pack(side=tk.LEFT, padx=2)
            row = {'combo': combo, 'min_entry': min_entry,
                   'max_entry': max_entry, 'log_var': log_var,
                   'log_check': log_check}
            self.opt_param_rows.append(row)

            if restored:
                self._set_entry_value(min_entry, old_data[i]['min'])
                self._set_entry_value(max_entry, old_data[i]['max'])
                log_var.set(bool(old_data[i]['log']))
            else:
                self._apply_opt_row_defaults(row)
            combo.bind('<<ComboboxSelected>>',
                       lambda _event, rd=row: self._apply_opt_row_defaults(rd))
            combo.config(state=readonly)
            min_entry.config(state=state)
            max_entry.config(state=state)
            log_check.config(state=state)


    @staticmethod
    def _apply_opt_params(x_vals, params, fit_specs):
        """Apply immutable fit-parameter snapshots to a parameter dictionary."""
        updated = dict(params)
        updated['src_params'] = np.asarray(params['src_params'], dtype=float).copy()
        for value, spec in zip(x_vals, fit_specs):
            attribute = spec.attribute
            if attribute == 'R':
                updated['R'] = float(value) * 1e-4
                updated['R_um'] = float(value)
            elif attribute.startswith('src'):
                parts = attribute.split('_')
                src_idx = int(parts[0][3:]) - 1
                offset = {'c0': 0, 'A': 1, 'Ea': 2}[parts[1]]
                arr_idx = src_idx * 3 + offset
                if arr_idx >= len(updated['src_params']):
                    raise ValueError(
                        f'Source {src_idx + 1} does not exist in the captured model state.')
                updated['src_params'][arr_idx] = float(value)
            else:
                updated[attribute] = float(value)
        return updated

    def _terminate_optimization(self):
        """Signal the running optimizer to stop after the current evaluation."""
        self._opt_stop = True
        self.opt_stop_btn.config(state='disabled')
        self.status_label.config(text="Termination requested — finishing current evaluation...",
                                 foreground="orange")


    def run_optimization(self):
        """Fit immutable model/data snapshots with balanced residual blocks."""
        if getattr(self, '_busy', False) or not self.opt_enabled.get():
            return
        try:
            experimental = self._snapshot_experimental(report=True)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Experimental Data Error', str(exc))
            return
        target = {'Concentration (ppbv)': 'Concentration',
                  'Mass (ng)': 'Mass'}.get(self.opt_target.get(), self.opt_target.get())
        conc_series = experimental.concentration
        mass_series = experimental.mass
        has_conc = conc_series.raw_time is not None
        has_mass = mass_series.raw_time is not None

        if target == 'Concentration' and not has_conc:
            messagebox.showerror('No Experimental Data',
                                 'Enter concentration observations before fitting.')
            return
        if target == 'Mass' and not has_mass:
            messagebox.showerror('No Experimental Data',
                                 'Enter mass observations before fitting.')
            return
        if target == 'Both' and not (has_conc or has_mass):
            messagebox.showerror('No Experimental Data',
                                 'Enter concentration and/or mass observations before fitting.')
            return
        if target == 'Both' and not has_conc:
            target = 'Mass'
        if target == 'Both' and not has_mass:
            target = 'Concentration'

        try:
            base = self._collect_params()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Parameter Error', str(exc))
            return

        def clip_series(series):
            if series.raw_time is None:
                return None, None, None, 0
            keep = ((series.raw_time >= 0.0)
                    & (series.raw_time <= base['tfinal']))
            sigma = (series.raw_sigma[keep] if series.raw_sigma is not None else None)
            return (series.raw_time[keep], series.raw_values[keep], sigma,
                    int(np.count_nonzero(~keep)))

        conc_t, conc_y, conc_sigma, dropped_c = clip_series(conc_series)
        mass_t, mass_y, mass_sigma, dropped_m = clip_series(mass_series)
        has_conc = conc_t is not None and conc_t.size > 0
        has_mass = mass_t is not None and mass_t.size > 0
        n_dropped = dropped_c + dropped_m
        if n_dropped:
            messagebox.showwarning(
                'Data outside simulated window',
                f'{n_dropped} observation(s) outside 0 to {base["tfinal"]:g} min '
                'were excluded from the fit.')
        if (target == 'Concentration' and not has_conc) or (
                target == 'Mass' and not has_mass) or not (has_conc or has_mass):
            messagebox.showerror('No Usable Data',
                                 'No target observations fall inside the simulated time window.')
            return

        selected_labels = [row['combo'].get() for row in self.opt_param_rows]
        if not selected_labels or len(selected_labels) != len(set(selected_labels)):
            messagebox.showerror('Optimization Error',
                                 'Each fit row must select a different parameter.')
            return
        fit_specs = []
        n_sources = len(base['src_params']) // 3
        try:
            for row, label in zip(self.opt_param_rows, selected_labels):
                if label not in self.SWEEP_PARAMS:
                    raise ValueError('Select a valid parameter in every fit row.')
                attribute = self.SWEEP_PARAMS[label][0]
                if attribute.startswith('src'):
                    src_idx = int(attribute.split('_')[0][3:]) - 1
                    if src_idx >= n_sources:
                        raise ValueError(
                            f'{label} was selected, but source {src_idx + 1} does not exist.')
                lo = float(row['min_entry'].get())
                hi = float(row['max_entry'].get())
                use_log = bool(row['log_var'].get())
                if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                    raise ValueError('Every fit bound must be finite with min < max.')
                if use_log and lo <= 0:
                    raise ValueError('Log-scaled fit bounds must be positive.')
                fit_specs.append(FitParameterSnapshot(
                    label=label, attribute=attribute,
                    lower=lo, upper=hi, use_log=use_log))
            max_evals = int(self.opt_maxiter.get())
            n_starts = int(self.opt_nstarts.get())
            conc_weight = float(self.opt_conc_weight.get())
            mass_weight = float(self.opt_mass_weight.get())
            if not 1 <= max_evals <= MAX_OPT_EVALS:
                raise ValueError(f'Max evaluations must be 1 to {MAX_OPT_EVALS}.')
            if not 1 <= n_starts <= MAX_OPT_STARTS:
                raise ValueError(f'Multi-start count must be 1 to {MAX_OPT_STARTS}.')
            if n_starts * max_evals > MAX_OPT_TOTAL_EVALS:
                raise ValueError(
                    f'Multi-starts × max evaluations cannot exceed '
                    f'{MAX_OPT_TOTAL_EVALS:,} simulations.')
            if not (np.isfinite(conc_weight) and conc_weight > 0
                    and np.isfinite(mass_weight) and mass_weight > 0):
                raise ValueError('Target block weights must be finite and > 0.')
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Optimization Settings Error', str(exc))
            return

        blocks = []
        def add_block(kind, times, values, sigmas, weight):
            if times is None or times.size == 0:
                return
            fallback = float(np.std(values))
            if not np.isfinite(fallback) or fallback <= 0:
                fallback = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0)
            scale = np.full(values.shape, fallback, dtype=float)
            if sigmas is not None:
                valid = np.isfinite(sigmas) & (sigmas > 0)
                scale[valid] = sigmas[valid]
            # sqrt(weight/n) makes each target block's aggregate influence
            # independent of its sampling density while retaining optional
            # pointwise measurement uncertainties.
            factor = np.sqrt(weight / max(times.size, 1))
            blocks.append((kind, times, values, scale, factor))

        if target in ('Concentration', 'Both') and has_conc:
            add_block('conc', conc_t, conc_y, conc_sigma, conc_weight)
        if target in ('Mass', 'Both') and has_mass:
            add_block('mass', mass_t, mass_y, mass_sigma, mass_weight)
        n_resid = int(sum(block[1].size for block in blocks))
        if n_resid == 0:
            messagebox.showerror('No Usable Data', 'The residual vector is empty.')
            return

        bounds_search = [
            (np.log10(spec.lower), np.log10(spec.upper))
            if spec.use_log else (spec.lower, spec.upper)
            for spec in fit_specs
        ]
        log_flags = [spec.use_log for spec in fit_specs]
        conc_unit = conc_series.unit
        mass_unit = mass_series.unit

        def apply_trial(x_natural):
            return self._apply_opt_params(x_natural, base, fit_specs)

        self._active_job_id += 1
        job_id = self._active_job_id
        self._opt_stop = False
        self._set_busy(True)
        self.opt_stop_btn.config(state='normal')
        self._opt_iter = 0
        self._opt_best_sse = np.inf
        self._opt_best_x = None
        self._opt_cur_start = 1
        self._opt_n_starts = n_starts
        self._opt_start_base = 0
        self._opt_start_best_sse = np.inf
        self._opt_start_best_x = None
        self._opt_eval_cap = max_evals
        self._opt_status = ('Starting optimization...', 'orange')
        self._opt_outcome = None
        self._opt_error = None
        self._opt_done = False
        self._opt_ctx = dict(
            base=copy.deepcopy(base), fit_specs=tuple(fit_specs),
            experimental=experimental, conc_unit=conc_unit,
            mass_unit=mass_unit, max_evals=max_evals,
            started_at=_utc_now_iso())
        self.opt_multistart_results = None
        self.opt_multistart_attrs = [spec.attribute for spec in fit_specs]
        self._opt_fail_counts = {}

        class StopOptimization(Exception):
            pass

        fail_residual = np.full(n_resid, np.sqrt(1e30 / n_resid))

        def note_failure(kind):
            self._opt_fail_counts[kind] = self._opt_fail_counts.get(kind, 0) + 1

        def to_natural(x_search):
            return np.array([10.0 ** value if use_log else value
                             for value, use_log in zip(x_search, log_flags)], dtype=float)

        def residual_vector(x_search):
            if self._opt_stop or self._opt_iter >= self._opt_eval_cap:
                raise StopOptimization()
            x_natural = to_natural(x_search)
            try:
                result = self._run_one(apply_trial(x_natural))
            except (RuntimeError, ValueError, ArithmeticError, KeyError) as exc:
                self._opt_iter += 1
                note_failure(type(exc).__name__)
                return fail_residual
            mb = result.get('mass_bal_error', np.nan)
            if not np.isfinite(mb) or abs(mb) > self.MB_BAD_PCT:
                self._opt_iter += 1
                note_failure('conservation residual')
                return fail_residual

            pieces = []
            for kind, times, observed, scale, factor in blocks:
                if kind == 'conc':
                    predicted_curve = self._conc_from_uM(
                        result['c_gas_uM'], np.asarray(result['T_C']) + 273.15,
                        conc_unit, result['MW'])
                else:
                    predicted_curve = self._mass_from_ng(
                        result['delta_mass'], mass_unit)
                predicted = np.interp(times, result['t'], predicted_curve)
                pieces.append((predicted - observed) / scale * factor)
            residual = np.concatenate(pieces)
            self._opt_iter += 1
            sse = float(residual @ residual)
            if sse < self._opt_best_sse:
                self._opt_best_sse = sse
                self._opt_best_x = np.asarray(x_search, dtype=float).copy()
            if sse < self._opt_start_best_sse:
                self._opt_start_best_sse = sse
                self._opt_start_best_x = np.asarray(x_search, dtype=float).copy()
            this_start = self._opt_iter - self._opt_start_base
            self._opt_status = (
                f'S{self._opt_cur_start}/{n_starts} simulation '
                f'{this_start}/{max_evals} (total {self._opt_iter}): '
                f'SSE={sse:.4g}, best={self._opt_best_sse:.4g}', 'orange')
            return residual

        def uncertainty_diagnostics(ls_result, n_obs, n_params):
            info = dict(stderr=None, rank=0, condition=np.inf, dof=n_obs - n_params,
                        active_mask=None, full_rank=False)
            if ls_result is None or getattr(ls_result, 'jac', None) is None:
                return info
            J = np.asarray(ls_result.jac, dtype=float)
            active = np.asarray(getattr(ls_result, 'active_mask', np.zeros(n_params)), dtype=int)
            info['active_mask'] = active
            if J.ndim != 2 or J.shape[1] != n_params or not np.all(np.isfinite(J)):
                return info
            try:
                _, singular, vt = np.linalg.svd(J, full_matrices=False)
            except np.linalg.LinAlgError:
                return info
            if singular.size == 0:
                return info
            tolerance = max(J.shape) * np.finfo(float).eps * singular[0]
            rank = int(np.count_nonzero(singular > tolerance))
            condition = (float(singular[0] / singular[-1])
                         if singular[-1] > tolerance else np.inf)
            well_conditioned = bool(np.isfinite(condition)
                                    and condition <= MAX_JACOBIAN_CONDITION)
            info.update(rank=rank, condition=condition,
                        full_rank=(rank == n_params),
                        well_conditioned=well_conditioned)
            if info['dof'] <= 0 or rank < n_params or not well_conditioned:
                return info
            inv_s2 = np.zeros_like(singular)
            inv_s2[singular > tolerance] = 1.0 / singular[singular > tolerance] ** 2
            covariance = ((vt.T * inv_s2) @ vt
                          * (2.0 * float(ls_result.cost) / info['dof']))
            diagonal = np.diag(covariance)
            if np.any(~np.isfinite(diagonal)) or np.any(diagonal < 0):
                return info
            stderr = np.sqrt(diagonal)
            stderr[active != 0] = np.nan
            info['stderr'] = stderr
            return info

        n_params = len(fit_specs)
        x0_warm = []
        for spec, (lo, hi) in zip(fit_specs, bounds_search):
            attribute = spec.attribute
            if attribute == 'R':
                value = base.get('R_um', np.nan)
            elif attribute.startswith('src'):
                parts = attribute.split('_')
                src_idx = int(parts[0][3:]) - 1
                offset = {'c0': 0, 'A': 1, 'Ea': 2}[parts[1]]
                arr_idx = src_idx * 3 + offset
                value = (base['src_params'][arr_idx]
                         if arr_idx < len(base['src_params']) else np.nan)
            else:
                value = base.get(attribute, np.nan)
            if spec.use_log:
                value = np.log10(value) if np.isfinite(value) and value > 0 else np.nan
            x0_warm.append(float(np.clip(value, lo, hi))
                           if np.isfinite(value) else 0.5 * (lo + hi))
        start_points = [np.asarray(x0_warm, dtype=float)]
        if n_starts > 1:
            unit = qmc.LatinHypercube(d=n_params, seed=42).random(n_starts - 1)
            lower = np.array([bound[0] for bound in bounds_search])
            upper = np.array([bound[1] for bound in bounds_search])
            start_points.extend(lower + row * (upper - lower) for row in unit)

        def r_squared(sim_t, sim_y, times, observed):
            if times is None or times.size < 2:
                return None
            predicted = np.interp(times, sim_t, sim_y)
            ss_total = float(np.sum((observed - np.mean(observed)) ** 2))
            return (None if ss_total <= 0 else
                    1.0 - float(np.sum((predicted - observed) ** 2)) / ss_total)

        def worker():
            records = []
            try:
                for start_index, x0 in enumerate(start_points, start=1):
                    if self._opt_stop:
                        break
                    self._opt_cur_start = start_index
                    self._opt_start_base = self._opt_iter
                    self._opt_start_best_sse = np.inf
                    self._opt_start_best_x = None
                    self._opt_eval_cap = self._opt_start_base + max_evals
                    status = 'converged'
                    ls_result = None
                    try:
                        lower = np.array([bound[0] for bound in bounds_search])
                        upper = np.array([bound[1] for bound in bounds_search])
                        ls_result = least_squares(
                            residual_vector, np.clip(x0, lower, upper),
                            bounds=(lower, upper), method='trf', jac='2-point',
                            x_scale='jac', loss='linear', xtol=1e-10,
                            ftol=1e-10, gtol=1e-10,
                            max_nfev=max(max_evals, 2 * n_params + 2))
                        if ls_result.status <= 0:
                            status = 'simulation budget reached'
                    except StopOptimization:
                        status = ('terminated' if self._opt_stop
                                  else 'simulation budget reached')

                    uncertainty = uncertainty_diagnostics(
                        ls_result, n_resid, n_params)
                    if uncertainty['rank'] < n_params:
                        status += '; rank-deficient'
                    elif not uncertainty.get('well_conditioned', False):
                        status += '; ill-conditioned'
                    if (uncertainty['active_mask'] is not None
                            and np.any(uncertainty['active_mask'] != 0)):
                        status += '; bound-active'
                    record = dict(
                        start=start_index, x0_nat=to_natural(x0),
                        evals=self._opt_iter - self._opt_start_base,
                        status=status, uncertainty=uncertainty,
                        log_flags=list(log_flags))
                    if self._opt_start_best_x is None:
                        record.update(x_nat=None, sse=None, r2_conc=None,
                                      r2_mass=None, res=None)
                    else:
                        x_natural = to_natural(self._opt_start_best_x)
                        record.update(x_nat=x_natural,
                                      sse=float(self._opt_start_best_sse))
                        try:
                            best_result = self._run_one(apply_trial(x_natural))
                        except (RuntimeError, ValueError, ArithmeticError, KeyError):
                            best_result = None
                        record['res'] = best_result
                        record['r2_conc'] = (
                            r_squared(
                                best_result['t'], self._conc_from_uM(
                                    best_result['c_gas_uM'],
                                    np.asarray(best_result['T_C']) + 273.15,
                                    conc_unit, best_result['MW']), conc_t, conc_y)
                            if best_result is not None and has_conc else None)
                        record['r2_mass'] = (
                            r_squared(best_result['t'], self._mass_from_ng(
                                best_result['delta_mass'], mass_unit), mass_t, mass_y)
                            if best_result is not None and has_mass else None)
                    records.append(record)
                self._opt_outcome = dict(
                    records=records, terminated_early=bool(self._opt_stop),
                    n_starts=n_starts, failures=dict(self._opt_fail_counts))
            except Exception as exc:
                self._opt_error = exc
                self._log_exception('optimization worker', exc)
                self._opt_outcome = dict(
                    records=records, terminated_early=True,
                    n_starts=n_starts, failures=dict(self._opt_fail_counts))
            finally:
                self._opt_done = True

        self._opt_thread = threading.Thread(target=worker, daemon=True)
        self._opt_thread.start()
        self.status_label.config(text='Starting optimization...', foreground='orange')
        self.root.after(150, lambda: self._opt_poll(job_id))


    def _opt_poll(self, job_id=None):
        """Poll only the currently active optimization job."""
        if job_id is not None and job_id != self._active_job_id:
            return
        try:
            text, color = self._opt_status
            self.status_label.config(text=text, foreground=color)
        except (AttributeError, tk.TclError):
            pass
        if not self._opt_done:
            self.root.after(150, lambda: self._opt_poll(job_id))
        else:
            self._opt_finish(job_id)


    def _opt_finish(self, job_id=None):
        """Apply a completed fit only when it belongs to the active job."""
        if job_id is not None and job_id != self._active_job_id:
            return
        self._set_busy(False)
        outcome = self._opt_outcome or {}
        records = list(outcome.get('records', []))
        if self._opt_error is not None and not records:
            self.status_label.config(
                text=f'Optimization error: {self._opt_error}', foreground='red')
            messagebox.showerror('Optimization Error', str(self._opt_error))
            return
        valid = [record for record in records if record.get('sse') is not None]
        if not valid or self._opt_best_x is None:
            self.status_label.config(
                text='Optimization ended before any valid simulation completed.',
                foreground='blue' if self._opt_stop else 'red')
            return

        context = self._opt_ctx
        base = context['base']
        fit_specs = context['fit_specs']
        best = min(valid, key=lambda record: record['sse'])
        x_best = np.asarray(best['x_nat'], dtype=float)
        result_best = best.get('res')
        best_params = self._apply_opt_params(x_best, base, fit_specs)
        if result_best is None:
            try:
                result_best = self._run_one(best_params)
            except (RuntimeError, ValueError, ArithmeticError, KeyError) as exc:
                self.status_label.config(text=f'Best-fit re-run failed: {exc}', foreground='red')
                messagebox.showerror('Optimization Error', str(exc))
                return

        # Update canonical GUI state before recalculating readouts and plots.
        self._write_opt_results_to_gui(x_best, fit_specs)
        self._update_derived()
        # Plot the best result first.  It becomes the reference trajectory
        # for temperature-dependent experimental-unit conversion, while the
        # remaining starts are retained for visual comparison.
        labeled = []
        ordered_valid = [best] + [record for record in valid if record is not best]
        for record in ordered_valid:
            if record.get('res') is None:
                continue
            tag = ' (best)' if record is best else ''
            labeled.append((f"S{record['start']}: SSE={record['sse']:.3g}{tag}",
                            record['res']))
        if not labeled:
            labeled = [('Best Fit', result_best)]
        self.current_run_record = self._make_run_record(
            'optimization', 'Best Fit', best_params, result_best,
            context['experimental'], context['started_at'])
        self.sweep_run_records = []
        self.current_results = self.current_run_record.results
        self.all_sweep_results = []
        self._last_result_mode = 'optimization'
        self._last_experimental_snapshot = context['experimental']
        # Preserve the multi-start comparison on screen, but exports use the
        # single immutable best-fit record above.
        self.update_plots(
            labeled, experimental=context['experimental'],
            result_mode='optimization')

        self.opt_multistart_results = records
        if self.opt_enabled.get():
            self.opt_results_btn.config(state='normal')
        failures = outcome.get('failures') or {}
        failure_text = (' | rejected: ' + ', '.join(
            f'{count}x {kind}' for kind, count in sorted(failures.items()))
            if failures else '')
        terminated = bool(outcome.get('terminated_early'))
        self.status_label.config(
            text=(f"{'Terminated' if terminated else 'Complete'} — best "
                  f"balanced SSE={best['sse']:.4g} (start {best['start']}) | "
                  f"{len(valid)}/{outcome.get('n_starts', len(records))} starts | "
                  f"simulations={self._opt_iter}{failure_text}"),
            foreground='blue' if terminated else 'green')
        if self._opt_error is not None:
            messagebox.showerror('Optimization Error',
                                 f'Stopped early due to an error:\n{self._opt_error}')
        self._show_multistart_window()


    def _show_multistart_window(self):
        """Show fit quality, local identifiability, and bound diagnostics."""
        records = getattr(self, 'opt_multistart_results', None)
        if not records:
            messagebox.showinfo('Fit Results', 'No optimization results are available.')
            return
        if getattr(self, '_ms_win', None) is not None:
            try:
                if self._ms_win.winfo_exists():
                    self._ms_win.destroy()
            except tk.TclError:
                pass

        attrs = getattr(self, 'opt_multistart_attrs', [])
        show_r2c = any(record.get('r2_conc') is not None for record in records)
        show_r2m = any(record.get('r2_mass') is not None for record in records)
        columns = ['Start', 'Status', 'Sims', 'SSE', 'Rank', 'Condition']
        if show_r2c:
            columns.append('R² (conc)')
        if show_r2m:
            columns.append('R² (mass)')
        for attribute in attrs:
            columns.extend((f'{attribute} (init)',
                            f'{attribute} (fit ± local SE)'))

        def fmt(value, r2=False):
            if value is None or not np.isfinite(value):
                return '—'
            return f'{value:.5f}' if r2 else f'{value:.6g}'

        def fit_text(record, index):
            if record.get('x_nat') is None:
                return '—'
            x_value = float(record['x_nat'][index])
            uncertainty = record.get('uncertainty') or {}
            active = uncertainty.get('active_mask')
            if active is not None and index < len(active) and active[index] != 0:
                return f'{x_value:.6g} [bound]'
            stderr = uncertainty.get('stderr')
            if stderr is None or index >= len(stderr) or not np.isfinite(stderr[index]):
                return f'{x_value:.6g} [SE unavailable]'
            use_log = (record.get('log_flags') or [False] * len(attrs))[index]
            sigma = (abs(x_value) * np.log(10.0) * stderr[index]
                     if use_log else stderr[index])
            return f'{x_value:.6g} ± {sigma:.2g}'

        def values(record):
            uncertainty = record.get('uncertainty') or {}
            row = [record['start'], record['status'], record['evals'],
                   fmt(record.get('sse')), uncertainty.get('rank', 0),
                   fmt(uncertainty.get('condition'))]
            if show_r2c:
                row.append(fmt(record.get('r2_conc'), r2=True))
            if show_r2m:
                row.append(fmt(record.get('r2_mass'), r2=True))
            for index in range(len(attrs)):
                x0 = (record['x0_nat'][index]
                      if record.get('x0_nat') is not None else None)
                row.extend((fmt(x0), fit_text(record, index)))
            return row

        win = tk.Toplevel(self.root)
        win.title('Multi-Start Fit Results')
        win.geometry('1200x390')
        self._ms_win = win
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=13)
        for column in columns:
            tree.heading(column, text=column)
            width = 60 if column in ('Start', 'Sims', 'Rank') else 125
            if '(fit' in column:
                width = 180
            elif column == 'Status':
                width = 230
            tree.column(column, width=width,
                        anchor=tk.W if column == 'Status' else tk.E,
                        stretch=False)
        vbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        hbar = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        valid = [record for record in records if record.get('sse') is not None]
        best = min(valid, key=lambda record: record['sse']) if valid else None
        ordered = sorted(records, key=lambda record: (
            record.get('sse') is None,
            record.get('sse') if record.get('sse') is not None else 0.0))
        tree.tag_configure('best', background='#d8f5d8')
        for record in ordered:
            tree.insert('', tk.END, values=values(record),
                        tags=('best',) if record is best else ())

        footer = ttk.Frame(win)
        footer.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(
            footer,
            text=('Uncertainties are local linearized standard errors. They are '
                  'suppressed when rank-deficient, ill-conditioned, bound-active, '
                  'or supported by insufficient degrees of freedom.'),
            foreground='gray').pack(side=tk.LEFT)

        def copy_table():
            lines = ['\t'.join(columns)]
            lines.extend('\t'.join(str(value) for value in values(record))
                         for record in ordered)
            self.root.clipboard_clear()
            self.root.clipboard_append('\n'.join(lines))
            self.status_label.config(text='Fit results copied to clipboard.',
                                     foreground='green')

        ttk.Button(footer, text='Copy Table (TSV)',
                   command=copy_table).pack(side=tk.RIGHT, padx=4)
        ttk.Button(footer, text='Close', command=win.destroy).pack(side=tk.RIGHT)


    def _write_opt_results_to_gui(self, x_natural, fit_specs):
        """Write fit values using the immutable parameter identities captured at launch."""
        widget_map = {
            'T0': self.T0, 'Tfinal': self.Tfinal, 'RR': self.RR,
            'Q': self.Q, 'feed_tank_volume': self.feed_tank_volume,
            'mSample': self.mSample, 'rhoSample': self.rhoSample,
            'Vvessel': self.Vvessel, 'K_ref': self.K_ref,
            'EaK': self.EaK, 'D_ref': self.D_ref, 'EaD': self.EaD,
            'cD': self.cD, 'F': self.F, 'c0free': self.c0free,
            'base_conc': self.base_conc,
        }
        for value, spec in zip(x_natural, fit_specs):
            attribute = spec.attribute
            text_value = f'{float(value):.8g}'
            if attribute == 'R':
                self._set_entry_value(self.R, text_value)
            elif attribute.startswith('src'):
                parts = attribute.split('_')
                src_idx = int(parts[0][3:]) - 1
                widget_index = {'c0': 1, 'A': 3, 'Ea': 5}[parts[1]]
                if src_idx < len(self.source_entries):
                    self._set_entry_value(
                        self.source_entries[src_idx][widget_index], text_value)
            elif attribute in widget_map:
                self._set_entry_value(widget_map[attribute], text_value)
        self._update_derived()

    def _fmt_sweep_val(self, attr_name, val, log_space):
        """Format labels without discarding the exact numeric sweep value."""
        if attr_name in self.SWEEP_INT_ATTRS:
            return str(int(round(float(val))))
        return f'{float(val):.8e}' if log_space else f'{float(val):.10g}'

    MB_WARN_PCT = 1e-3
    MB_BAD_PCT = 1e-1

    def _mb_advisory(self, results):
        """Post-run conservation-residual quality note.

        The residual is the difference between what left through the purge and
        what the inventory says should have left.  With the conservative
        (Kirchhoff-flux) scheme it reflects ODE-solver accuracy, so an elevated
        value points at the tolerances rather than the grid or cD.
        Returns (status_suffix, status_color_or_None).
        """
        notes = []
        color = None
        mb = abs(results.get('mass_bal_error', 0.0))
        if not np.isfinite(mb) or mb > self.MB_BAD_PCT:
            notes.append(f"conservation residual INVALID ({mb:.2g}%) — "
                         "check parameters / tighten rtol & atol")
            color = 'red'
        elif mb > self.MB_WARN_PCT:
            notes.append(f"conservation residual {mb:.2g}% high — "
                         "tighten solver tolerances (rtol/atol)")
            color = '#b36b00'
        suffix = ("  [" + "; ".join(notes) + "]") if notes else ""
        return suffix, color

    # ------------------------------------------------------------------
    # Simulation driver.  The solve runs on a worker thread and the UI is
    # refreshed from a root.after() poll, mirroring the optimizer.  Previously
    # this ran inline with root.update(), which froze the window for the whole
    # run and -- because update() reprocesses the event queue -- let a second
    # click re-enter the handler while the first was still solving.
    # ------------------------------------------------------------------


    @staticmethod
    def _walk_widgets(widget):
        for child in widget.winfo_children():
            yield child
            yield from OutgassingGUI._walk_widgets(child)

    def _set_busy(self, busy, stoppable=False):
        """Freeze every model-mutating control and restore exact prior states."""
        busy = bool(busy)
        if busy == getattr(self, '_busy', False):
            return
        if busy:
            self._busy = True
            self._busy_state_cache = []
            mutable_types = (ttk.Entry, ttk.Combobox, ttk.Button,
                             ttk.Checkbutton, tk.Entry, tk.Text, tk.Button,
                             tk.Checkbutton)
            skip = {getattr(self, 'stop_sim_btn', None),
                    getattr(self, 'opt_stop_btn', None)}
            for widget in self._walk_widgets(self.root):
                if widget in skip or not isinstance(widget, mutable_types):
                    continue
                try:
                    state = widget.cget('state')
                    self._busy_state_cache.append((widget, state))
                    widget.config(state='disabled')
                except tk.TclError:
                    continue
            if stoppable and getattr(self, 'stop_sim_btn', None) is not None:
                self.stop_sim_btn.config(state='normal')
            return

        for widget, state in self._busy_state_cache:
            try:
                if widget.winfo_exists():
                    widget.config(state=state)
            except tk.TclError:
                pass
        self._busy_state_cache = []
        self._busy = False
        try:
            self.stop_sim_btn.config(state='disabled')
            self.opt_stop_btn.config(state='disabled')
        except tk.TclError:
            pass

    def _stop_simulation(self):
        """Ask the running sweep to stop after the current point."""
        self._sim_stop = True
        try:
            self.stop_sim_btn.config(state='disabled')
        except tk.TclError:
            pass
        self.status_label.config(text="Stop requested — finishing current point...",
                                 foreground='orange')


    @staticmethod
    def _readonly_array(values):
        if values is None:
            return None
        arr = np.asarray(values, dtype=float).copy()
        arr.setflags(write=False)
        return arr

    def _snapshot_experimental(self, report=False):
        """Capture raw fit observations and independently resampled plot data."""
        policy = self.duplicate_policy.get()
        conc = self.parse_experimental_data(
            self.conc_data_text, self.conc_interp_var, self.conc_interp_n,
            self.conc_interp_method, apply_resampling=True, return_details=True,
            duplicate_policy=policy)
        mass = self.parse_experimental_data(
            self.mass_data_text, self.mass_interp_var, self.mass_interp_n,
            self.mass_interp_method, apply_resampling=True, return_details=True,
            duplicate_policy=policy)

        def series(details, unit):
            return ExperimentalSeries(
                raw_time=self._readonly_array(details['raw_time']),
                raw_values=self._readonly_array(details['raw_values']),
                raw_sigma=self._readonly_array(details['raw_sigma']),
                plot_time=self._readonly_array(details['plot_time']),
                plot_values=self._readonly_array(details['plot_values']),
                unit=unit, raw_text=details['raw_text'],
                diagnostics=tuple(details['diagnostics']),
                duplicate_policy=details.get(
                    'duplicate_policy', DUPLICATE_POLICIES[0]))

        snap = ExperimentalSnapshot(
            concentration=series(conc, self._conc_cmp_unit()),
            mass=series(mass, self._mass_cmp_unit()))
        if report:
            notes = list(snap.concentration.diagnostics) + list(snap.mass.diagnostics)
            if notes:
                detail = '\n'.join(f'  {note}' for note in notes[:12])
                if len(notes) > 12:
                    detail += f'\n  ... and {len(notes) - 12} more'
                messagebox.showwarning(
                    'Experimental Data Diagnostics',
                    'The data were loaded with the following notes:\n\n' + detail)
        return snap

    def _build_sweep_plan(self, base):
        """Validate sweep settings and retain exact values separately from labels."""
        param_label = self.sweep_param_var.get()
        if param_label not in self.SWEEP_PARAMS:
            raise ValueError('Select a valid sweep parameter.')
        attr_name, _ = self.SWEEP_PARAMS[param_label]
        try:
            start_val = float(self.sweep_start.get())
            stop_val = float(self.sweep_stop.get())
            n_requested = int(self.sweep_n.get())
        except (ValueError, tk.TclError) as exc:
            raise ValueError('Sweep endpoints and point count must be numeric.') from exc
        if not (np.isfinite(start_val) and np.isfinite(stop_val)):
            raise ValueError('Sweep endpoints must be finite.')
        if not 2 <= n_requested <= MAX_SWEEP_POINTS:
            raise ValueError(
                f'Sweep must use 2 to {MAX_SWEEP_POINTS} requested points.')
        log_space = bool(self.sweep_log.get())
        if log_space and (start_val <= 0 or stop_val <= 0):
            raise ValueError('Log-spaced sweeps require positive start and stop values.')

        if attr_name.startswith('src'):
            src_idx = int(attr_name.split('_')[0][3:]) - 1
            if src_idx >= len(base['src_params']) // 3:
                raise ValueError(
                    f'Source {src_idx + 1} does not exist. Add it before sweeping.')
        if attr_name == 'grid_stretch' and base.get('grid_scheme') == 'uniform':
            raise ValueError(
                "Grid Stretch does not affect a uniform grid. Select a non-uniform grid.")

        values = (np.logspace(np.log10(start_val), np.log10(stop_val), n_requested)
                  if log_space else np.linspace(start_val, stop_val, n_requested))
        if attr_name in self.SWEEP_INT_ATTRS:
            values = np.asarray([int(round(float(v))) for v in values], dtype=float)

        # Preserve order while removing exactly duplicate runs (integer rounding
        # is the common cause). Revalidate after deduplication.
        unique = []
        seen = set()
        for value in values:
            key = int(value) if attr_name in self.SWEEP_INT_ATTRS else float(value)
            if key not in seen:
                seen.add(key)
                unique.append(float(value))
        values = np.asarray(unique, dtype=float)
        if values.size < 2:
            raise ValueError(
                'The sweep collapses to fewer than two unique values after '
                'integer rounding/deduplication. Widen the range or use more separation.')

        cmap = plt.get_cmap(self.sweep_cmap.get())
        colors = [cmap(i / max(values.size - 1, 1)) for i in range(values.size)]
        labels = [self._fmt_sweep_val(attr_name, value, log_space)
                  for value in values]
        # Guard against any display-label collision without changing exact values.
        if len(set(labels)) != len(labels):
            labels = [str(int(v)) if attr_name in self.SWEEP_INT_ATTRS
                      else format(float(v), '.17g') for v in values]
        return dict(attr=attr_name, param_label=param_label, vals=values,
                    labels=labels, colors=colors, log=log_space,
                    n_requested=n_requested)

    @staticmethod
    def _apply_sweep_value(p, attr_name, val, int_attrs):
        """Write one swept value into a parameter dict (widget-free)."""
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
                    f"Source {src_idx + 1} does not exist in the current source "
                    "configuration. Add more sources before sweeping this parameter.")
            p['src_params'][arr_idx] = val
        elif attr_name in int_attrs:
            p[attr_name] = int(round(float(val)))
        else:
            p[attr_name] = val
        return p

    def run_simulation(self):
        if getattr(self, '_busy', False):
            return
        try:
            # Reconcile the dynamic source rows with the Number of Sources
            # entry before capturing parameters.  This makes clicking Run
            # equivalent to pressing Enter or clicking "Update Source Fields"
            # first, and prevents stale source widgets from remaining active.
            raw_source_count = self.num_sources.get().strip()
            try:
                requested_source_count = int(raw_source_count)
            except ValueError as exc:
                raise ValueError('Number of Sources must be an integer.') from exc
            if not 0 <= requested_source_count <= MAX_UI_SOURCES:
                raise ValueError(
                    f'Number of Sources must be between 0 and {MAX_UI_SOURCES}.')
            if requested_source_count != len(self.source_entries):
                self.update_source_fields()
            base = self._collect_params()
            plan = self._build_sweep_plan(base) if self.sweep_enabled.get() else None
            experimental = self._snapshot_experimental(report=True)
        except (ValueError, tk.TclError) as exc:
            self.status_label.config(text=f'Error: {exc}', foreground='red')
            messagebox.showerror('Simulation Error', str(exc))
            return

        self._active_job_id += 1
        job_id = self._active_job_id
        started_at = _utc_now_iso()
        self._sim_stop = False
        self._sim_done = False
        self._sim_error = None
        self._sim_out = None
        self._sim_status = ('Running simulation...', 'orange')
        self._set_busy(True, stoppable=plan is not None)
        self.status_label.config(text='Running simulation...', foreground='orange')
        threading.Thread(
            target=self._sim_worker,
            args=(copy.deepcopy(base), copy.deepcopy(plan), experimental,
                  job_id, started_at), daemon=True).start()
        self.root.after(120, lambda: self._sim_poll(job_id))

    def _sim_worker(self, base, plan, experimental, job_id, started_at):
        """Run from immutable inputs; this worker never reads or writes Tk widgets."""
        try:
            if plan is None:
                results = self._run_one(base)
                record = self._make_run_record(
                    'single', 'Model', base, results, experimental, started_at)
                outcome = dict(kind='single', record=record)
            else:
                records, failures = [], []
                wall_start = time.time()
                n_points = len(plan['vals'])
                for index, value in enumerate(plan['vals']):
                    if self._sim_stop:
                        break
                    label = plan['labels'][index]
                    self._sim_status = (
                        f"Sweep {index + 1}/{n_points}: {plan['attr']} = {label}",
                        'orange')
                    params = dict(base)
                    params['src_params'] = np.asarray(base['src_params']).copy()
                    try:
                        params = self._apply_sweep_value(
                            params, plan['attr'], value, self.SWEEP_INT_ATTRS)
                        results = self._run_one(params)
                    except (RuntimeError, ValueError, ArithmeticError, KeyError) as exc:
                        failures.append((float(value), label, str(exc)))
                        self._sim_status = (
                            f'Sweep {index + 1}/{n_points}: {label} FAILED '
                            f'({str(exc)[:60]})', '#b36b00')
                        continue
                    results['_sweep_color'] = tuple(plan['colors'][index])
                    record = self._make_run_record(
                        'sweep', label, params, results, experimental, started_at,
                        sweep_parameter=plan['param_label'],
                        sweep_attribute=plan['attr'], sweep_value=float(value))
                    records.append(record)
                    self._sim_status = (
                        f"Sweep {index + 1}/{n_points}: {plan['attr']} = {label} done | "
                        f"residual {results['mass_bal_error']:.1e}%", 'orange')
                outcome = dict(
                    kind='sweep', records=records, plan=plan, failures=failures,
                    elapsed=time.time() - wall_start, stopped=bool(self._sim_stop))
            if job_id == self._active_job_id:
                self._sim_out = outcome
        except Exception as exc:
            if job_id == self._active_job_id:
                self._sim_error = exc
                self._log_exception('simulation worker', exc)
        finally:
            if job_id == self._active_job_id:
                self._sim_done = True

    def _sim_poll(self, job_id=None):
        if job_id is not None and job_id != self._active_job_id:
            return
        try:
            status_text, color = self._sim_status
            self.status_label.config(text=status_text, foreground=color)
        except (AttributeError, tk.TclError):
            pass
        if not self._sim_done:
            self.root.after(120, lambda: self._sim_poll(job_id))
        else:
            self._sim_finish(job_id)

    def _sim_finish(self, job_id=None):
        """Apply only the outcome belonging to the active simulation job."""
        if job_id is not None and job_id != self._active_job_id:
            return
        self._set_busy(False)
        if self._sim_error is not None:
            self.status_label.config(text=f'Error: {self._sim_error}', foreground='red')
            messagebox.showerror('Simulation Error', str(self._sim_error))
            return

        outcome = self._sim_out or {}
        if outcome.get('kind') == 'single':
            record = outcome['record']
            results = record.results
            experimental = record.experimental
            self.current_run_record = record
            self.sweep_run_records = []
            self.current_results = results
            self.all_sweep_results = []
            self._last_result_mode = 'single'
            self._last_experimental_snapshot = experimental
            self.update_plots(
                [('Model', results)], experimental=experimental,
                result_mode='single')

            conc = experimental.concentration
            mass = experimental.mass
            r2_conc = (self._compute_r_squared(
                results['t'], self._model_conc_in(results, conc.unit),
                conc.raw_time, conc.raw_values)
                if conc.raw_time is not None else None)
            r2_mass = (self._compute_r_squared(
                results['t'], self._model_mass_in(results, mass.unit),
                mass.raw_time, mass.raw_values)
                if mass.raw_time is not None else None)
            r2_parts = []
            if r2_conc is not None:
                r2_parts.append(f'R²(conc)={r2_conc:.6f}')
            if r2_mass is not None:
                r2_parts.append(f'R²(mass)={r2_mass:.6f}')
            r2_text = (' | ' + ' | '.join(r2_parts)) if r2_parts else ''
            advisory, advisory_color = self._mb_advisory(results)
            self.status_label.config(
                text=(f"Complete! Solve time: {results['solve_time']:.2f}s "
                      f"({results.get('n_segments', 1)} segments) | "
                      f"Conservation residual: {results['mass_bal_error']:.2e}%"
                      f'{r2_text}{advisory}'),
                foreground=advisory_color or 'green')
            return

        records = list(outcome.get('records', []))
        plan = outcome.get('plan', {})
        failures = list(outcome.get('failures', []))
        if not records:
            message = 'Sweep produced no successful runs.'
            if failures:
                message += f' First failure ({failures[0][1]}): {failures[0][2]}'
            self.status_label.config(text=message, foreground='red')
            messagebox.showerror('Simulation Error', message)
            return

        labeled = self._labeled_from_records(records)
        self.sweep_run_records = records
        self.current_run_record = records[-1]
        self.current_results = records[-1].results
        self.all_sweep_results = labeled
        self._last_result_mode = 'sweep'
        self._last_experimental_snapshot = records[0].experimental
        self.update_plots(
            labeled, sweep_label=plan.get('param_label'),
            experimental=records[0].experimental, result_mode='sweep')

        residuals = np.asarray([
            abs(float(record.results.get('mass_bal_error', np.nan)))
            for record in records])
        finite = np.isfinite(residuals)
        worst_index = (int(np.argmax(np.where(finite, residuals, -np.inf)))
                       if np.any(finite) else 0)
        worst = residuals[worst_index]
        residual_text = (
            f" | max |residual|: {worst:.2e}% "
            f"({plan.get('attr')} = {records[worst_index].label})")
        if not np.isfinite(worst) or worst > self.MB_BAD_PCT:
            residual_text += ' — INVALID; check parameters / tighten tolerances'
            status_color = 'red'
        elif worst > self.MB_WARN_PCT:
            residual_text += ' — tighten solver tolerances'
            status_color = '#b36b00'
        else:
            status_color = 'green'

        requested = len(plan.get('vals', records))
        heading = 'Sweep stopped' if outcome.get('stopped') else 'Sweep complete!'
        failed_text = f' | {len(failures)} failed' if failures else ''
        self.status_label.config(
            text=(f'{heading} {len(records)}/{requested} runs in '
                  f"{outcome.get('elapsed', 0.0):.1f}s{failed_text}{residual_text}"),
            foreground=('#b36b00' if failures and status_color == 'green'
                        else status_color))
        if failures:
            detail = '\n'.join(
                f'  {label} (exact {value:.17g}): {error}'
                for value, label, error in failures[:10])
            if len(failures) > 10:
                detail += f'\n  ... and {len(failures) - 10} more'
            messagebox.showwarning(
                'Sweep points failed',
                f'{len(failures)} of {requested} sweep points failed and were '
                f'omitted from the plot:\n\n{detail}')

    def _time_axis_limits(self, labeled_results, settings=None):
        """Resolve finite time limits without clipping t=0 on a linear axis."""
        if settings is None:
            log_time = bool(getattr(self, 'logtime_var', None)
                            and self.logtime_var.get())

            def parse(entry):
                try:
                    raw = entry.get().strip()
                    if not raw:
                        return None
                    value = float(raw)
                    return value if np.isfinite(value) else None
                except (ValueError, AttributeError, tk.TclError):
                    return None

            requested_min = parse(getattr(self, 'time_min', None))
            requested_max = parse(getattr(self, 'time_max', None))
        else:
            log_time = bool(settings.get('log_time', False))
            def finite_or_none(value):
                try:
                    number = float(value)
                    return number if np.isfinite(number) else None
                except (TypeError, ValueError):
                    return None
            requested_min = finite_or_none(settings.get('time_min'))
            requested_max = finite_or_none(settings.get('time_max'))
        arrays = [np.asarray(result['t'], dtype=float)
                  for _, result in labeled_results if len(result.get('t', []))]
        all_times = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
        all_times = all_times[np.isfinite(all_times)]
        data_max = float(np.max(all_times)) if all_times.size else None

        left = right = None
        if log_time:
            positive = all_times[all_times > 0]
            left = float(np.min(positive)) if positive.size else None
            if (requested_min is not None and requested_min > 0
                    and (data_max is None or requested_min < data_max)):
                left = requested_min
            if (requested_max is not None and requested_max > 0
                    and (left is None or requested_max > left)):
                right = requested_max
        else:
            left = requested_min
            right = requested_max
            if left is not None and right is not None and left >= right:
                right = None
        return log_time, left, right

    def _refresh_plots(self):
        """Re-render the stored result/data snapshot without re-running."""
        labeled = getattr(self, '_last_labeled_results', None)
        if labeled:
            self.update_plots(
                labeled, getattr(self, '_last_sweep_label', None),
                experimental=getattr(self, '_last_experimental_snapshot', None),
                result_mode=getattr(self, '_last_result_mode', None))

    def update_plots(self, labeled_results, sweep_label=None,
                     experimental=None, result_mode=None):
        """Render model results against the frozen experimental-data snapshot."""
        self._last_labeled_results = labeled_results
        self._last_sweep_label = sweep_label
        if experimental is not None:
            self._last_experimental_snapshot = experimental
        experimental = experimental or self._last_experimental_snapshot
        if result_mode is not None:
            self._last_result_mode = result_mode
        result_mode = result_mode or self._last_result_mode or 'single'

        axes = [self.ax1, self.ax2, self.ax3, self.ax4,
                self.ax5, self.ax6, self.ax7]
        for axis in axes:
            axis.clear()

        is_sweep = result_mode == 'sweep'
        is_multi = len(labeled_results) > 1
        plot_unit = self._conc_plot_unit()
        mass_plot_unit = self._mass_plot_unit()
        conc_series = (experimental.concentration if experimental else ExperimentalSeries())
        mass_series = (experimental.mass if experimental else ExperimentalSeries())

        reference_results = labeled_results[0][1] if labeled_results else None
        exp_conc_raw = (self._exp_conc_convert(
            conc_series.raw_time, conc_series.raw_values,
            conc_series.unit, plot_unit, reference_results)
            if conc_series.raw_time is not None else None)
        exp_conc_display = (self._exp_conc_convert(
            conc_series.plot_time, conc_series.plot_values,
            conc_series.unit, plot_unit, reference_results)
            if conc_series.plot_time is not None else None)
        exp_mass_raw = (self._exp_mass_convert(
            mass_series.raw_values, mass_series.unit, mass_plot_unit)
            if mass_series.raw_time is not None else None)
        exp_mass_display = (self._exp_mass_convert(
            mass_series.plot_values, mass_series.unit, mass_plot_unit)
            if mass_series.plot_time is not None else None)

        line_width = 2.0 if (is_sweep or is_multi) else 2.5
        for label, result in labeled_results:
            kwargs = {'linewidth': line_width, 'label': label}
            if result.get('_sweep_color') is not None:
                kwargs['color'] = result['_sweep_color']
            time_values = result['t']
            self.ax1.plot(time_values, result['T_C'], **kwargs)
            self.ax2.plot(time_values, self._model_feed_in(result, plot_unit), **kwargs)
            self.ax3.plot(time_values, result['Q'], **kwargs)
            self.ax4.plot(time_values, result['D'], **kwargs)
            self.ax5.plot(time_values, result['S'], **kwargs)
            self.ax6.plot(time_values,
                          self._model_mass_in(result, mass_plot_unit), **kwargs)
            self.ax7.plot(time_values,
                          self._model_conc_in(result, plot_unit), **kwargs)

        log_time, left_limit, right_limit = self._time_axis_limits(labeled_results)
        labels = [
            (self.ax1, 'Time (min)', 'Temperature (°C)', 'Temperature'),
            (self.ax2, 'Time (min)', self._feed_axis_label(plot_unit),
             'Feed Concentration'),
            (self.ax3, 'Time (min)', 'Flow Rate (ml/min)', 'Flow Rate'),
            (self.ax4, 'Time (min)', 'Diffusivity (cm²/s)',
             'Diffusivity at Surface'),
            (self.ax5, 'Time (min)', 'S (cm³@STP/atm/cm³)', 'Solubility'),
            (self.ax6, 'Time (min)', self._mass_axis_label(mass_plot_unit),
             'Sample Mass Change'),
            (self.ax7, 'Time (min)', self._conc_axis_label(plot_unit),
             'Headspace Gas Concentration'),
        ]
        for axis, xlabel, ylabel, title in labels:
            axis.set_xlabel(xlabel, fontsize=11)
            axis.set_ylabel(ylabel, fontsize=11)
            axis.set_title(title, fontsize=12)
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_color('black')
            axis.minorticks_on()
            axis.tick_params(which='major', direction='in', length=6, width=1.2,
                             top=True, bottom=True, left=True, right=True,
                             labeltop=False, labelright=False, labelsize=10)
            axis.tick_params(which='minor', direction='in', length=3, width=1.0,
                             top=True, bottom=True, left=True, right=True)
            if log_time:
                axis.set_xscale('log')
            if left_limit is not None or right_limit is not None:
                axis.set_xlim(left=left_limit, right=right_limit)

        self.ax4.set_yscale('log')
        self.ax4.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
        self.ax4.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
        diffusivity = np.concatenate([
            np.asarray(result['D'], dtype=float) for _, result in labeled_results])
        positive_d = diffusivity[np.isfinite(diffusivity) & (diffusivity > 0)]
        if positive_d.size and positive_d.max() / positive_d.min() < 10:
            midpoint = float(np.sqrt(positive_d.min() * positive_d.max()))
            self.ax4.set_ylim(midpoint / 10, midpoint * 10)

        if getattr(self, 'logconc_var', None) and self.logconc_var.get():
            self.ax7.set_yscale('log')
            positive_parts = []
            for _, result in labeled_results:
                values = np.asarray(self._model_conc_in(result, plot_unit), dtype=float)
                positive_parts.append(values[np.isfinite(values) & (values > 0)])
            for values in (exp_conc_raw, exp_conc_display):
                if values is not None:
                    values = np.asarray(values, dtype=float)
                    positive_parts.append(values[np.isfinite(values) & (values > 0)])
            positive_parts = [part for part in positive_parts if part.size]
            if positive_parts:
                positive = np.concatenate(positive_parts)
                minimum, maximum = float(positive.min()), float(positive.max())
                self.ax7.set_ylim(
                    bottom=(minimum / 1.2 if maximum > minimum else minimum / 10))

        self.ax6.axhline(0, color='black', linestyle='--', linewidth=1)

        def has_display_curve(series):
            if series.raw_time is None or series.plot_time is None:
                return False
            return (len(series.raw_time) != len(series.plot_time)
                    or not np.array_equal(series.raw_time, series.plot_time)
                    or not np.array_equal(series.raw_values, series.plot_values))

        if mass_series.raw_time is not None:
            if has_display_curve(mass_series):
                self.ax6.plot(mass_series.plot_time, exp_mass_display, '--',
                              color='black', linewidth=1.5,
                              label='Display interpolation', zorder=8)
            self.ax6.plot(mass_series.raw_time, exp_mass_raw, 'o', color='black',
                          markersize=6, label='Experimental', zorder=10)
        if conc_series.raw_time is not None:
            if has_display_curve(conc_series):
                self.ax7.plot(conc_series.plot_time, exp_conc_display, '--',
                              color='black', linewidth=1.5,
                              label='Display interpolation', zorder=8)
            self.ax7.plot(conc_series.raw_time, exp_conc_raw, 'o', color='black',
                          markersize=6, label='Experimental', zorder=10)

        if not is_sweep and not is_multi and len(labeled_results) == 1:
            _, result = labeled_results[0]
            r2_conc = (self._compute_r_squared(
                result['t'], self._model_conc_in(result, conc_series.unit),
                conc_series.raw_time, conc_series.raw_values)
                if conc_series.raw_time is not None else None)
            r2_mass = (self._compute_r_squared(
                result['t'], self._model_mass_in(result, mass_series.unit),
                mass_series.raw_time, mass_series.raw_values)
                if mass_series.raw_time is not None else None)
            box = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85)
            if r2_conc is not None:
                self.ax7.text(0.02, 0.97, f'R² = {r2_conc:.6f}',
                              transform=self.ax7.transAxes, fontsize=10,
                              verticalalignment='top', bbox=box)
            if r2_mass is not None:
                self.ax6.text(0.02, 0.97, f'R² = {r2_mass:.6f}',
                              transform=self.ax6.transAxes, fontsize=10,
                              verticalalignment='top', bbox=box)

        if is_sweep or is_multi:
            title = ((sweep_label or '').split('–')[-1].strip()
                     if is_sweep else 'Optimization starts')
            for axis in (self.ax6, self.ax7):
                legend = axis.legend(
                    title=title, fontsize=8, title_fontsize=8, loc='best',
                    framealpha=0.9,
                    ncol=1 if len(labeled_results) <= 8 else 2)
                legend.get_frame().set_linewidth(0.5)
        else:
            if mass_series.raw_time is not None:
                self.ax6.legend(fontsize=9, loc='best')
            if conc_series.raw_time is not None:
                self.ax7.legend(fontsize=9, loc='best')
        self.canvas.draw()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = OutgassingGUI(root)
    root.mainloop()
