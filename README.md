# A Method Of Lines Integrator for Emissions Research and Exploration (MOLIERE)

MOLIERE is a Python simulation tool for modeling diffusion, partitioning, outgassing, and uptake of volatile species between a solid sample and a gas-phase headspace. The model discretizes the solid phase with the Method of Lines and integrates the resulting stiff ODE system with SciPy's implicit BDF solver.

The current code uses a conservative finite-volume Kirchhoff-flux discretization. It is designed to conserve mass to solver tolerance for constant or concentration-dependent diffusivity, finite or infinite surface transfer, slab/cylindrical/spherical geometries, and uniform or boundary-refined spatial grids.

## Files

| File | Description |
| --- | --- |
| `outgassing_gui_with_sweep_15.pyw` | Tkinter GUI application. Includes the current simulation engine, parameter sweeps, experimental-data overlays, multi-start parameter fitting, a tighter-tolerance accuracy check, CSV export, and individual plot export. |
| `Outgassing_MOL_Model_v14.ipynb` | Reference Jupyter notebook for the conservative v14 model. Runs a default simulation and plots headspace concentration and sample mass change. Its solver core matches the GUI's. |
| `outgassing_defaults_example_sweep.json` | Example GUI parameter file that enables a five-point logarithmic sweep of diffusivity. |
| `kumar_fig7_updated_with F fit.json` | Example GUI parameter file for fitting a finite surface-transfer coefficient and transport parameters to embedded concentration data. |
| `Old/` | Previous GUI and notebook versions, kept for reference. |
| `requirements.txt` | Python package requirements. |
| `LICENSE.txt` | MIT license text. |
| `NOTICE.txt` | Lawrence Livermore National Laboratory / U.S. Department of Energy notice. |

## Installation

Requirements:

- Python 3.9 or later
- NumPy >= 1.21
- SciPy >= 1.7
- Matplotlib >= 3.5
- pandas >= 1.5

Install dependencies from this folder:

```bash
pip install -r requirements.txt
```

To run the notebook, also install Jupyter if it is not already available:

```bash
pip install jupyter
```

## GUI User Guide

Launch the graphical interface:

```bash
python "outgassing_gui_with_sweep_15.pyw"
```

The GUI is the main application. It combines the current simulation engine with parameter entry, plotting, parameter-file management, sweeps, experimental-data overlays, parameter fitting, and export tools.

### Basic Workflow

1. Start the GUI with the command above.
2. Enter parameters manually or click `Load Params...` to load a JSON parameter file.
3. Choose display units and optional plot scales in the `Run & Display` bar.
4. Paste experimental data if you want overlays, R-squared values, or optimization.
5. Leave `Parameter Sweep` and `Parameter Optimization` disabled for a single model run.
6. Click `Run Simulation`.
7. Click `Export Data` after a successful run to save a CSV and plot PNG files.

To try the included examples, open the GUI, click `Load Params...`, select one of the JSON files, then run the simulation or optimization.

### Run And Display Bar

The top bar is always visible and contains the primary actions and display controls.

| Control | Use |
| --- | --- |
| `Run Simulation` | Runs either one simulation or a sweep, depending on whether `Parameter Sweep` is enabled. |
| `Stop` | During a sweep, stops after the current sweep point. |
| `Save as Default` | Saves the current GUI state to the active parameter JSON file. |
| `Export Data` | Saves results after a run. Single runs export one data block; sweeps export a summary and per-run blocks. PNG plot panels are saved beside the CSV. |
| `Log time` | Re-renders plots with a logarithmic time axis. For log time, non-positive times are omitted from the displayed range. |
| `Log conc.` | Re-renders the headspace gas-concentration plot on a logarithmic y-axis. |
| `Conc unit` | Selects plotted headspace/feed concentration units: `ppbv`, `ppmv`, `ug/m3`, or `mg/m3`. |
| `Mass unit` | Selects plotted sample mass-change units: `ng`, `ug`, or `mg`. |
| `t min` / `t max` | Optional x-axis limits. Leave blank for autoscale. |
| `Save Params As...` | Saves all current inputs, sweep settings, optimization settings, display settings, and pasted experimental data to a chosen JSON file. |
| `Load Params...` | Loads all saved settings from a JSON file and makes that file the active parameter file. |

The GUI remembers the active parameter-file path in a per-user config file: `%APPDATA%\OutgassingModel\config.json` on Windows, or `$XDG_CONFIG_HOME/OutgassingModel/config.json` (default `~/.config/OutgassingModel/config.json`) elsewhere. A path remembered by older versions in `~/.outgassing_gui_config.json` is still used as a fallback. If no remembered file exists, `Save as Default` writes `outgassing_defaults.json` in that same `OutgassingModel` folder.

### Parameter Sections

The left-side controls are grouped by purpose. Most values are plain numeric fields; `inf` is accepted for `cD` and `F`.

| Section | Fields |
| --- | --- |
| `Simulation Parameters` | Final time, grid points, relative tolerance, absolute tolerance, accuracy check, grid scheme, and grid stretch. `N` must be at least 3. |
| `Temperature Profile` | Initial hold time, initial temperature, final temperature, and ramp rate. If initial and final temperatures differ, ramp rate must be positive. |
| `Flow Parameters` | Flow basis (actual flow at chamber temperature, or standard flow from a mass-flow controller in sccm), initial no-flow time, carrier-gas flow rate and, for sccm, its reference temperature. The GUI also computes vessel gas residence time from headspace volume and flow. |
| `Feed Concentration Profile` | Number of feed steps, concentration increment per step, step duration, base concentration, initial/final holds, and optional feed-tank volume. |
| `Sample Properties` | Geometry, sample radius or half-thickness, sample mass, density, vessel volume, and analyte molecular weight. |
| `Transport Properties` | Reference temperature, partition coefficient, signed sorption enthalpy, diffusivity, diffusivity activation energy, plasticization power, surface transfer coefficient, initial mobile concentration, and initial gas concentration. |
| `Source Terms` | Number of first-order source terms and each source's initial concentration, pre-exponential factor, and activation energy. |
| `Parameter Sweep` | Enables and configures one-parameter sweeps. |
| `Parameter Optimization` | Enables and configures parameter fitting against pasted experimental data. |

The GUI displays derived readouts where useful, including feed-tank residence time, feed-cycle period, vessel residence time, characteristic diffusion time, mass-transfer Biot number, purge Péclet number, phase ratio, no-flow equilibrium concentration, and concentration-to-mass-basis conversions.

### GUI Input Parameter Reference

The tables below define the editable GUI fields. Units are the GUI input units, chosen for convenience; the solver converts values internally where needed.

#### Run And Display Inputs

| GUI label | Meaning |
| --- | --- |
| `Log time` | Displays all time axes on a logarithmic scale. This only changes plotting, not the simulation. |
| `Log conc.` | Displays the headspace gas-concentration plot on a logarithmic y-axis. This only changes plotting. |
| `Conc unit` | Display and experimental-data plotting unit for gas/feed concentrations: `ppbv`, `ppmv`, `ug/m3`, or `mg/m3`. This only changes plotting; the `Initial gas concentration` entry has its own unit selector. |
| `Mass unit` | Display unit for sample mass change: `ng`, `ug`, or `mg`. |
| `t min` | Optional lower time-axis limit in minutes. Leave blank for autoscale. |
| `t max` | Optional upper time-axis limit in minutes. Leave blank for autoscale. |

#### Simulation Parameters

| GUI label | Symbol | Meaning |
| --- | --- | --- |
| `Final Time (min)` | `t_final` | End time for the simulation. |
| `Grid Points` | `N` | Number of spatial nodes in the solid sample. Must be at least 3. Larger values improve spatial resolution but increase run time. |
| `Relative Tolerance` | `rtol` | Relative error tolerance passed to SciPy's BDF ODE solver. |
| `Absolute Tolerance` | `atol` | Absolute error tolerance passed to SciPy's BDF ODE solver. |
| `Accuracy check (re-run at rtol/10)` | none | When checked (the default), every single run, sweep point, and best fit is solved a second time at 10x tighter `rtol` and `atol`, and the largest difference in headspace concentration and in sample mass change is reported as the estimated error. Roughly doubles run time; uncheck it for long sweeps. |
| `Grid Scheme` | none | Spatial grid type: `uniform`, `tanh`, or `geometric`. Non-uniform grids refine nodes toward the sample surface. |
| `Grid Stretch (beta / ratio)` | `beta_grid` | Stretch parameter for non-uniform grids. For `tanh`, this is the tanh stretching parameter; for `geometric`, it is the overall coarse-to-fine spacing ratio. Ignored by `uniform`. |

#### Temperature Profile

| GUI label | Symbol | Meaning |
| --- | --- | --- |
| `Time at Initial Temp (min)` | `dt1` | Time held at the initial temperature before the ramp starts. |
| `Initial Temp (C)` | `T0` | Initial sample/gas temperature. The model assumes the sample is spatially isothermal. |
| `Final Temp (C)` | `Tfinal` | Target temperature after the ramp. If equal to `T0`, the simulation is isothermal. |
| `Ramp Rate (C/min)` | `RR` | Heating or cooling rate magnitude. Must be positive when `T0` and `Tfinal` differ. |

#### Flow Parameters

| GUI label | Symbol | Meaning |
| --- | --- | --- |
| `Flow basis` | `flow_basis` | What the flow rate means; see [Carrier-flow basis](#carrier-flow-basis). `Actual @ chamber T` (the default, and the behavior of earlier versions) or `Standard (sccm)` for a mass-flow-controller setpoint. A short note under the selector states what the choice implies. |
| `Initial time at Q = 0 (min)` | `tEq` | Initial no-flow or equilibration period. During this time the carrier flow is zero. |
| `Flow Rate (ml/min)` or `Flow Rate (sccm)` | `Q` | Carrier-gas flow rate after the initial no-flow period. The flow through the chamber drives removal from the well-mixed headspace through the term `Q_ch(c_feed - c_gas)/V_headspace`, with the feed's ppbv converted to a molar concentration at the chamber temperature. For the actual basis `Q_ch = Q`; for the standard basis `Q_ch = Q*T/T_std`. |
| `sccm reference T, T_std (C)` | `flow_T_std` | Standard temperature of the mass-flow controller's sccm (0 C by default; makers also use 20, 21.1 or 25 C). Shown only for the standard basis. |

#### Feed Concentration Profile

| GUI label | Symbol | Meaning |
| --- | --- | --- |
| `Number of Steps` | `n_steps` | Number of upward feed-concentration steps before the profile steps back down. Use 0 for a constant feed; the hold times may then both be zero. |
| `Delta c per Step (ppbv)` | `delta` | Feed concentration increment for each step. |
| `Step Time (min)` | `step_time` | Duration of each feed-concentration step. |
| `Base Conc (ppbv)` | `base_conc` | Baseline inlet/feed gas concentration. |
| `Initial Hold Time (min)` | `hold_time_initial` | Time at `base_conc` before the stepped feed sequence begins. |
| `Final Hold Time (min)` | `hold_time_final` | Time at `base_conc` after the stepped feed sequence ends. |
| `Feed Tank Volume (ml)` | `V_feed` | Optional upstream feed-tank volume that smooths the ideal step profile with residence time `V_feed/Q`. Use 0 for ideal square/staircase feed changes. |

#### Sample Properties

| GUI label | Symbol | Meaning |
| --- | --- | --- |
| `Geometry` | `m` | Solid geometry factor in the radial diffusion equation: `0 - Slab`, `1 - Cylinder`, or `2 - Sphere`. |
| `Sample radius / half-thickness (um)` | `R` | Characteristic sample dimension. It is radius for cylinders/spheres and half-thickness for slabs. The GUI input is micrometers; the solver converts to centimeters. |
| `Total sample mass (mg)` | `m_sample` | Total mass of the sample. Used with density to compute sample volume. |
| `Density (g/ml)` | `rho_sample` | Sample density. Used to compute sample volume and headspace volume. |
| `Vessel Volume (ml)` | `V_vessel` | Total vessel/chamber volume. The model uses `V_headspace = V_vessel - V_sample`. |
| `Species MW (g/mol)` | `MW` | Molecular weight of the analyte. Used for mass-change and mass-concentration unit conversions. |

#### Transport Properties And Initial Conditions

| GUI label | Symbol | Meaning |
| --- | --- | --- |
| `Reference temperature Tref (C)` | `Tref` | Temperature at which `K_ref` and `D_ref` are defined. |
| `Partition coeff K @ Tref` | `K_ref` | Dimensionless solid/gas partition coefficient at `Tref`. In equilibrium, `c_gas = c_surface/K`. |
| `Sorption Enthalpy (kJ/mol)` | `Delta Hs` / `EaK` | Signed sorption enthalpy controlling `K(T)`. Negative values represent exothermic sorption and usually make `K` decrease on heating. The GUI stores this under the historical JSON key `EaK`. |
| `Diffusivity D @ Tref,c=0 (cm^2/s)` | `D_ref` | Diffusion coefficient at `Tref` and zero mobile concentration. This is the baseline value in the Arrhenius diffusivity law. |
| `Diffusivity Ea (kJ/mol)` | `EaD` | Activation energy for diffusion. Positive values make diffusion faster at higher temperature. |
| `Plasticizer power cD (uM, 'inf')` | `cD` | Concentration scale for concentration-dependent diffusivity, where finite `cD` gives `D` proportional to `exp(c/cD)`. Enter `inf` for concentration-independent diffusivity. |
| `Surface transfer coeff F (cm/s, 'inf')` | `F` | Surface mass-transfer coefficient in the Robin boundary condition. Enter `inf` for instantaneous surface/headspace equilibrium. Finite values model surface or gas-boundary-layer resistance. |
| `Initial mobile concentration (uM)` | `c0` | Initial mobile analyte concentration inside the solid sample. |
| `Initial gas concentration` | `cgas_init` | Initial well-mixed headspace concentration, entered in the unit chosen in the selector beside it (`ppbv`, `ppmv`, `ug/m3`, or `mg/m3`). Changing that selector converts the displayed value. The solver converts it to micromolar gas concentration, interpreting volumetric units at `T0`. |

#### Source Terms

| GUI label | Symbol | Meaning |
| --- | --- | --- |
| `Number of Sources` | none | Number of first-order source/reactant pools shown in the GUI. Click `Update Source Fields` after changing it. |
| `c0,i (uM)` | `c_source,i,0` | Initial concentration of source/reactant pool `i`. Source concentration is depleted as analyte is generated. |
| `Ai (1/min)` | `A_i` | Pre-exponential factor for first-order generation from source `i`. |
| `Ea,i (kJ/mol)` | `Ea,i` | Activation energy for source `i`. Higher values make generation more temperature-sensitive. |

#### Parameter Sweep Inputs

| GUI label | Meaning |
| --- | --- |
| `Enable` | Turns on sweep mode. `Run Simulation` will run multiple simulations instead of one. |
| `Sweep Parameter` | Selects the GUI parameter to vary. Source-parameter sweeps require that the selected source exists. |
| `Start Value` | First value in the sweep range, in the selected parameter's input units. |
| `Stop Value` | Last value in the sweep range, in the selected parameter's input units. |
| `Number of Points` | Number of sweep values. For `N`, values are rounded to whole numbers and duplicate rounded values are removed. |
| `Log-spaced Values` | Uses logarithmic spacing between start and stop. Both bounds must be positive. |
| `Colormap` | Matplotlib colormap used to color the sweep curves. |

#### Parameter Optimization Inputs

| GUI label | Meaning |
| --- | --- |
| `Enable` | Turns on optimization controls. Optimization is run with `Run Optimization`, not `Run Simulation`. |
| `# Parameters to Fit (1-6)` | Number of model parameters to fit simultaneously. |
| `Build Parameter Rows` | Rebuilds the optimization rows after changing the number of fit parameters. |
| `Parameter` | Parameter selected for fitting in that row. Each row must use a different parameter. Selecting a parameter fills in its default bounds; for `cD` the default lower bound is scaled to the run's expected concentrations so that `exp(c/cD)` stays finite. |
| `Min` | Lower bound for that fit parameter, in the parameter's input units. |
| `Max` | Upper bound for that fit parameter, in the parameter's input units. |
| `Log?` | Optimizes that parameter in log10 space. Bounds must be positive when enabled. |
| `Fit Target` | Experimental quantity used in the objective: `Concentration`, `Mass`, or `Both`. |
| `Concentration block weight` / `Mass block weight` | Relative weights of the two data blocks when `Fit Target` is `Both`. Each block is also normalized by its number of points, so a densely sampled block does not dominate. Must be > 0. |
| `Multi-Starts` | Number of least-squares starts. Start 1 uses current GUI values; additional starts are Latin-hypercube samples within the bounds. |
| `Max Evals per Start` | Maximum simulations allowed for each start, including those used to estimate the finite-difference Jacobian. |

#### Experimental Data Inputs

| GUI label | Meaning |
| --- | --- |
| `Duplicate times` | How observations that share a time are treated: `Retain replicates` (default; every observation is fitted), `Average replicates` (combined before fitting, inverse-variance weighted when every replicate has a σ), or `Reject duplicates` (repeated times are an error). |
| `Conc.: t, y[, σ]` | Pasted experimental concentration data: time (minutes), value, and an optional 1σ uncertainty. Values use the adjacent unit selector. |
| Concentration `Units` | Units for pasted concentration data: `ppbv`, `ppmv`, `ug/m3`, or `mg/m3`. |
| `Mass: t, y[, σ]` | Pasted experimental sample mass-change data: time (minutes), value, and an optional 1σ uncertainty. Values use the adjacent unit selector. |
| Mass `Units` | Units for pasted mass data: `ng`, `ug`, or `mg`. |
| `Resample` / `N` | When checked, draws a smooth guide through the pasted data at `N` equally spaced times. Display only: R-squared and fitting always use the original observations. |
| Interpolation method | Method for that display guide: `linear`, `cubic`, `pchip`, or `akima`. |

### Simulation Parameters

`Grid Scheme` can be `uniform`, `tanh`, or `geometric`. Non-uniform grids refine toward the sample surface. `Grid Stretch` is ignored for the uniform grid, is beta for the `tanh` grid, and is the coarse-to-fine spacing ratio for the `geometric` grid.

Solver tolerances are passed directly to SciPy's BDF solver as `rtol` and `atol`. The integration is split at every forcing discontinuity (flow switch, feed steps, ramp start and end), and each segment sees its forcing one-sided at its ends, so no solver step straddles a jump.

#### Accuracy Check And Mass-Balance Residual

The status line reports two quality measures:

- **Estimated error** (the accuracy check, on by default): the run is repeated at 10x tighter `rtol` and `atol`, and the largest difference in headspace concentration and in sample mass change is reported as a percentage of each curve's maximum. The comparison uses the curves as they are plotted and fitted (straight lines between output points). Values above 0.1 % are flagged and values above 1 % are marked as too large; tighten `rtol` and `atol` if either appears.
- **Mass-balance residual**: compares the analyte that left through the purge with the change in inventory. It is a bookkeeping check, not an accuracy measure. BDF conserves this balance exactly for finite `F` and for isothermal runs however inaccurate the solution, so a small residual proves nothing; it can only grow during temperature ramps with `F = inf`. A high residual still means the tolerances are too loose.

During fitting, runs with a high mass-balance residual are rejected, and the accuracy check is applied to the best fit.

### Carrier-flow basis

A gas flow in ml/min needs a temperature to mean anything, so the `Flow basis` selector says which one the `Flow Rate` uses. Both choices assume the chamber is at 1 atm.

- `Actual @ chamber T` (default): `Q` is the volumetric flow at the chamber's current temperature. When the chamber is heated at a fixed `Q`, fewer moles of carrier pass per minute (molar flow proportional to `1/T`). This matches a flow measured at chamber conditions and reproduces earlier versions exactly. If `Q` was really set or measured at room temperature, the purge is underestimated by about 30 % at 150 C and 43 % at 250 C, and a purge-dominated headspace concentration is overpredicted by the same factor.
- `Standard (sccm)`: `Q` is a mass-flow-controller setpoint referenced to `T_std` and 1 atm. The molar carrier flow is fixed whatever the chamber does, so the flow through the chamber is `Q_ch = Q*T/T_std`. While the flow is on, heating also expels headspace gas as it expands, which removes analyte at the fractional rate `(dT/dt)/T` (cooling draws gas back in). During the initial `Q = 0` period the vessel is treated as sealed. The feed tank's residence time uses the setpoint as entered (`Feed Tank Volume / Flow Rate`), which treats the tank as being at `T_std`.

With the standard basis, the vessel residence time readout shows its values at `T0` and `Tfinal`, and the Péclet number uses the chamber flow at `Tref`. The plotted and exported `Q (ml/min)` is always the actual flow through the chamber; the standard basis adds a `Q setpoint (sccm)` column. Sweeps and fits of `Q` vary the number as entered, so with the standard basis the sweep label's `ml/min` should be read as sccm. Parameter files store the choice as `flow_basis` and `flow_T_std` (schema 4); files from earlier versions load with the actual basis. The notebook implements the actual basis only.

### Temperature, Flow, And Feed Profiles

The temperature profile holds at `T0` for `Time at Initial Temp`, then ramps toward `Tfinal` at `Ramp Rate`, then holds at `Tfinal`.

The flow profile starts with `Q = 0` for `Initial time at Q = 0`, then uses the entered flow rate. Internally, flow is represented as an alternating no-flow/flush profile, with the GUI using the final simulation time as the flush interval.

The feed profile is a repeated staircase. After the initial hold, it rises by `delta` for `n_steps` steps, then falls by `delta` for `n_steps` steps, then uses the final hold. If `Number of Steps` is 0, the feed remains at `Base Conc`. A finite `Feed Tank Volume` smooths the ideal staircase with a residence time of `Feed Tank Volume / Flow Rate`; a volume of 0 uses the ideal square/staircase feed.

### Transport And Initial Conditions

`K_ref` and `D_ref` are evaluated at `Tref`. The GUI uses a signed sorption enthalpy convention in `Sorption Enthalpy`: negative values represent exothermic sorption.

`D_ref` is entered in `cm^2/s`. Internally, the solver uses minute-based rates where needed and reports diffusivity back in `cm^2/s`.

`cD` controls concentration-dependent diffusivity. Enter `inf` for concentration-independent diffusivity.

`F` controls surface mass transfer. Enter `inf` for instantaneous surface equilibrium, or a finite value in `cm/s` for mass-transfer-limited exchange.

`Initial gas concentration` is entered in the unit chosen in the selector beside it. The GUI converts it to the model's internal micromolar gas concentration, interpreting volumetric units at the initial temperature. Parameter files written by older versions are migrated on load: v10 files keep the value in the plot unit it was entered in, and files from v9 or earlier, which stored µM, are converted to ppbv.

### Source Terms

Set `Number of Sources`, then click `Update Source Fields` to rebuild the source rows. Each source has:

| Field | Meaning |
| --- | --- |
| `c0` | Initial source concentration in micromolar units. |
| `A` | First-order pre-exponential factor in `1/min`. |
| `Ea` | Source activation energy in `kJ/mol`. |

The GUI caps the visible source-term rows at 20. The sweep and optimization selectors expose source parameters for Source 1 through Source 3.

### Plots

Each successful run updates seven plots:

- Temperature.
- Feed concentration.
- Flow rate.
- Surface diffusivity.
- Solubility.
- Sample mass change.
- Headspace gas concentration.

Experimental mass data overlays on the sample mass-change plot. Experimental concentration data overlays on the headspace gas-concentration plot. R-squared annotations are shown for single runs when compatible experimental data are available.

### Experimental Data

The bottom-right `Experimental Data` panel has separate boxes for concentration data and mass data. Each row should contain a time and a value, optionally followed by a 1σ uncertainty. Points with an uncertainty are weighted by it in fitting; points without one are scaled by the standard deviation of that block's values.

Accepted formats:

```text
0, 0
10, 123.4, 5.0
20 150.2
30	175.0
```

Comma, tab, and whitespace delimiters are accepted. Lines beginning with `#` are ignored. Malformed rows, such as a header, are skipped and listed in a diagnostics message. The `Duplicate times` selector controls repeated observation times: retain every replicate (default), average replicates before fitting, or reject them.

Use the unit dropdown next to each data box to describe the units of the pasted data. These data-unit dropdowns are independent from the plot-unit dropdowns in the top bar. The GUI converts model curves for plotting and comparison as needed.

Check `Resample` to draw a smooth guide through the pasted data at `N` evenly spaced points, using `linear`, `cubic`, `pchip`, or `akima` interpolation. This is display only: R-squared and fitting always use the original observations. Akima interpolation falls back to linear if there are too few points, and a guide that dips below zero for nonnegative data is redrawn with PCHIP.

### Parameter Sweeps

Enable `Parameter Sweep` to run the same model over a range of one selected parameter.

Sweep workflow:

1. Check `Enable` in the `Parameter Sweep` box.
2. Select a sweep parameter.
3. Enter start value, stop value, and number of points.
4. Check `Log-spaced Values` for logarithmic spacing. Start and stop must be positive for log spacing.
5. Choose a colormap.
6. Click `Run Simulation`.

Sweepable parameters include temperature parameters, flow rate, feed-tank volume, sample properties, grid point count, grid stretch, transport properties, initial mobile concentration, feed base concentration, and Source 1 through Source 3 parameters.

Notes:

- Sweeping `N` rounds values to whole numbers and removes duplicates introduced by rounding.
- Sweeping `Grid Stretch` only affects `tanh` and `geometric` grids. The GUI rejects this sweep for the uniform grid because every run would be identical.
- Sweeping a source parameter requires that the corresponding source row exists.
- Sweep results are overlaid on the plots and exported with a sweep summary plus per-run data blocks.

### Parameter Optimization

Enable `Parameter Optimization` to fit selected model parameters to experimental data.

Optimization workflow:

1. Paste concentration and/or mass experimental data.
2. Select the correct data units for the pasted data.
3. Check `Enable` in the `Parameter Optimization` box.
4. Set `# Parameters to Fit` from 1 to 6.
5. Click `Build Parameter Rows`.
6. For each row, choose a parameter, lower bound, upper bound, and whether to optimize in log space.
7. Choose `Fit Target`: `Concentration`, `Mass`, or `Both`, and for `Both` the two block weights.
8. Set `Multi-Starts` and `Max Evals per Start`.
9. Click `Run Optimization`.

The optimizer is SciPy's bounded `least_squares` (trust-region reflective) with a finite-difference Jacobian. Start 1 uses the current GUI values, clipped into the specified bounds. Additional starts are generated with a reproducible Latin-hypercube sample of the bounds.

Fit requirements and behavior:

- Each optimization row must use a different parameter.
- Log-scaled bounds must be positive.
- `N` and `Grid Stretch` are not fit parameters.
- Source parameters can only be fit when the corresponding source exists. A changed `Number of Sources` is applied before fitting, as it is before a simulation.
- The selected target must have matching experimental data; observations outside the simulated time window are excluded with a warning.
- The comparison uses the unit selected beside the pasted experimental data.
- Runs that fail or have a high mass-balance residual are penalized during fitting.
- `Terminate Fit` stops after the current model evaluation.
- With `Accuracy check` on, the best fit is re-solved at tighter tolerances and its estimated error is shown on the status line and exported.

When fitting finishes, the GUI writes the best-fit values back into the corresponding input fields, overlays each completed start's best-fit curve, and opens a multi-start results table. The table reports start status, simulations used, SSE, Jacobian rank and condition number, optional R-squared values, initial guesses, and fitted values with local linearized standard errors. Standard errors are suppressed when the fit is rank-deficient, ill-conditioned, at a bound, or short of degrees of freedom, and are unavailable when a start stops at its simulation budget. Use `Copy Table (TSV)` to copy that table to the clipboard.

### Exporting Results

Click `Export Data` after a simulation, sweep, or optimization result has been generated.

For a single run, the CSV includes:

- Release identifier.
- Base simulation parameters.
- Calculated constants shown in the GUI.
- Mass-balance residual, accuracy-check result, and solve time.
- R-squared values when experimental data are available.
- Time-series results.
- Pasted experimental data when present.

For a sweep, the CSV includes:

- Sweep metadata.
- Base simulation parameters.
- Calculated constants.
- One-row-per-sweep-point summary, including each point's accuracy-check result.
- Per-run time-series blocks.
- Pasted experimental data when present.

Every export also saves seven 300-dpi PNG files beside the CSV, one for each plot panel, using the plot units and plot-scale settings captured when the run started.

## Running the Notebook

Launch Jupyter and open the reference notebook:

```bash
jupyter notebook "Outgassing_MOL_Model_v14.ipynb"
```

Run all cells to execute the default case. Edit the simulation-parameter cell to change geometry, temperature, flow, feed profile, source terms, transport parameters, or spatial grid settings.

## Model Features

The model supports:

- Geometries: slab (`m = 0`), cylinder (`m = 1`), and sphere (`m = 2`).
- Spatial grids: `uniform`, `tanh`, and `geometric`, with non-uniform grids refined toward the sample surface.
- Diffusivity: Arrhenius temperature dependence with optional concentration dependence through finite `cD`.
- Surface boundary conditions: instantaneous equilibrium (`F = inf`) or finite surface mass transfer (`F` in cm/s).
- Temperature profiles: initial hold followed by heating or cooling to a target temperature.
- Flow profiles: alternating no-flow/equilibration and flushing periods, with the flow given as actual flow at chamber temperature or as a standard (sccm) mass-flow-controller setpoint (GUI).
- Feed concentration profiles: stepped rise/fall profiles with optional finite feed-tank roll-over in the GUI.
- Chemical sources: any number of first-order Arrhenius source terms in the model; the GUI exposes up to 20 source rows.
- Outputs: temperature, feed concentration, flow, surface diffusivity, solubility, sample mass change, headspace gas concentration, solve time, and mass-balance error.

The main transport modes are controlled by `F` and `cD`:

| `F` | `cD` | Diffusivity | Surface condition |
| --- | --- | --- | --- |
| `inf` | `inf` | Constant at a given temperature | Instantaneous equilibrium |
| `inf` | finite | Concentration-dependent | Instantaneous equilibrium |
| finite | `inf` | Constant at a given temperature | Mass-transfer limited |
| finite | finite | Concentration-dependent | Mass-transfer limited |

## Parameter Conventions

- Time is in minutes unless otherwise noted.
- Temperatures are entered in degrees Celsius.
- Diffusivity inputs are in `cm^2/s`; the GUI reports diffusivity in `cm^2/s`.
- Surface transfer coefficient `F` is in `cm/s`; enter `inf` for instantaneous surface equilibrium.
- `cD`, mobile concentration, source concentration, and model gas concentration use micromolar units internally.
- GUI sample radius / half-thickness is entered in micrometers; the notebook example uses centimeters.
- Feed concentration is entered in ppbv, and the GUI can display concentration as ppbv, ppmv, ug/m^3, or mg/m^3.
- Sample mass change is computed in ng and can be displayed/exported as ng, ug, or mg.
- Sorption enthalpy is signed: negative values represent exothermic sorption. The GUI stores this value under the historical key `EaK`; the notebook uses `DeltaHs`.
- The GUI uses a configurable reference temperature `Tref` for `K_ref` and `D_ref`. The notebook variables `K50` and `D50` are referenced to 50 C.

## Example Parameter Files

`outgassing_defaults_example_sweep.json` demonstrates the sweep workflow. It models a 200 um sphere with two chemical source terms, a temperature ramp from 25 C to 250 C, stepped feed concentration, and a sweep of `D_ref` from `1e-8` to `1e-6 cm^2/s`.

`kumar_fig7_updated_with F fit.json` demonstrates fitting against embedded experimental concentration data. It models a slab geometry with phenol-like analyte properties, finite surface mass transfer, a tanh-refined grid, finite feed-tank volume, and optimization of `K_ref`, `D_ref`, and `F`.

## License And Release

This project is released under the MIT License. See `LICENSE.txt` for the license terms and `NOTICE.txt` for the LLNL/DOE notice.

Release identifier: `LLNL-CODE-2017385`

SPDX-License-Identifier: `MIT`
