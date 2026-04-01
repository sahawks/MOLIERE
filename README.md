# A Method Of Lines Integrator for Emissions Research and Exploration (MOLIERE)

MOLIERE is a numerical simulation tool for modeling outgassing and diffusion from solid materials into a gas-phase headspace. It solves the coupled diffusion–partitioning–flow system using the Method of Lines (MOL) with SciPy's BDF integrator. The model supports slab, cylindrical, and spherical geometries with optional concentration-dependent diffusivity, finite surface mass-transfer resistance, chemical source terms, and cyclic flow/feed concentration profiles.

## Getting Started

### Prerequisites

- Python 3.9 or later
- NumPy >= 1.21
- SciPy >= 1.7
- Matplotlib >= 3.5
- pandas >= 1.3 (required for the GUI and data export only)

Install all dependencies with:

```bash
pip install numpy scipy matplotlib pandas
```

To run the Jupyter notebooks you will also need Jupyter:

```bash
pip install jupyter
```

## Getting Started Example

**Jupyter notebooks** — Open either notebook and run all cells. The default parameters simulate a 150-minute outgassing experiment on a 200 µm radius sphere with a temperature ramp from 50 °C to 250 °C at 5 °C/min, a single chemical source term, and a 20 ml/min carrier gas flush:

```bash
jupyter notebook Outgassing_MOL_Model_v8.ipynb
```

Edit the "Input Parameters" cell to change any simulation conditions. Results are plotted automatically and can be exported to the clipboard by uncommenting the final line.

**GUI** — Launch the graphical interface directly:

```bash
python outgassing_gui_with_sweep_4.pyw
```

Enter parameters in the left panel and click "Run Simulation" to generate plots. The GUI also supports parameter sweeps (vary one parameter across a range of values), experimental data overlay with R² calculation, and least-squares optimization to fit model parameters to measured data. Parameters can be saved to and loaded from JSON files.

## What the codes do

The repository contains three implementations of the same underlying physics:

| File | Description |
|------|-------------|
| `Outgassing_MOL_Model_v8.ipynb` | **Reference notebook.** A clear, self-contained Jupyter notebook intended as the canonical implementation. All functions are documented with docstrings and the code prioritizes readability. |
| `Outgassing_MOL_Model_v8_speed_optimized.ipynb` | **Speed-optimized notebook.** Preserves the same API and produces numerically identical results, but runs approximately 35–40% faster through pre-computed constants, closure-based ODE factories, sparse Jacobian storage, and an analytical Jacobian for the most common case (infinite surface transfer with constant diffusivity). |
| `outgassing_gui_with_sweep_4.pyw` | **Tkinter GUI application.** Provides a graphical interface for all model parameters with seven real-time plots, parameter sweeps with configurable colormaps, experimental data input with interpolation options, and multi-parameter curve-fitting optimization using differential evolution, dual annealing, or Nelder-Mead. Supports saving/loading parameter sets as JSON files and exporting results to CSV. |
| `outgassing_defaults_example_sweep.json` | **Example parameter file — sweep demo.** Demonstrates the parameter sweep feature. Load into the GUI via "Load Params…" and click "Run Simulation." |
| `outgassing_defaults_kumar_fit.json` | **Example parameter file — optimization demo.** Demonstrates curve fitting against experimental sorption/desorption data. Corresponds to Figure 3 in the manuscript. Load into the GUI and click "Run Optimization." |

**Model physics.** The model discretizes the radial diffusion equation on a uniform grid and integrates the resulting ODE system with SciPy's implicit BDF solver. Four operating modes are selected by the values of `F` (surface mass-transfer coefficient) and `cD` (plasticizer power):

| `F` | `cD` | Diffusivity | Surface boundary condition |
|-----|------|-------------|----------------------------|
| `inf` | `inf` | Constant (Arrhenius only) | Instantaneous equilibrium |
| `inf` | finite | Concentration-dependent | Instantaneous equilibrium |
| finite | `inf` | Constant (Arrhenius only) | Mass-transfer limited |
| finite | finite | Concentration-dependent | Mass-transfer limited |

Additional features include temperature ramps (heating or cooling), cyclic flow on/off profiles, stepped feed-gas concentration profiles, and first-order chemical source terms with Arrhenius kinetics.

## Example Parameter Files

Two JSON parameter files are included to demonstrate the GUI's sweep and optimization features. To use them, launch the GUI, click "Load Params…", and select the desired file. All GUI fields — including sweep settings, optimization configuration, and experimental data — will be populated automatically.

**`outgassing_defaults_example_sweep.json`** — A parameter sweep example that varies the diffusivity `D` across five logarithmically spaced values from 10⁻⁸ to 10⁻⁶ cm²/s. The scenario models a 200 µm sphere with two chemical source terms, a temperature ramp from 25 °C to 250 °C at 2 °C/min, and a stepped feed concentration profile. Load the file and click "Run Simulation" to see the sweep overlay plots showing how diffusivity affects the outgassing profile.

**`outgassing_defaults_kumar_fit.json`** — An optimization example that fits four transport parameters (K, D, cD, and F) to experimental sorption/desorption concentration data. The scenario models a 1.9 cm slab geometry with phenol (MW = 94.11 g/mol) at 25.6 °C with concentration-dependent diffusivity. The experimental data (32 time–concentration pairs) is embedded in the file and will appear in the GUI's experimental data panel. Load the file and click "Run Optimization" to fit the model; the optimizer will update the GUI fields with the best-fit values when complete. You can also click "Run Simulation" first to see the initial (pre-fit) model overlaid on the experimental data.

**Important: EaK sign convention.** The notebooks and the GUI use opposite sign conventions for the sorption enthalpy parameter `EaK`. The notebooks treat `EaK` as a positive magnitude (e.g., `EaK = 35`), while the GUI uses the thermodynamic sign convention where exothermic sorption is negative (e.g., `EaK = -35`). Both produce identical physics because the internal formulas account for the convention. You must negate `EaK` when transferring parameters between the notebooks and the GUI. The GUI also accepts a user-defined reference temperature (`Tref`) for the partition coefficient and diffusivity, whereas the notebooks fix the reference at 50 °C.

## License
MIT License

Copyright (c) 2026, Lawrence Livermore National Security, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Release

LLNL-CODE-2017385

SPDX-License-Identifier: MIT
