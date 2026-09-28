# SWR network simulations with OLM interneurons

This repository contains Brian2 simulations of a hippocampal spiking network
with pyramidal cells (PC), parvalbumin-positive basket cells (PVBC), and
oriens-lacunosum moleculare (OLM) interneurons. The main script reproduces four
network-state examples and saves publication-ready raster, firing-rate,
sharp-wave, and ripple panels.

## Repository contents

```text
network_simulations/
|-- run_spiking.py
|-- helper_functions/
|   |-- construct_spiking_network.py
|   |-- simulate_spiking.py
|   |-- plot_spiking_results.py
|   |-- utils_spiking.py
|   `-- detect_peaks.py
`-- results/
    |-- swr_slice_network_step4.npz
    |-- swr_slice_network_step6.npz
    `-- figures/
        |-- Fig1.png / Fig1.pdf / Fig1.svg
        |-- Fig2.png / Fig2.pdf / Fig2.svg
        |-- Fig3.png / Fig3.pdf / Fig3.svg
        `-- Fig4.png / Fig4.pdf / Fig4.svg
```

The precomputed `step4` and `step6` network files are required for direct
figure reproduction. Do not omit them when publishing the repository.

## Tested environment

The current code was tested with:

- Python 3.9.19
- Brian2 2.6.0
- NumPy 1.23.5
- SciPy 1.10.1
- Matplotlib 3.9.2

Later package versions may also work, but the versions above are recommended
for reproducibility.

## Installation

Create and activate a clean environment.

### Conda

```bash
conda create -n swr-olm python=3.9.19 -y
conda activate swr-olm
python -m pip install brian2==2.6.0 numpy==1.23.5 scipy==1.10.1 matplotlib==3.9.2
```

### Python virtual environment

Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install brian2==2.6.0 numpy==1.23.5 scipy==1.10.1 matplotlib==3.9.2
```

Linux/macOS:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install brian2==2.6.0 numpy==1.23.5 scipy==1.10.1 matplotlib==3.9.2
```

## Reproducing the four figures

Run the script from the repository root:

```bash
python run_spiking.py
```

`run_spiking.py` sets the Brian2 random seed to `3` and executes all four
simulations sequentially. Each plot is written in PNG, PDF, and SVG format to
`results/figures/`. Existing files with the same names are overwritten.

| Output | Function | Network file | Simulation condition |
|---|---|---|---|
| `Fig1` | `simulate_1` | `swr_slice_network_step4.npz` | Baseline non-SWR activity without current stimulation |
| `Fig2` | `simulate_2` | `swr_slice_network_step4.npz` | A 200 pA, 10 ms pulse to all PC neurons switches the network into the SWR state |
| `Fig3` | `simulate_3` | `swr_slice_network_step4.npz` | Positive and negative 200 pA, 10 ms PC pulses switch the network from non-SWR to SWR and back to non-SWR |
| `Fig4` | `simulate_4` | `swr_slice_network_step6.npz` | Current-evoked activity with dynamic PVBC-to-OLM synaptic depression |

The current calls are defined in `run_spiking.py`:

```python
simulate_1(filename_spiking, figure_name='Fig1')
simulate_2(filename_spiking, sigma_P=200 * pA,
           fraction_stim=1.0, figure_name='Fig2')
simulate_3(filename_spiking, sigma_P=200 * pA,
           fraction_stim=1.0, figure_name='Fig3')
simulate_4(filename_spiking, sigma_P=200 * pA,
           fraction_stim=1.0, figure_name='Fig4')
```

`figure_name` is a filename stem. For example, `figure_name='control_seed3'`
creates `control_seed3.png`, `control_seed3.pdf`, and `control_seed3.svg` and
therefore prevents results from different simulations from overwriting one
another.

## Rebuilding the network

Precomputed network files are included to make direct reproduction easier. To
rebuild the network from the construction stages, uncomment the construction
block in `run_spiking.py`:

```python
step1_PA(filename_spiking, make_plots=False)
step2_PB(filename_spiking, make_plots=False)
step3_add_Wab(filename_spiking, make_plots=False)
step4_merge_subnetworks(filename_spiking, make_plots=False)
step5_current_injection(filename_spiking, curr_to_pop='P')
step6_spontaneous(filename_spiking)
```

## Figure contents

Each output figure contains:

1. Raster plots for sampled PC, PVBC, and OLM neurons.
2. Smoothed PC population firing rate.
3. Smoothed PVBC population firing rate.
4. Smoothed OLM population firing rate.
5. Low-frequency sharp-wave component.
6. Ripple-band component (90-180 Hz).

The plotting implementation is separated from the network simulations in
`helper_functions/plot_spiking_results.py`.