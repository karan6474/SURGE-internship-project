# Physics-Informed Neural Network for 3D Metal Additive Manufacturing Simulation

A PyTorch-based **Physics-Informed Neural Network (PINN)** that models the temperature, velocity, and pressure fields inside the melt pool of a Laser Powder Bed Fusion (L-PBF) metal 3D printing process, and uses the predicted fields to estimate manufacturing defect risk.

## Overview

Simulating the melt pool in metal additive manufacturing with traditional CFD is computationally expensive. This project trains a neural network to approximate the solution of the governing 3D transient Navier–Stokes and energy equations (with a moving Gaussian laser heat source and a solid–liquid phase-change term), using a combination of:

- **Data loss** — fitting simulation output (temperature, velocity, pressure) at sampled points.
- **Physics loss** — enforcing the PDE residuals (momentum, continuity, energy) at randomly sampled collocation points via automatic differentiation, so the network respects the underlying physics even where no data exists.

Once trained, the model is queried on a dense 3D grid to reconstruct the melt pool and estimate:
- **Melt pool geometry** (length, width, depth) from a temperature threshold (liquidus).
- **Balling Susceptibility Index (BSI)** — risk of the melt track breaking into discrete balls instead of a continuous bead.
- **Cracking Susceptibility Index (CSI)** — risk of solidification cracking, from cooling rate, thermal gradient/growth-rate ratio, relaxation-time ratio, and solidification stress.

## Physics Modelled

The PINN enforces:

- **Continuity**: ∇·u = 0
- **Momentum** (x, y, z): ρ(∂u/∂t + u·∇u) = −∇P + μ∇²u
- **Energy**, including latent heat of fusion via a liquid-fraction term f_L(T):
  ρc_p(∂T/∂t + u·∇T) + ρL(∂f_L/∂t + u·∇f_L) = κ∇²T + Q
- **Moving Gaussian laser heat source**:
  Q(x, y, t) = (2·P_laser·η / πr_b²) · exp(−2[(x − V_s·t)² + y²] / r_b²)

Material properties are set for **Ti-6Al-4V** and can be edited in the script:

| Property | Symbol | Value |
|---|---|---|
| Density | ρ | 4430 kg/m³ |
| Dynamic viscosity | μ | 0.004 Pa·s |
| Specific heat | c_p | 525 J/(kg·K) |
| Thermal conductivity | κ | 29 W/(m·K) |
| Latent heat of fusion | L | 2.84×10⁴ J/kg |
| Solidus temperature | T_s | 1878 K |
| Liquidus temperature | T_l | 1928 K |
| Young's modulus (for stress calc) | E | 110 GPa |

## Model Architecture

- **Network**: fully-connected feedforward network (`FCNN`), input `[x, y, z, t]` (4-D), output `[T, vx, vy, vz, P]` (5-D). Configurable hidden layers (e.g. `[4, 256, 256, 256, 256, 5]`).
- **Activation**: Tanh (later variants use ReLU).
- **Collocation points**: sampled via **Latin Hypercube Sampling** (`pyDOE.lhs`) over the domain bounds for evaluating the physics loss.
- **Training**: two-stage — pretraining on data (MSE against simulation output), then main training combining data loss and physics-residual loss, with Adam and an optional `ReduceLROnPlateau` scheduler.
- **Outputs are normalized** (z-score) during training and rescaled at inference.

## Repository Structure

```
├── ML3D.ipynb          # Main notebook: data loading, PINN definition, training, evaluation, melt-pool analysis
└── README.md
```

> Note: the notebook iterates through a few versions of the PINN (`Tanh`/`ReLU` activations, with/without second-derivative terms, single-loop vs. two-stage training). The final cells contain the most complete version, including melt-pool analysis and defect-index calculation.

## Data

The notebook expects per-timestep simulation output as CSV files (e.g. exported from a CFD tool such as OpenFOAM), each containing the columns:

| Column | Description |
|---|---|
| `Points:0`, `Points:1`, `Points:2` | Spatial coordinates (x, y, z) |
| `Time` | Simulation time |
| `T` | Temperature |
| `U:0`, `U:1`, `U:2` | Velocity components (vx, vy, vz) |
| `p_rgh` | Pressure |

Multiple per-timestep CSVs are concatenated into one merged dataset before training.

## Installation

```bash
pip install torch numpy pandas scikit-learn matplotlib scipy pyDOE openpyxl
```

## Usage

1. Update the CSV file paths at the top of the notebook to point to your simulation output.
2. Run the data-merging cell to combine all timestep files into a single dataframe.
3. Run the PINN definition and training cells (`train_pre` / `train_main`, or the combined `train` method in the final version).
4. Run the evaluation cells to get MAE / MSE / R² per output field and predicted-vs-true plots.
5. Run `analyze_melt_pool(...)` to reconstruct the melt pool on a dense grid and compute geometry and defect-susceptibility indices.

## Outputs

- Predicted vs. true evaluation (MAE / MSE / R²) for temperature, velocity components, and pressure.
- Melt pool geometry, peak temperature/velocity, and BSI / CSI defect predictions.

## Results

*From a training run of 200 pretraining iterations + 500 main-training iterations (final loss: 3.87×10⁻¹).*

### Evaluation metrics

| Field | MAE | MSE | R² |
|---|---|---|---|
| Temperature (T) | 77.17 K | 19,472.66 | 0.885 |
| Velocity x (U:0) | 0.194 m/s | 0.108 | 0.441 |
| Velocity y (U:1) | 0.164 m/s | 0.077 | 0.436 |
| Velocity z (U:2) | 0.210 m/s | 0.135 | 0.589 |
| Pressure (p_rgh) | 8,736.00 | 2.476×10⁸ | 0.765 |

> Temperature and pressure are predicted well (R² > 0.76), while the velocity components are notably weaker (R² ≈ 0.44–0.59) — the momentum/physics loss weighting or collocation sampling likely needs further tuning for the velocity field.

### Melt pool characteristics

- **Dimensions (L×W×D):** 0.167 mm × 0.126 mm × 0.133 mm
- **Max temperature:** 4423.3 K
- **Max velocity:** 3.61 m/s
- **Cooling rate:** 4.24×10⁶ K/s
- **Thermal gradient / growth-rate ratio (G/R):** 1.35×10⁷ K·s/m
- **Solidification stress:** 8.06 MPa (characteristic crack length: 37.7 μm)
- **Balling Susceptibility Index (BSI):** −7.72 → **balling unlikely**
- **Cracking Susceptibility Index (CSI):** 0.783 → **cracking likely**

## Limitations & Future Work

- Velocity field predictions (R² ≈ 0.44–0.59) are notably weaker than temperature and pressure (R² > 0.76); revisiting the physics-loss weighting, collocation point density, or network capacity may help.
- Trained per print condition; a new run is needed for different laser power/speed settings unless these are added as network inputs.
- Physics loss currently omits explicit boundary conditions (free surface, substrate) in the primary pretrain/main-train version.
- Defect-index formulas (BSI, CSI, solidification stress) use empirical correlations from literature and should be validated against experimental data.
- Potential extensions: parameterize laser power/scan speed as network inputs, add boundary conditions, use a Fourier feature/SIREN-style network for higher-frequency melt-pool fields.

## License

Add a license of your choice (e.g. MIT) if you intend to share this publicly.
