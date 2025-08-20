# Spectral Neural Networks for Solving PDEs

This repository contains the code for a Master's thesis research project on using Spectral Neural Networks (SNNs) for solving partial differential equations (PDEs). The framework is built on `DeepXDE` and `PyTorch`, with experiment tracking managed by `MLflow`.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://gitlab.inria.fr/begiraul/master2025-eduard-snn.git
    cd master2025-eduard-snn
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start the MLflow Server (optional, for local tracking):**
    ```bash
    mlflow server --host 127.0.0.1 --port 5000
    ```
    Then, open your browser to `http://127.0.0.1:5000` to view the MLflow UI.

## How to Run an Experiment

Experiments are launched using the `run_experiment.py` script, which requires a configuration file. To run a single experiment (e.g., the `SpectralNN` model on the wave equation), use the following command:

```bash
python run_experiment.py --config configs/wave_snn.yaml
```

You can add a random seed for reproducibility:

```bash
python run_experiment.py --config configs/wave_snn.yaml --seed 42
```