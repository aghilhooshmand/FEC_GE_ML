# FEC_GE - Fitness Evaluation Cache for Grammatical Evolution

This project implements a Fitness Evaluation Cache (FEC) system for Grammatical Evolution (GE) experiments, with support for multiple sampling methods and comprehensive result tracking.

## Project Structure

```
FEC_GE/
├── classification_fec_ge.py  # Main entry point
├── config.py                # Configuration file
├── util.py                  # Core utilities and FEC implementation
├── requirements.txt         # Python dependencies
├── run_experiment.sh        # Bash script to run experiments
├── data/                    # Dataset files (CSV)
├── grammars/                # BNF grammar files
├── operators/               # Custom operators
├── grape/                   # Grape library (GE implementation)
└── results/                 # Output directory (created automatically)
```

## Setup

1. **Install dependencies:**
   ```bash
   ./run_experiment.sh
   ```
   This will create a virtual environment and install all required packages.

   Or manually:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure your experiment:**
   Edit `config.py` to set:
   - Dataset file and label column
   - Grammar file
   - Evolution parameters (population, generations, runs)
   - FEC settings (sample sizes, sampling methods, modes)
   - MLflow tracking settings

## Running Experiments

### Basic Usage

```bash
# Using the bash script (recommended)
./run_experiment.sh

# Or directly with Python
source venv/bin/activate
python3 classification_fec_ge.py
```

### Resuming Experiments

The script automatically detects and resumes from the latest CSV file. To resume from a specific file:

1. Edit `config.py`:
   ```python
   "resume.from_csv": "run002/run002_all_experiments_20251123_152010.csv",
   ```

2. Or set to `None` for auto-detection:
   ```python
   "resume.from_csv": None,  # Auto-detect latest
   ```

## Configuration

### FEC Modes

Control which FEC modes to run:
- `fec.modes.fec_disabled`: Baseline without FEC
- `fec.modes.fec_enabled_behaviour`: FEC with behavior similarity
- `fec.modes.fec_enabled_structural`: FEC with structural similarity
- `fec.modes.fec_enabled_behaviour_structural`: FEC with both similarities

### Sampling Methods

Enable/disable sampling methods:
```python
"fec.sampling_methods.enabled": {
    "kmeans": True,
    "kmedoids": True,
    "farthest_point": True,
    "stratified": False,
    "random": True,
}
```

## Output

Results are saved in `results/` directory:
- **CSV files**: `runXXX_all_experiments_YYYYMMDD_HHMMSS.csv` - All experiment results
- **HTML reports**: Consolidated charts and comparisons
- **MLflow**: Experiment tracking (if enabled)

## Features

- **Multiple Sampling Methods**: K-means, K-medoids, Farthest Point Sampling, Stratified, Random
- **Resume Capability**: Automatically resume from last completed configuration
- **Comprehensive Metrics**: Hit rates, fake hit rates, fitness, complexity
- **Interactive Charts**: Plotly-based HTML reports
- **MLflow Integration**: Experiment tracking and artifact storage

## Dependencies

- Python 3.7+
- numpy<2.0
- pandas
- plotly
- mlflow
- deap
- scikit-learn
- scikit-learn-extra

## License

[Add your license here]

