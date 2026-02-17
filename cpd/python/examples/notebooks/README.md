# Notebook Examples

This folder contains quickstart Jupyter notebooks for the main changepoint-doctor workflows:

- `01_offline_algorithms.ipynb`: compare offline detectors on noisy KPI-style data.
- `02_online_algorithms.ipynb`: run streaming detectors on service-latency data.
- `03_doctor_recommendations.ipynb`: generate and inspect doctor recommendations, with live CLI + fallback snapshot support.

## Run

From `/Users/admin/Documents/Work/claude-doctor-changepoint/cpd/python`:

```bash
python -m pip install jupyter matplotlib
jupyter lab
```

Then open `examples/notebooks/` in Jupyter and run notebooks top-to-bottom.

## Notes

- Sample data is generated in notebook cells with fixed random seeds for reproducibility.
- Notebook 3 will try live `cpd doctor` first, then fall back to `data/doctor_recommendations_snapshot.json` if CLI execution is unavailable.
