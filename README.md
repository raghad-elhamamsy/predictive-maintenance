# Signal Monitor: Synthetic Early-Warning Demo

A reproducible example of time-series feature engineering, early-warning classification, anomaly detection, and a local dashboard. The repository uses **generated synthetic data only**. It contains no organization records, source-system identifiers, physical tag names, or claims about a real asset or sector.

All public examples use generic `channel_NNN` identifiers. Figures, metrics, and dates produced by the notebooks describe only the generated fixture and must not be presented as operational evidence.

## Run the Example

Install the packages in `requirements.txt`, then run the full isolated workflow:

```bash
python tests/fresh_clone_smoke.py
```

The smoke test creates a temporary copy, generates an anonymous workbook, runs notebooks 01–08, and checks regenerated outputs. It does not read local files under `data/`.

To open the local Flask interface after generating the example artifacts:

```bash
python app.py
```

The dashboard is a demonstration, not a validated decision or safety system. Do not upload or commit organization data.

## Notebooks

1. `01_load_and_channel_dictionary.ipynb` loads the synthetic channel workbook.
2. `02_eda.ipynb` explores data quality and signal behavior.
3. `03_cleaning_and_shutdowns.ipynb` demonstrates cleaning and state segmentation.
4. `04_features_and_labels.ipynb` builds time-aware features and labels.
5. `05_early_warning_models.ipynb` trains example classifiers.
6. `06_anomaly_detection_.ipynb` compares anomaly scores.
7. `07_evaluation_and_comparison.ipynb` compares generated score streams.
8. `08_evidence_audit.ipynb` summarizes the synthetic example and its limitations.

Generated workbooks, tables, models, and plots are excluded from version control. The only data generator is `tests/generate_synthetic_channels.py`.
