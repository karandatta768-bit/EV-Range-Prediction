# EV Range Prediction

This project is a Streamlit app that estimates EV driving range from battery, driving, terrain, and weather inputs.

## What is in the project

- `app.py`: Streamlit interface for interactive predictions.
- `ev_range_core.py`: Shared feature definitions, model pipeline, and UI helper logic.
- `train_model.py`: Reproducible training script that builds the saved model used by the app.
- `model/ev_range_model.joblib`: Trained scikit-learn pipeline loaded by the app.
- `Electric_Vehicle_Population_Data (2).csv`: Source dataset used to generate training scenarios.

## Important note about the model

The Washington EV population dataset does not directly contain all real-world driving inputs shown in the app, such as speed, tyre pressure, payload, or terrain. To keep the project reproducible, `train_model.py` creates realistic scenario features from the source data and trains the model on those generated driving conditions.

That means this project is best understood as a portfolio/demo ML app for EV range estimation, not a certified real-world vehicle range calculator.

## Run the app

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train or refresh the model:

```bash
python train_model.py
```

4. Start the Streamlit app:

```bash
streamlit run app.py
```

## Verify the project

Run the lightweight test suite with:

```bash
python -m unittest discover -s tests
```

## Current structure note

There is also a nested `EV-range-prediction` repository inside this folder. I left it untouched because removing or rewriting it could discard work if you still need it. If you want, that can be cleaned up as a separate step after confirming which copy is the canonical one.
