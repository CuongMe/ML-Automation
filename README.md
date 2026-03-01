# Machine Learning Automation (Streamlit)

A Python Streamlit app that automates core ML training steps:

- Load CSV data
- Select target column
- Auto detect classification vs regression
- Auto preprocess numeric/categorical features
- Train one model or auto-select the best model by cross-validation
- Evaluate with relevant metrics
- Download trained model pipeline
- Run batch predictions on new CSV files

## 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run

```bash
streamlit run app.py
```

## 3. Workflow

1. Upload a training CSV.
2. Select the target column.
3. Keep task type on auto-detect or set it manually.
4. Choose model settings in the sidebar.
5. Click **Train Model**.
6. Review metrics, download the model, and optionally run batch predictions.

## Notes

- For regression, target values must be numeric.
- The downloaded `.joblib` file contains preprocessing + model in one pipeline object.
