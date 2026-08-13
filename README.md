# Graduate Placement Prediction

**A Refactory Academy capstone project predicting graduate employment outcomes to support proactive placement.**

## Machine learning goal

Predict whether a Refactory Academy graduate will be employed within 6 months
of graduation, so the placement team can identify and support at-risk
graduates early.

**Task:** Binary classification — `Employed` vs `Open to work`.

## Dataset

- **Source:** Historical Refactory Academy graduate placement records
  (internal, anonymized — not publicly available on Kaggle or elsewhere).
- **Size:** 2,387 rows, 12 raw columns.

## Repository structure

```
graduate_placement_dataset.csv          Raw source data
clean_pipeline.py                        Standalone cleaning script (used by the scheduled workflow)
graduate_placement_cleaned_FINAL.csv     Cleaned output
graduate_placement.db                    Cleaned data loaded into SQLite
Captsone_project_Joan_Twinomugyisha.ipynb  Full analysis notebook (EDA, modeling, fairness audit)
app.py                                   Streamlit demo application
requirements.txt                         Python dependencies
graduate_model.pkl                       Trained Random Forest model
label_encoders.pkl                       Fitted label encoders for categorical inputs
feature_columns.pkl                      Feature column order expected by the model
historical_placement_data.pkl            Cleaned data used for the app's analytics tab
.github/workflows/etl.yml                Scheduled ETL pipeline (GitHub Actions, weekly cron)
```

## Architecture decisions

- **SQLite, not a cloud warehouse.** At 2,387 rows, Postgres/BigQuery/Snowflake
  would add infrastructure overhead with no performance benefit. SQLite ships
  as a single file — anyone can clone this repo and query the data with zero
  setup. This would change if Refactory scaled to 100,000+ records across many
  cohorts with concurrent writes from multiple staff.
- **Random Forest over Neural Network.** Both achieve comparable accuracy
  (~77% vs ~76%). Random Forest was selected for its feature importance
  scores, which make predictions explainable to non-technical placement
  staff — a black-box neural network prediction is harder to act on.
- **Fairness-by-design feature selection.** Nationality, Refugee Status, and
  Disability status are excluded from the model's input features entirely,
  and used only in a post-hoc fairness audit to check the model's predictions
  don't disproportionately affect these groups via other correlated features.
- **GitHub Actions weekly cron** re-runs the cleaning pipeline automatically
  via `clean_pipeline.py`, so the system doesn't require manual
  re-processing if new graduate records are added to the raw CSV.

## Running it yourself

```bash
pip install -r requirements.txt

# Run the cleaning pipeline standalone
python clean_pipeline.py

# Launch the interactive demo
streamlit run app.py
```

Or open `Captsone_project_Joan_Twinomugyisha.ipynb` in Google Colab / Jupyter
to see the full analysis: data cleaning, EDA, model training and evaluation,
fairness audit, and business recommendations.

## Key findings

- **Model performance:** Random Forest achieves 77% accuracy, 73% recall on
  at-risk graduates (the group most in need of proactive support).
- **Top predictors:** Program Name, Sponsorship Type, and Education Level
  drive employment predictions most — not demographic attributes.
- **Fairness audit:** No meaningful disparity in predicted employment rate by
  Refugee Status (58.1% vs 57.9%) or Nationality. Disability status shows a
  higher predicted rate for disabled graduates (66.7% vs 57.5%), though based
  on a small sample (n=30).

## Working demo

Live app: **https://graduate-employment-prediction-9a6xstqzbmk5xyejhnfqcz.streamlit.app/**

## Author

Joan Twinomugyisha — Refactory Academy
