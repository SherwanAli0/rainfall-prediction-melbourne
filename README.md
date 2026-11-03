  Project Overview : Rainfall Prediction for Melbourne Area

A machine learning project that predicts whether it will rain on the day after an observation, using historical weather data from Australia. This is coursework for the **IBM AI Engineering Certificate, Machine Learning course** (IBM AI Engineering Professional Certificate on Coursera), completed by Sherwan Ali. The course supplied the dataset and the task outline; the implementation in this repository is Sherwan Ali's own work. The dataset is hosted on IBM's course storage (see the URL under "Dataset").

This project builds a classification model to predict rainfall using the Australian weather dataset. The model focuses on the Melbourne metropolitan area and uses weather features such as temperature, humidity, pressure, and wind to make predictions.

 Dataset:

- Source: Australian weather dataset (`weatherAUS`, described as originating from the R "rattle" package), downloaded at run time from the IBM course storage URL hard-coded in `rainfall_prediction.py`:
  `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv`
  The data is not stored in this repository. The script needs internet access; if the URL is unreachable it stops with the error raised by `pandas.read_csv`.
- Focus Area: Melbourne, Melbourne Airport, and Watsonia
- Features: Temperature, humidity, pressure, wind speed/direction, cloud cover, rainfall, etc., plus a season derived from the date

What is being predicted (read this before interpreting any output):

- Each row holds the weather observed on its `Date`. In the source CSV, `RainToday` is the rain flag for that date and `RainTomorrow` is the rain flag for the following day. Checked against the downloaded CSV on 2026-11-03: for consecutive dates at the same location, `RainTomorrow` equals the next day's `RainToday` in all 142,017 pairs, and `RainToday` agrees with `Rainfall >= 1` mm on 98.8% of rows.
- The script renames `RainToday` to `RainYesterday` (used as a feature) and `RainTomorrow` to `RainToday` (the target). This renaming follows the course exercise.
- So the **target column named `RainToday` in the code and outputs is the original `RainTomorrow`**: rain (Yes/No) on the day after the row's date, predicted from that date's observations. It is not a same-day nowcast.

Technologies Used:

- Python 3.x (see "Environment checked" below)
- Pandas - Data manipulation and analysis
- Scikit-learn - Machine learning pipeline and algorithms
- Matplotlib - Visualization
- Random Forest Classifier - Main algorithm used

Methodology:

1. Data Preprocessing:
   - Handled missing values by removing incomplete records
   - Created seasonal features from dates
   - Focused on Melbourne area locations for consistency

2. Feature Engineering:
   - Automatic detection of numerical vs categorical features
   - Standard scaling for numerical features
   - One-hot encoding for categorical features

3. Model Development:
   - Built a complete scikit-learn pipeline
   - Used Random Forest Classifier
   - Hyperparameter tuning with GridSearchCV
   - 5-fold cross-validation (`cv=5`, which for a classifier means stratified, unshuffled 5-fold on the training set)
   - Stratified random 80/20 train/test split of rows (`random_state=42`); the split is not chronological

4. Model Evaluation:
   - Classification report (precision, recall, F1-score)
   - Confusion matrix visualization
   - Feature importance analysis

 Re-run on 2026-11-03 (measured with the repaired script, `python rainfall_prediction.py`, Python 3.11.9, pandas 3.1.0, scikit-learn 1.9.1, matplotlib 3.11.2, numpy 2.4.6; about 3 minutes on the author's Windows PC):

- Dataset: the CSV has 145,460 rows and 23 columns with no leakage-type column (no `RISK_MM`); 56,420 rows remain after `dropna()`, and 7,557 rows for the three Melbourne-area locations (5,766 No, 1,791 Yes, 76.3% / 23.7%)
- Best parameters: `max_depth` 20, `min_samples_split` 2, `n_estimators` 100
- Best cross-validation accuracy: 0.8506
- Test accuracy: 0.8446 (1,512 test rows)
- Confusion matrix (rows = true No/Yes, columns = predicted): [[1094, 60], [175, 183]]
- No rain: precision 0.86, recall 0.95, F1 0.90; Rain: precision 0.75, recall 0.51, F1 0.61; weighted F1 0.83
- Top 5 features: Humidity3pm 0.121, Pressure3pm 0.091, Pressure9am 0.086, Sunshine 0.082, WindGustSpeed 0.051

These match the historical results below (same parameters, same confusion matrix, same top-feature ranking), even though the library versions differ from the original run. `images/` was regenerated from this run.

 Historical results (from the original run, kept for reference):

- Test Accuracy: 84% (0.84)
- Cross-validation Score: 85% (0.85)
- Dataset Size: 7,557 samples after cleaning (Melbourne area only)
- Class Distribution: ~76% No Rain, ~24% Rain (imbalanced dataset)
- Best Model Parameters: 
  - n_estimators: 100
  - max_depth: 20
  - min_samples_split: 2

Model Performance Breakdown:
- No Rain Prediction: 86% precision, 95% recall
- Rain Prediction: 75% precision, 51% recall
- Overall F1-Score: 0.83 (weighted average)

Top 5 Most Important Features (original run; the 2026-11-03 re-run gives Pressure9am 0.086, i.e. 0.09 when rounded):
1. Humidity3pm (0.12) - Afternoon humidity levels
2. Pressure3pm (0.09) - Afternoon atmospheric pressure  
3. Pressure9am (0.08) - Morning atmospheric pressure
4. Sunshine (0.08) - Daily sunshine hours
5. WindGustSpeed (0.05) - Maximum wind gust speed

The model beats the baseline accuracy of about 76% (always predicting "no rain") and does well on the more common no-rain days, with moderate recall on rainy days.

How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/SherwanAli0/rainfall-prediction-melbourne.git
   cd rainfall-prediction-melbourne
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis from the repository directory (requires internet access to download the CSV):
   ```bash
   python rainfall_prediction.py
   ```

4. The script saves `images/confusion_matrix.png` and `images/feature_importance.png` (overwriting existing files) and also opens each figure in a window; close the first window to continue. The committed PNG files were regenerated by the 2026-11-03 run.

Environment checked

- Full run on the real dataset: Python 3.11.9, pandas 3.1.0, scikit-learn 1.9.1, matplotlib 3.11.2, numpy 2.4.6 (results above).
- The offline tests were also run with pandas 2.3.3; the full real-data run was not repeated on pandas 2.x or other Python versions.

Checks without the dataset

```bash
pip install pytest
pytest tests
```

The tests run the script on a tiny synthetic dataset (nothing is downloaded and the committed images are not touched). They check the label renaming, that the target is not among the features, that feature names match importances, that `cv=5` is used, and that both figures are saved.

 Project Structure:

```
rainfall-prediction-melbourne/
├── README.md                    # Project documentation
├── LICENSE                      # MIT license
├── rainfall_prediction.py       # Main analysis script
├── requirements.txt             # Python dependencies
├── tests/                       # Offline tests using a synthetic dataset
└── images/                      # Saved figures
    ├── confusion_matrix.png     # Confusion matrix on the test set
    └── feature_importance.png   # Feature importance plot
```

Limitations:

- The train/test split and cross-validation are random over rows, not chronological. Consecutive days from the same locations appear in both training and test data, so scores are likely optimistic compared with genuine forecasting.
- Features are same-date observations (including the date's own rain flag and rainfall amount), and the target is the next day's rain.
- Only three Melbourne-area locations are used, and rows with any missing value are dropped.
- This is a coursework model, not validated for operational forecasting; any real-world use (weather services, agriculture, event planning, transport) would need proper validation.

 Future Improvements:

-  Handle class imbalance with techniques like SMOTE
-  Try additional algorithms (XGBoost, Neural Networks)
-  Include more locations for broader applicability
-  Add time series features for temporal patterns
-  Deploy model as a web application

Certification

This project was completed as part of the **IBM AI Engineering Professional Certificate** program on Coursera, specifically the "Machine Learning with Python" course.

License

Released under the MIT License; see `LICENSE`.

Contact

Feel free to reach out if you have questions 
