  Project Overview : Rainfall Prediction for Melbourne Area

A machine learning project that predicts whether it will rain on the day after an observation, using historical weather data from Australia. This is **coursework**: it was completed as part of the **IBM AI Engineering Professional Certificate** on Coursera ("Machine Learning with Python" course). The dataset is hosted on IBM's course storage (see the URL under "Dataset").

This project builds a classification model to predict rainfall using the Australian weather dataset. The model focuses on the Melbourne metropolitan area and uses weather features such as temperature, humidity, pressure, and wind to make predictions.

 Dataset:

- Source: Australian weather dataset (`weatherAUS`, described as originating from the R "rattle" package), downloaded at run time from the IBM course storage URL hard-coded in `rainfall_prediction.py`:
  `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv`
  The data is not stored in this repository. The script needs internet access; if the URL is unreachable it stops with the error raised by `pandas.read_csv`.
- Focus Area: Melbourne, Melbourne Airport, and Watsonia
- Features: Temperature, humidity, pressure, wind speed/direction, cloud cover, rainfall, etc., plus a season derived from the date

What is being predicted (read this before interpreting any output):

- Each row holds the weather observed on its `Date`. In the source CSV, `RainToday` is the rain flag for that date and `RainTomorrow` is the rain flag for the following day (based on the column names; not re-checked against the CSV in the latest pass).
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

 Historical results (from the original run; not re-run in the latest repair pass because that requires downloading the dataset, so these numbers are unverified against a fresh run):

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

The test-set figures above are consistent with the confusion matrix saved in `images/confusion_matrix.png` (1,512 test rows), which was produced by the original run.

Top 5 Most Important Features (original run):
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

4. The script saves `images/confusion_matrix.png` and `images/feature_importance.png` (overwriting existing files) and also opens each figure in a window; close the first window to continue. The PNG files currently committed come from the original run (the feature-importance title text there differs slightly from the current script).

Environment checked

- Python 3.11.9 with pandas 2.3.3 and pandas 3.1.0, scikit-learn 1.9.1, matplotlib 3.11.2, numpy 2.4.6.
- These versions were checked only on a small synthetic stand-in dataset (the real dataset was not downloaded), so the pipeline and figure saving are verified but the metrics above are not.

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
