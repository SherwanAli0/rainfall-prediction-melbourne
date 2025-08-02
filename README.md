  Project Overview:Rainfall Prediction for Melbourne Area

A machine learning project that predicts whether it will rain tomorrow based on historical weather data from Australia. This project was completed as part of the **IBM AI Engineering Certificate Program** on Coursera.


This project builds a classification model to predict rainfall using the Australian weather dataset. The model focuses on the Melbourne metropolitan area and uses various weather features like temperature, humidity, pressure, and wind patterns to make predictions.

 Dataset:

- Source: Australian weather dataset from the R "rattle" package
- Focus Area: Melbourne, Melbourne Airport, and Watsonia
- Target Variable: RainToday (Yes/No)
- Features: Temperature, humidity, pressure, wind speed/direction, cloud cover, etc.

Technologies Used:

- Python 3.x
- Pandas - Data manipulation and analysis
- Scikit-learn - Machine learning pipeline and algorithms
- Matplotlib & Seaborn - Data visualization
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
   - 5-fold cross-validation for robust evaluation

4. Model Evaluation:
   - Classification report (precision, recall, F1-score)
   - Confusion matrix visualization
   - Feature importance analysis

 Key Results:

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

Top 5 Most Important Features:
1. Humidity3pm (0.12) - Afternoon humidity levels
2. Pressure3pm (0.09) - Afternoon atmospheric pressure  
3. Pressure9am (0.08) - Morning atmospheric pressure
4. Sunshine (0.08) - Daily sunshine hours
5. WindGustSpeed (0.05) - Maximum wind gust speed

The model successfully beats the baseline accuracy of 76% (always predicting "no rain") and shows strong performance in identifying non-rainy days, with moderate success in detecting rainy days.

How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/SherwanAli0./rainfall-prediction-melbourne.git
   cd rainfall-prediction-melbourne
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis:
   ```bash
   python rainfall_prediction.py
   ```

4. View the generated plots in the `images/` folder

 Project Structure:

```
rainfall-prediction-melbourne/
├── README.md                    # Project documentation
├── rainfall_prediction.py      # Main analysis script
├── requirements.txt            # Python dependencies
└── images/                     # Generated visualizations
    ├── confusion_matrix.png    # Model performance visualization
    └── feature_importance.png  # Feature importance plot
```

Business Impact:

This model could be used by:
- Weather services for improved rainfall forecasting
- Agricultural planning for irrigation and crop management
- Event planning for outdoor activities
- Transportation for weather-related disruptions

 Future Improvements:

-  Handle class imbalance with techniques like SMOTE
-  Try additional algorithms (XGBoost, Neural Networks)
-  Include more locations for broader applicability
-  Add time series features for temporal patterns
-  Deploy model as a web application

 Learning Outcomes:

This project demonstrates proficiency in:
- End-to-end machine learning pipeline development
- Data preprocessing and feature engineering
- Model evaluation and interpretation
- Professional code organization and documentation
- Handling real-world, imbalanced datasets

Certification

This project was completed as part of the **IBM AI Engineering Professional Certificate** program on Coursera, specifically the "Machine Learning with Python" course.

Contact

Feel free to reach out if you have questions about this project or want to discuss machine learning opportunities!

---

This project showcases practical machine learning skills applied to a real-world weather prediction problem.
