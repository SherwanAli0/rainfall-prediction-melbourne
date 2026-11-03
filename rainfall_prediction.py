from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Figures are saved next to this script, in images/.
IMAGES_DIR = Path(__file__).resolve().parent / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

# The CSV is downloaded from this URL every time the script runs (internet access required).
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)

df = df.dropna()
df.info()

# Label renaming (kept from the course exercise). Each row holds the weather
# observed on its Date. In the source CSV:
#   original 'RainToday'    = rain flag for the row's own date
#   original 'RainTomorrow' = rain flag for the day after the row's date
# After the rename below:
#   'RainYesterday' (feature) = original 'RainToday'   (rain flag of the row's date)
#   'RainToday'     (target)  = original 'RainTomorrow' (rain on the day AFTER the row's date)
# So the model predicts next-day rain from same-date observations; the target
# column is called 'RainToday' only because of this renaming.
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df.info()

def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

df['Date'] = pd.to_datetime(df['Date'])


df['Season'] = df['Date'].apply(date_to_season)

df = df.drop(columns=['Date'])

# Target = renamed column 'RainToday' (original 'RainTomorrow'); it is excluded from the features.
X = df.drop(columns='RainToday')
y = df['RainToday']

print(y.value_counts())

# Random stratified 80/20 split of rows (not chronological).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])


categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# cv=5 with a classifier means scikit-learn uses StratifiedKFold(n_splits=5) without shuffling.
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))


y_pred = grid_search.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig(IMAGES_DIR / 'confusion_matrix.png', bbox_inches='tight')
plt.show()

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_


feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20
top_features = importance_df.head(N)


plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title(f'Top {N} Features for Predicting Next-Day Rain (original RainTomorrow)')
plt.xlabel('Importance Score')
plt.savefig(IMAGES_DIR / 'feature_importance.png', bbox_inches='tight')
plt.show()
