import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
import joblib
import os  # Import the os module

def preprocess_data():
    # Your existing preprocess_data() function
    d1 = pd.read_csv('dataset/Bank Marketing Data Set.csv')
    
    # Data Cleaning & Imputation
    significant_cat_variables = ['education', 'job']
    for var in significant_cat_variables:
        d1[var + '_un'] = (d1[var] == 'unknown').astype(int)

    d1.loc[(d1['age'] > 60) & (d1['job'] == 'unknown'), 'job'] = 'retired'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'admin.'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'blue-collar'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'entrepreneur'), 'education'] = 'tertiary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'housemaid'), 'education'] = 'primary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'management'), 'education'] = 'tertiary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'retired'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'self-employed'), 'education'] = 'tertiary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'services'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'student'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'technician'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'unemployed'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'unknown'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'secondary') & (d1['job'] == 'unknown'), 'job'] = 'blue-collar'
    d1.loc[(d1['education'] == 'tertiary') & (d1['job'] == 'unknown'), 'job'] = 'blue-collar'
    d1.loc[(d1['education'] == 'primary') & (d1['job'] == 'unknown'), 'job'] = 'management'
    
    d1['pdays'] = d1['pdays'].replace(to_replace=-1, value=0)
    d1.rename(columns={'class': 'deposit', 'campain': 'campaign'}, inplace=True)
    d1['deposit'] = d1['deposit'].replace(to_replace=[1, 2], value=[0, 1])
    d1.drop(['education_un', 'job_un'], axis=1, inplace=True)
    
    num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']
    for col in num_cols:
        d1[col] = d1[col].apply(lambda x: np.cbrt(x) if x >= 0 else -np.cbrt(abs(x)))

    d2 = d1.copy()
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    d2 = pd.get_dummies(data=d2, columns=cat_cols, drop_first=True)
    
    cols_to_drop_pre_modeling = ['day', 'previous', 'job_retired', 'marital_single', 'education_secondary', 'default_yes', 'contact_telephone', 'poutcome_other', 'poutcome_unknown', 'month_may']
    final_features = [col for col in d2.columns if col not in cols_to_drop_pre_modeling + ['deposit']]
    
    X = d2[final_features]
    y = d2['deposit']

    sm = SMOTE(random_state=2)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=1)
    
    return X, y, X_resampled, y_resampled, X_train, X_test, y_train, y_test, final_features

# Run preprocessing
X_data, y_data, X_resampled, y_resampled, X_train, X_test, y_train, y_test, final_features = preprocess_data()

# Define and train models
models_to_train = {
    'Logistic Regression': LogisticRegression(solver='liblinear', max_iter=1000, random_state=1),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_leaf=18, random_state=0),
    'KNN Classifier': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'Random Forest': RandomForestClassifier(n_estimators=13, random_state=0),
    'Bagging Classifier': BaggingClassifier(n_estimators=8, random_state=0),
    'AdaBoost': AdaBoostClassifier(n_estimators=90, random_state=0),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=98, random_state=0),
    'Stacked Classifier': StackingClassifier(classifiers=[
        BaggingClassifier(n_estimators=8, random_state=0),
        RandomForestClassifier(n_estimators=13, random_state=0),
        AdaBoostClassifier(n_estimators=90, random_state=0)
    ], meta_classifier=KNeighborsClassifier(n_neighbors=5, weights='distance'))
}

# --- Create the models/ folder if it doesn't exist ---
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Train and save each model
print("Starting model training...")
for name, model in models_to_train.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    # Update the file path to include the 'models/' folder
    joblib.dump(model, os.path.join(model_dir, f'trained_model_{name.replace(" ", "_").lower()}.joblib'))
    print(f"Finished training and saving {name} to {model_dir}/.")
print("All models trained and saved successfully.")