import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv(r'C:\Users\prave\OneDrive\Desktop\papers\ckd-dataset-v2.csv')

# Drop the first two rows since they contain metadata
df = df.drop([0, 1])

# Drop rows with missing values
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Extract stage columns
stage_columns = [col for col in df.columns if 'stage_' in col]

# Create a single target column 'stage'
df['stage'] = df[stage_columns].idxmax(axis=1)

# Splitting features and target variable
X = df.drop(stage_columns + ['stage'], axis=1)
y = df['stage']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the model with the best parameters
optimized_rf_model = grid_search.best_estimator_

# Making predictions on the test set
y_pred = optimized_rf_model.predict(X_test)

# Evaluating the model
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
