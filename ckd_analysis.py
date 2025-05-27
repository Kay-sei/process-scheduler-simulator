import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv(r'C:\Users\prave\OneDrive\Desktop\papers\ckd-dataset-v2.csv')

# Drop the first two rows since they contain metadata
df = df.drop([0, 1])

# Drop rows with missing values
df = df.dropna()

# Encode categorical variables(Categorical variables contain label values rather than numeric values.Stages of CKD )
df = pd.get_dummies(df)

# Extract stage columns
stage_columns = [col for col in df.columns if 'stage_' in col]

# Create a single target column 'stage'
df['stage'] = df[stage_columns].idxmax(axis=1)

# Splitting features and target variable
X = df.drop(stage_columns + ['stage'], axis=1) #X(Features):All columns except the stage column.For training the model.
y = df['stage']                                #The'stage' column.Model will learn to predict.

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)#100 decision trees
rf_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluating the model
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
