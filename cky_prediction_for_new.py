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

#Standardize features to have zero mean and unit variance, which helps improve the performance           
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)#Computes the mean and standarddeviation of the trainingdata and then scales it.
X_test = scaler.transform(X_test)#Scales the test data using the mean and standard deviation from the training data.

# Training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)#100 decision trees
rf_model.fit(X_train, y_train)




# Assuming the new data is the last row in the dataset
new_data = df.iloc[-1, :-1]  # Extract the last row excluding the target column

# Ensure the new data has the same columns as the training data
new_data = new_data.reindex(X.columns, fill_value=0)

# Scale the new data
new_data_scaled = scaler.transform([new_data])

# Make prediction for the new data
new_prediction = rf_model.predict(new_data_scaled)

# Print the predicted stage
print(f"The predicted stage for the new data is: {new_prediction[0]}")