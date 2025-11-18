
import streamlit as st
import pandas as pd
import joblib # Often used for saving/loading models/scalers
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# --- Streamlit App Setup ---
st.title("Heart Disease Prediction")
st.write("Enter patient information to predict the likelihood of heart disease.")

# --- Simulate loading of pre-trained model and scaler ---
# In a real application, these objects (model, scaler) and the `X_train_columns` list
# would be saved to files (e.g., using joblib) after training and then loaded here.
# For the purpose of this exercise, we will re-create them based on the previous notebook steps
# to ensure the app is self-contained and runnable.

# 1. Load original data
# Assume '/content/heart_disease_uci.csv' is available as per the environment.
original_df = pd.read_csv('heart_disease_uci.csv')

# 2. Prepare target and drop columns ('id', 'dataset', 'num')
df_temp_for_processing = original_df.copy()
df_temp_for_processing['target'] = (df_temp_for_processing['num'] > 0).astype(int)
df_temp_for_processing = df_temp_for_processing.drop(['id', 'dataset', 'num'], axis=1)

# 3. Handle missing values (Numerical with median, Categorical with mode)
# Define columns based on prior analysis
numerical_cols_for_processing = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_cols_for_processing = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

for col in numerical_cols_for_processing:
    if df_temp_for_processing[col].isnull().any():
        median_val = df_temp_for_processing[col].median()
        df_temp_for_processing[col] = df_temp_for_processing[col].fillna(median_val)

for col in categorical_cols_for_processing:
    if df_temp_for_processing[col].isnull().any():
        mode_val = df_temp_for_processing[col].mode()[0]
        df_temp_for_processing[col] = df_temp_for_processing[col].fillna(mode_val).infer_objects(copy=False)

# 4. Fit StandardScaler on numerical features *before* one-hot encoding for the entire dataset
scaler = StandardScaler()
scaler.fit(df_temp_for_processing[numerical_cols_for_processing])

# 5. Apply one-hot encoding to categorical features for the full processed dataframe
df_processed_final = pd.get_dummies(df_temp_for_processing, columns=categorical_cols_for_processing, drop_first=True)

# 6. Separate features (X) and target (y) for model training
X_full_for_training = df_processed_final.drop('target', axis=1)
y_full_for_training = df_processed_final['target']

# Store the column names for alignment later
X_train_columns = X_full_for_training.columns.tolist()

# 7. Train the RandomForestClassifier model
# In a real app, this would be `model = joblib.load('random_forest_model.pkl')`
model = RandomForestClassifier(random_state=42)
model.fit(X_full_for_training, y_full_for_training)

# --- End of Simulation for Loaded Objects ---

# Define the numerical and categorical columns that were used in feature engineering
# These lists are used for processing user input consistently with training data
numerical_features_app = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_features_app = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# --- Streamlit Input Fields ---
with st.sidebar:
    st.header("Patient Data Input")

    age = st.slider("Age", 20, 80, 50)
    sex_input = st.radio("Sex", ['Male', 'Female'])
    cp_input = st.selectbox("Chest Pain Type (cp)", ['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'])
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
    chol = st.slider("Serum Cholestoral (chol)", 100, 600, 200)
    fbs_input = st.checkbox("Fasting Blood Sugar > 120 mg/dl (fbs)", False)
    restecg_input = st.selectbox("Resting Electrocardiographic Results (restecg)", ['normal', 'st-t abnormality', 'lv hypertrophy'])
    thalch = st.slider("Maximum Heart Rate Achieved (thalch)", 70, 210, 150)
    exang_input = st.checkbox("Exercise Induced Angina (exang)", False)
    oldpeak = st.slider("ST depression induced by exercise relative to rest (oldpeak)", 0.0, 6.0, 1.0, 0.1)
    slope_input = st.selectbox("The slope of the peak exercise ST segment (slope)", ['upsloping', 'flat', 'downsloping'])
    ca = st.slider("Number of major vessels (0-3) colored by flourosopy (ca)", 0, 3, 0)
    thal_input = st.selectbox("Thal", ['normal', 'fixed defect', 'reversable defect'])

    predict_button = st.button("Predict Heart Disease")

# --- Prediction Logic ---
if predict_button:
    # 1. Create a Pandas DataFrame from the user's input values.
    user_input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex_input],
        'cp': [cp_input],
        'trestbps': [float(trestbps)], # Ensure type consistency
        'chol': [float(chol)],         # Ensure type consistency
        'fbs': [fbs_input],
        'restecg': [restecg_input],
        'thalch': [float(thalch)],     # Ensure type consistency
        'exang': [exang_input],
        'oldpeak': [float(oldpeak)],   # Ensure type consistency
        'slope': [slope_input],
        'ca': [float(ca)],             # Ensure type consistency
        'thal': [thal_input]
    })

    st.write("### User Input Data:")
    st.dataframe(user_input_df)

    # 2. Apply one-hot encoding to the categorical features in the user input DataFrame.
    user_input_encoded = pd.get_dummies(user_input_df, columns=categorical_features_app, drop_first=True)

    # 3. Align the columns of the one-hot encoded user input DataFrame with the columns of X_train_columns.
    # This ensures that the user input DataFrame has the same columns in the same order as the training data.
    user_input_aligned = user_input_encoded.reindex(columns=X_train_columns, fill_value=0)

    # 4. & 5. Scale the numerical features in the aligned user input DataFrame.
    user_input_aligned[numerical_features_app] = scaler.transform(user_input_aligned[numerical_features_app])

    st.write("### Preprocessed User Input for Prediction:")
    st.dataframe(user_input_aligned)

    # 6. Use the loaded `model` to make a prediction.
    prediction = model.predict(user_input_aligned)
    prediction_proba = model.predict_proba(user_input_aligned)[:, 1] # Probability of heart disease (class 1)

    # 7. Display the prediction result.
    st.write("---")
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"The model predicts **Heart Disease** with a probability of {prediction_proba[0]:.2f}")
    else:
        st.success(f"The model predicts **No Heart Disease** with a probability of {1 - prediction_proba[0]:.2f}")
