
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title= "Medical Insurance Cost Prediction")
st.title("Medical Insurance Cost Prediction App")

# --- Paths ---
model_path = "best_model.joblib"
metrics_path = "metrics.json"


# --- Load model (ONCE) ---
try:
    loaded_model = joblib.load(model_path)  # model_path must be a string path
    st.success(f"Model loaded successfully from: {model_path}")
except Exception as e:
    st.error(f"Failed to load model from '{model_path}': {e}")
    st.stop()

# --- Load metrics if present ---
metrics = None
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        st.info("Loaded metrics.json — will use reported RMSE/MAE to show approximate error margins.")
    except Exception as e:
        st.warning(f"Could not read metrics.json: {e}")
else:
    st.info("No metrics.json found. The app can still show a fallback error margin if desired.")



df = pd.read_csv("medical_insurance.csv")
df

st.subheader("Exploratory Data Analysis (EDA)")
# Plot 1: Age vs. Charges
st.write("#### Age vs. Charges")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=df, ax=ax)
st.pyplot(fig)
st.write("This scatter plot shows the relationship between age and medical insurance charges. Generally, charges tend to increase with age.")

# Plot 2: BMI vs. Charges
st.write("#### BMI vs. Charges")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', data=df, ax=ax)
st.pyplot(fig)
st.write("This scatter plot illustrates the relationship between Body Mass Index (BMI) and medical insurance charges. Higher BMI often correlates with higher charges.")

# Plot 3: Smoker vs. Charges
st.write("#### Smoking Status vs. Charges")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
st.pyplot(fig)
st.write("This box plot compares the distribution of medical insurance charges for smokers (1) and non-smokers (0). Smokers typically have significantly higher charges.")

# Plot 4: Region vs. Charges
st.write("#### Region vs. Charges")
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x='region', y='charges', data=df, ax=ax)
st.pyplot(fig)
st.write("This box plot shows how medical insurance charges vary across different regions.")


st.subheader("Enter Your Information:")

age = st.number_input("Age", min_value=18, max_value=120, value=30)
sex = st.selectbox("Sex", ['female', 'male'])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, value=0)
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])


#show confidence / error margins
show_CI = st.checkbox("Show approximate 95% confidence interval / error margin (if metrics available)",value=True)
use_fallback_percent = st.slider("If metrics not available, use fallback relative error (%)",min_value=5,max_value=100,value=20)
# Create a DataFrame from user input
sample = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

# Apply binary mapping for 'sex'
sample['sex'] = sample['sex'].map({"female": 1, "male": 0})

# Apply binary mapping for 'smoker'
sample['smoker'] = sample['smoker'].map({"yes": 1, "no": 0})

# Apply one-hot encoding for 'region'
sample = pd.get_dummies(sample, columns=['region'], drop_first=True)

# Define the bmi_category function
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if 18.5 <= bmi < 25:
        return "normal"
    elif 25 <= bmi < 30:
        return "overweight"
    else:
        return "obese"

# Apply bmi_category function
sample["bmi_category"] = sample["bmi"].apply(bmi_category)

# Apply Label Encoding to 'bmi_category'
lb = LabelEncoder()
sample['bmi_category'] = lb.fit_transform(sample['bmi_category'])

# Create engineered features
sample['smoker_age'] = sample['smoker'] * sample['age']
sample['smoker_bmi'] = sample['smoker'] * sample['bmi']

# create obese column for eda
df['obese'] = df['bmi']>30

# Reindex to match training data columns
# We need to make sure x_train is available in this scope or load its columns
# For this example, assuming x_train.columns is accessible or can be reconstructed.
# In a real Streamlit app, you might save the column order or load it.
# For demonstration, let's assume we have x_train_columns available
# (in a real app, you'd likely load this or pass it from training)
# As a workaround for this environment, let's try to reconstruct the expected columns
# based on the training steps: age, sex, bmi, children, smoker, region_northwest,
# region_southeast, region_southwest, bmi_category, smoker_age, smoker_bmi
expected_columns = ['age', 'sex', 'bmi', 'children', 'smoker',
                    'region_northwest', 'region_southeast', 'region_southwest',
                    'bmi_category', 'smoker_age', 'smoker_bmi','obese']

sample = sample.reindex(columns=expected_columns, fill_value=0)

st.write("Preprocessed Input Data:")
st.dataframe(sample)

# Make prediction
if st.button("Predict Insurance Cost"):
    try:
        prediction = loaded_model.predict(sample)[0]
    except Exception as e:
        st.error(f"Prediction failed :{e}")
        st.stop()

    # Determine an uncertainty estimate  
    uncertainty = None
    method_used = None

    if show_CI and metrics is not None:
        if "rmse" in metrics:
            uncertainty = float(metrics["rmse"])
            method_used = "rmse"
        elif "RMSE" in metrics:
            uncertainty = float(metrics["RMSE"])
            method_used = "RMSE"
        elif "MAE" in metrics:
            uncertainty = float(metrics["MAE"])  
            method_used = "MAE"      
        elif "mae" in metrics:
            uncertainty = float(metrics["mae"])
            method_used= "mae"

    if show_CI and metrics is None:
        uncertainty = abs(prediction)*(use_fallback_percent /100.0)
        method_used = f"fallback{use_fallback_percent}pct"   

    st.subheader("Predicted Medical Insurance Cost:")
    st.write(f"${prediction:,.2f}")

    if show_CI and uncertainty is not None:
        # For a normal-approx 95% interval use 1.96 * sigma (if metric is RMSE or similar)
        if method_used in ["rmse","RMSE"]:
            sigma = uncertainty
            lower = prediction-1.96*(sigma)
            upper = prediction+1.96(sigma)
            st.info(f"Using {method_used} from metrics json as sigma(approax). 95% CI ")
            st.write(f"lower: ${max(0,lower):, .2f}-Upper: ${upper:,.2f}")
        elif method_used in ["MAE","mae"]:
            sigma = uncertainty
            lower = prediction-1.96*(sigma)
            upper = prediction+1.96*(sigma)
            st.info(f"Using {method_used} from metrics.json as approximate error. Approx. 95% interval:")
            st.write(f"Lower: ${max(0, lower):,.2f} — Upper: ${upper:,.2f}")
        else:
            lower = prediction - uncertainty
            upper = prediction + uncertainty
            st.info(f"Using fallback relative error ({use_fallback_percent}%). This is a heuristic, not a statistically derived CI.")
            st.write(f"Lower: ${max(0, lower):,.2f} — Upper: ${upper:,.2f}")    
    


    # Show a brief explanation so users understand the interval
    st.markdown("---")
    st.write("**Note:** The interval shown is an approximate error margin. The app uses a saved training metric (if available) such as RMSE/MAE to produce a rough 95% interval. If no metric is available, a user-selected fallback percentage is used. For statistically rigorous prediction intervals you would need access to residuals/variance information from the training procedure or use probabilistic models (e.g., Bayesian models) or techniques like conformal prediction or bootstrapping saved at training time.")


    # Optionally show the metric used
    if metrics is not None and method_used is not None:
       st.write(f"Metric used for uncertainty: {method_used} = {uncertainty}")







