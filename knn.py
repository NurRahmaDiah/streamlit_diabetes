# Import library
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score , train_test_split , RandomizedSearchCV , GridSearchCV
from sklearn.metrics import classification_report
from scipy import stats
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, KFold

import pickle
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('diabetes.csv')

# Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar
st.sidebar.header('Input Pengguna')
pregnancies = st.sidebar.slider('Jumlah Kehamilan', 0, 17, 3)
glucose = st.sidebar.slider('Kadar Glukosa', 0, 199, 117)
blood_pressure = st.sidebar.slider('Tekanan Darah', 0, 122, 72)
skin_thickness = st.sidebar.slider('Ketebalan Kulit', 0, 99, 23)
insulin = st.sidebar.slider('Insulin', 0, 846, 30)
bmi = st.sidebar.slider('Indeks Massa Tubuh (BMI)', 0.0, 67.1, 32.0)
diabetes_pedigree_function = st.sidebar.slider('Fungsi Pedigree Diabetes', 0.078, 2.42, 0.3725)
age = st.sidebar.slider('Usia', 21, 81, 29)

# Create user input DataFrame
user_input = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree_function],
    'Age': [age]
})

# Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predictions
user_prediction = knn_model.predict(scaler.transform(user_input))

# Display result
st.title('Prediksi Diabetes Menggunakan KNN')
st.write('Hasil Prediksi:', 'Diabetes' if user_prediction[0] == 1 else 'Tidak Diabetes')
