# import libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


st.set_page_config(
    page_title="NASA Turbofan Playground",
    page_icon="🧊",
    initial_sidebar_state="expanded",
    
)

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://w.wallhaven.cc/full/yx/wallhaven-yxldyk.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


# Title with Rainbow Transition Effect and Neon Glow
html_code = """
<div class="title-container">
  <h1 class="neon-text">
    NASA Turbofan Playground
  </h1>
</div>

<style>
@keyframes rainbow-text-animation {
  0% { color: white; }
  16.67% { color: grey; }
  33.33% { color: grey; }
  50% { color: black; }
  66.67% { color: grey; }
  83.33% { color: white; }
  100% { color: white; }
}

.title-container {
  text-align: center;
  margin: 1em 0;
  padding-bottom: 10px;
  border-bottom: 4  px solid #fcdee9; /* Magenta underline */
}

.neon-text {
  font-family: Trebuchet MS , sans-serif;
  font-size: 4em;
  margin: 0;
  animation: rainbow-text-animation 5s infinite linear;
  text-shadow: 0 0 5px rgba(0, 0, 0, 0.8),
               0 0 10px rgba(0, 0, 0, 0.7),
               0 0 20px rgba(0, 0, 0, 0.6),
               0 0 40px rgba(0, 0, 0, 0.6),
               0 0 80px rgba(0, 0, 0, 0.6),
               0 0 90px rgba(0, 0, 0, 0.6),
               0 0 100px rgba(0, 0, 0, 0.6),
               0 0 150px rgba(0, 0, 0, 0.6);
}
</style>
"""

st.markdown(html_code, unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .success-message {
        font-family: Arial, sans-serif;
        font-size: 24px;
        color: green;
        text-align: left;
    }
    .wng_txt {
        font-family: Arial, sans-serif;
        font-size: 24px;
        color: yellow;
        text-align: left;
    }
    .unsuccess-message {
        font-family: Arial, sans-serif;
        font-size: 22px;
        color: red;
        text-align: left;
    }
    .prompt-message {
        font-family: Arial, sans-serif;
        font-size: 24px;
        color: #333;
        text-align: center;
    }
    .success-message2 {
        font-family: Arial, sans-serif;
        font-size: 18px;
        color: white;
        text-align: left;
    }
    .inf_txt {
        font-family: Arial, sans-serif;
        font-size: 28px;
        color: white;
        text-align: left;
    }
    .message-box {
        text-align: center;
        background-color: white;
        padding: 5px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        font-size: 24px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load your ML model
model = joblib.load('random_forest_model.joblib')

# Define the columns and their min/max values along with descriptions
columns = {
    'cycles': {'min': 1, 'max': 362, 'type': 'int', 'desc': 'Number of operational cycles the turbofan engine has undergone.'},
    'T24': {'min': 641.21, 'max': 644.53, 'type': 'float', 'desc': 'Total temperature at fan inlet (sensor measurement).'},
    'T30': {'min': 1571.04, 'max': 1616.91, 'type': 'float', 'desc': 'Total temperature at Low-Pressure Compressor (LPC) outlet (sensor measurement).'},
    'T50': {'min': 1382.25, 'max': 1441.49, 'type': 'float', 'desc': 'Total temperature at High-Pressure Turbine (HPT) outlet (sensor measurement).'},
    'P30': {'min': 549.85, 'max': 556.06, 'type': 'float', 'desc': 'Pressure at HPC outlet (sensor measurement).'},
    'Nf': {'min': 2387.90, 'max': 2388.56, 'type': 'float', 'desc': 'Physical fan speed (shaft rotational speed).'},
    'Nc': {'min': 9021.73, 'max': 9244.59, 'type': 'float', 'desc': 'Physical core speed (shaft rotational speed).'},
    'Ps30': {'min': 46.85, 'max': 48.53, 'type': 'float', 'desc': 'Static pressure at HPC outlet (sensor measurement).'},
    'phi': {'min': 518.69, 'max': 523.38, 'type': 'float', 'desc': 'Ratio of fuel flow to the engine core airflow (calculated parameter).'},
    'NRf': {'min': 2387.88, 'max': 2388.56, 'type': 'float', 'desc': 'Corrected fan speed (shaft rotational speed).'},
    'NRc': {'min': 8099.94, 'max': 8293.72, 'type': 'float', 'desc': 'Corrected core speed (shaft rotational speed).'},
    'BPR': {'min': 8.3249, 'max': 8.5848, 'type': 'float', 'desc': 'Bypass ratio (ratio of the mass of air bypassing the engine core to the mass of air passing through the core).'},
    'htBleed': {'min': 388.00, 'max': 400.00, 'type': 'float', 'desc': 'Bleed Enthalpy (energy loss due to air bled off for use in other aircraft systems).'},
    'W31': {'min': 38.14, 'max': 39.43, 'type': 'float', 'desc': 'High-pressure turbine cooling bleed flow (sensor measurement).'},
    'W32': {'min': 22.8942, 'max': 23.6184, 'type': 'float', 'desc': 'Low-pressure turbine cooling bleed flow (sensor measurement).'}
}

# Sidebar inputs
st.sidebar.title("NASA Turbofan Playground")
inputs = {}
for col, limits in columns.items():
    if limits['type'] == 'int':
        inputs[col] = st.sidebar.number_input(col, min_value=limits['min'], max_value=limits['max'], value=(limits['min'] + limits['max']) // 2, step=1, help=limits['desc'])
    else:
        inputs[col] = st.sidebar.number_input(col, min_value=limits['min'], max_value=limits['max'], value=(limits['min'] + limits['max']) / 2, help=limits['desc'])

# Main page
# st.title("Turbofan Engine RUL Prediction")
# st.markdown('<p class="message-box">Turbofan Engine RUL Prediction and Anomaly Detection</p>', unsafe_allow_html=True)
# st.markdown('<p class="success-message2">Provide the necessary inputs in the sidebar to predict the Remaining Useful Life (RUL) of the turbofan engine.</p>', unsafe_allow_html=True)
# st.write("Provide the necessary inputs in the sidebar to predict the Remaining Useful Life (RUL) of the turbofan engine.")

# Prepare the input for prediction
input_data = [inputs[col] for col in columns]
input_data = [input_data]  # Assuming the model expects a 2D array

inp_sensor_data = input_data[0][1:]
# Predict the RUL
predict_rul = st.sidebar.button("Predict RUL")
if not predict_rul:
    
    # Title and Description
  
    st.write("Welcome to the NASA Turbofan Playground. This tool leverages machine learning "
            "to predict Remaining Useful Life (RUL) and detect anomalies in turbofan engine sensor data.")
    st.divider()
    # About the Application
    st.header("About the Application")
    st.markdown("""
    This application provides two main features:
    - **Remaining Useful Life (RUL) Prediction:** Predicts when a turbofan engine is likely to fail based on its current sensor readings.
    - **Sensor Anomaly Detection:** Identifies anomalies in the sensor data to detect potential issues before they lead to failures.
    """)
    st.divider()
    # Key Features
    st.header("Key Features")
    st.markdown("""
    - **RUL Prediction:** Input current sensor readings manually or via file upload to predict RUL.
    - **Anomaly Detection:** Detect anomalies in sensor readings to monitor engine health.
    """)
    st.divider()
    # Supported Sensors
    st.header("Supported Sensors")
    st.markdown("""
    - **Temperature Sensors:** T24, T30, T50
    - **Pressure Sensors:** P30, Ps30
    - **Speed Sensors:** Nf, Nc, NRf, NRc
    - **Bleed Air Sensors:** BPR, htBleed
    - **Flow Sensors:** W31, W32
    - **Other Parameters:** phi
    """)
    st.divider()
    # Technology Stack
    st.header("Technology Stack")
    st.markdown("""
    - **Frontend:** Streamlit
    - **Backend:** Python
    - **Machine Learning Models:** Random Forest (RUL Prediction), Isolation Forest (Anomaly Detection)
    """)

if predict_rul:
    prediction = model.predict(input_data)
    predicted_rul = int(prediction[0])
    input_cycles = inputs['cycles']
    
    # Calculate percentage of RUL
    percentage_rul = input_cycles / predicted_rul
    st.divider()
    st.subheader(f"The predicted Remaining Useful Life (RUL) is: {predicted_rul} cycles")
    st.divider()
 
    # Determine engine health status
    if percentage_rul >=1:
        
        st.markdown('<p class="unsuccess-message">The RUL is less than Current Cycle, this may be due to wrong prediction or Sensor Anomalies.</p>', unsafe_allow_html=True)
        max_deviation = 0
        anomaly_sensor = None
        for col, limits in columns.items():
            deviation = min(abs(inputs[col] - limits['min']), abs(inputs[col] - limits['max']))
            if deviation > max_deviation:
                max_deviation = deviation
                anomaly_sensor = col
        st.markdown(f'<p class="unsuccess-message">Potential anomaly detected in sensor: {anomaly_sensor}.</p>', unsafe_allow_html=True)
        
    elif percentage_rul < 0.4:
        st.markdown('<p class="success-message">The Remaining Useful is more than 60%, \nEngine Health is Excellent.</p>', unsafe_allow_html=True)
        
    elif percentage_rul < 0.6:
        
        st.markdown('<p class="success-message">The Remaining Useful is more than 40%, \nEngine Health is Fine.</p>', unsafe_allow_html=True)
    elif percentage_rul < 0.8:
        
        st.markdown('<p class="wng_txt">Warning: The Remaining Useful is between 60 to 80%, \nEngine needs to be checked.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="unsuccess-message">Warning: The Remaining Useful is critical and less than 20%, \nEngine maintenance Required.</p>', unsafe_allow_html=True)
        
        # Identify the sensor responsible for the anomaly
        max_deviation = 0
        anomaly_sensor = None
        for col, limits in columns.items():
            deviation = min(abs(inputs[col] - limits['min']), abs(inputs[col] - limits['max']))
            if deviation > max_deviation:
                max_deviation = deviation
                anomaly_sensor = col
        
        st.markdown(f'<p class="unsuccess-message">Potential anomaly detected in sensor: {anomaly_sensor}.</p>', unsafe_allow_html=True)

   

# Load the pre-trained model
model_path = 'AnomalyDetection.joblib'
iso_forest = joblib.load(model_path)

# Define sensor columns based on the dataset (excluding 'RUL' since it's not used for anomaly detection)
sensor_columns = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']



# Button to run anomaly detection with the provided sensor data
if st.sidebar.button("Detect Anomaly of Sensors"):
    # Convert inputs to DataFrame
    input_data = pd.DataFrame([inp_sensor_data], columns=sensor_columns)
    
    # Predict anomaly
    anomaly_score = iso_forest.decision_function(input_data)
    anomaly_flag = iso_forest.predict(input_data)
    
    # Display results
    st.write(f"Anomaly Score: {anomaly_score[0]:.4f}")
    if anomaly_flag[0] == -1:
        st.write("Anomaly Detected!")
    else:
        st.write("No Anomaly Detected.")

    # Identify contributing sensors
    sensor_importance = {}
    for sensor in sensor_columns:
        input_temp = input_data.copy()
        input_temp[sensor] = 0  # Remove the influence of this sensor
        temp_scores = iso_forest.decision_function(input_temp)
        importance = np.mean(np.abs(anomaly_score - temp_scores))
        sensor_importance[sensor] = importance

    sorted_sensors = sorted(sensor_importance.items(), key=lambda x: x[1], reverse=True)
    
    st.write("Sensors contributing to anomalies in descending order of importance: ")
    for sensor, importance in sorted_sensors:
        st.write(f"{sensor}: {importance:.4f}")