# import libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="NASA Turbofan Playground",
    page_icon="🧊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://w.wallhaven.cc/full/q6/wallhaven-q6vpxr.png");
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
  0% { color: red; }
  16.67% { color: orange; }
  33.33% { color: yellow; }
  50% { color: green; }
  66.67% { color: blue; }
  83.33% { color: indigo; }
  100% { color: violet; }
}

.title-container {
  text-align: center;
  margin: 1em 0;
  padding-bottom: 10px;
  border-bottom: 4  px solid #fcdee9; /* Magenta underline */
}

.neon-text {
  font-family: Times New Roman, sans-serif;
  font-size: 4em;
  margin: 0;
  animation: rainbow-text-animation 5s infinite linear;
  text-shadow: 0 0 5px rgba(255, 255, 255, 0.8),
               0 0 10px rgba(255, 255, 255, 0.7),
               0 0 20px rgba(255, 255, 255, 0.6),
               0 0 40px rgba(255, 0, 255, 0.6),
               0 0 80px rgba(255, 0, 255, 0.6),
               0 0 90px rgba(255, 0, 255, 0.6),
               0 0 100px rgba(255, 0, 255, 0.6),
               0 0 150px rgba(255, 0, 255, 0.6);
}
</style>
"""

st.markdown(html_code, unsafe_allow_html=True)
st.divider()

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
st.markdown('<p class="message-box">Turbofan Engine RUL Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="success-message2">Provide the necessary inputs in the sidebar to predict the Remaining Useful Life (RUL) of the turbofan engine.</p>', unsafe_allow_html=True)
# st.write("Provide the necessary inputs in the sidebar to predict the Remaining Useful Life (RUL) of the turbofan engine.")

# Prepare the input for prediction
input_data = [inputs[col] for col in columns]
input_data = [input_data]  # Assuming the model expects a 2D array

# Predict the RUL
if st.sidebar.button("Predict RUL"):
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

   