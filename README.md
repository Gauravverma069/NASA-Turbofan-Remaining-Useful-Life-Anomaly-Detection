# NASA-Turbofan-Remaing-Useful-Life-Anomaly-Detection
![image](https://github.com/user-attachments/assets/98896374-9e42-4662-8e49-67d41a5f3315)

## Project Overview
This project involves developing a machine learning model to predict the Remaining Useful Life (RUL) of NASA turbofan engines and detect anomalies in their performance. The model uses sensor data from turbofan engines, including measurements such as cycles, temperatures, pressures, speeds, and flow rates. The primary goal is to enhance predictive maintenance by accurately estimating the time remaining before engine failure, thereby preventing unexpected breakdowns and optimizing maintenance schedules. Additionally, the project aims to identify any abnormal patterns in the sensor data that could indicate potential issues, ensuring the reliability and safety of the turbofan engines. The machine learning model is implemented in a Streamlit app, providing a user-friendly interface for predictions and anomaly detection.

Click to go on App-> [NASA Turbofan RUL Detection](https://huggingface.co/spaces/Gaurav069/NASA_Turbofan_RUL_and_Anomaly_Prediction_Playground)

## Key Features
- **RUL Prediction**: Predict the Remaining Useful Life of turbofan engines using a Random Forest model.
- **Anomaly Detection**: Identify unusual patterns or failures in the engine data.
- **Comprehensive Dataset**: Utilizes the NASA Turbofan Engine Degradation Simulation Dataset.
- **Interactive Streamlit App**: Provides an easy-to-use interface for making predictions and visualizing data.
- **Modular Code Structure**: Organized code for easy understanding and modifications.
- **Extensive Documentation**: Detailed instructions and explanations to guide users through installation, usage, and contributions.
## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#Key-Features)
- [Dataset](#dataset)
- [Model](#Model)
- [Contributing](#contributing)

## Dataset
The dataset used in this project is the NASA Turbofan Engine Degradation Simulation Dataset. It consists of multiple time-series measurements for different engines, including:
- `cycles`: Number of operational cycles
- `T24`: Total temperature at LPC outlet
- `T30`: Total temperature at HPC outlet
- `T50`: Total temperature at LPT outlet
- `P30`: Pressure at HPC outlet
- `Nf`: Physical fan speed
- `Nc`: Physical core speed
- `Ps30`: Pressure at HPC outlet
- `phi`: Fuel flow ratio to Ps30
- `NRf`: Corrected fan speed
- `NRc`: Corrected core speed
- `BPR`: Bypass ratio
- `htBleed`: Bleed Enthalpy
- `W31`: HPT coolant bleed
- `W32`: LPT coolant bleed

## Model
To make predictions using the trained model,

--model random_forest_model.joblib [model](https://huggingface.co/spaces/Gaurav069/NASA_Turbofan_RUL_and_Anomaly_Prediction_Playground/blob/main/random_forest_model.joblib)

## Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the repository.
Create a new branch.
Make your changes and commit them.
Push your changes to your fork.
Create a pull request.
