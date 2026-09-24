\# Real-Time ECG Health Monitoring with Edge-AI



A software-based ECG health monitoring system that uses Edge-AI for ECG anomaly detection, clinical ECG analysis, real-time risk monitoring, alert generation, and logging.



\## Project Overview



This project implements a \*\*No-QNX software baseline\*\* for real-time ECG health monitoring.



ECG data is processed using Python and analyzed using clinical ECG signal processing techniques. A CNN-based TensorFlow Lite model is used for ECG beat classification into \*\*Normal\*\* and \*\*Arrhythmia\*\* categories.



The system also monitors heart-rate conditions, determines risk levels, generates critical alerts, and maintains monitoring logs.



\## System Workflow



```text

ECG Data

&#x20;  ↓

ECG Preprocessing

&#x20;  ↓

Clinical ECG Analysis

&#x20;  ↓

Edge-AI CNN Inference

&#x20;  ↓

Normal / Arrhythmia Detection

&#x20;  ↓

Risk-Level Detection

&#x20;  ↓

Alert Generation

&#x20;  ↓

Logging

&#x20;  ↓

Real-Time Dashboard



Key Features

ECG signal preprocessing

Clinical ECG signal analysis

CNN-based ECG beat classification

TensorFlow Lite Edge-AI inference

Normal and Arrhythmia detection

Heart-rate based risk monitoring

Risk-level classification

Critical condition alerts

Automated email notification

Fault-recovery testing

AI inference latency benchmarking

ECG preprocessing benchmarking

CPU performance testing

End-to-end performance testing

Resource monitoring

Real-time Flask dashboard

Technologies Used

Python

TensorFlow

TensorFlow Lite

NumPy

NeuroKit2

Flask

SMTP

MIT-BIH Arrhythmia Database

AI Model



The project uses a CNN model trained for ECG beat classification and converted to TensorFlow Lite for lightweight Edge-AI inference.



Model Input

187 ECG samples per beat

Model Output

Normal

Arrhythmia



The trained TensorFlow Lite model is stored as:



ecg\_model.tflite

Clinical ECG Analysis



The system performs analysis of important ECG signal characteristics.



The analysis includes:



P wave

QRS complex

T wave

RR interval

Heart rate

QT interval

QTc

ST-segment related analysis



These parameters provide additional ECG information beyond simple heart-rate monitoring.



ECG Data



The project uses ECG data derived from the MIT-BIH Arrhythmia Database.



The ECG data is processed into individual beats and prepared for CNN-based classification.



Large dataset files are excluded from the GitHub repository using .gitignore to keep the repository lightweight.



Risk Monitoring



The system uses heart-rate conditions and AI prediction to determine the monitoring status.



The implemented risk levels are:



Risk Level	Status	Description

GREEN	Normal	Normal monitoring condition

YELLOW	Caution	Elevated or low heart-rate condition

ORANGE	High Risk	Higher heart-rate condition

RED	Critical	Critical heart-rate condition



The system can generate an alert when a critical condition is detected.



Alert System



The system supports automated email alerts for critical conditions.



The email application password is not stored directly in the source code.



Instead, the application reads the password from an environment variable:



EMAIL\_APP\_PASSWORD



This prevents sensitive email credentials from being included in the GitHub repository.



Real-Time Dashboard



The project uses Flask to provide a local real-time ECG monitoring dashboard.



The dashboard provides monitoring information such as:



Heart rate

AI prediction

AI confidence

Risk level

Monitoring status

Alert status



The main Flask application is:



app.py

Installation \& Setup

Clone the Repository

git clone https://github.com/pavithra-code-source/ECG\_healthmonitoring.git

cd ECG\_healthmonitoring

Install Dependencies

pip install flask numpy tensorflow neurokit2

How to Run



Run the main Flask application:



python app.py



The application starts a local Flask server.



Open the following address in your browser:



http://127.0.0.1:5000



The real-time ECG monitoring dashboard will be displayed.



API Endpoints

Endpoint	Description

/	Real-time ECG monitoring dashboard

/api/reading	Returns the latest monitoring data as JSON

/api/critical	Triggers a critical demonstration state

Performance Evaluation



The repository contains multiple scripts for evaluating different aspects of the No-QNX ECG monitoring system.



The evaluation includes:



AI inference latency

ECG preprocessing performance

CPU performance

End-to-end processing

Fault recovery performance

Resource monitoring



The repository includes benchmarking scripts such as:



ai\_inference\_benchmark.py

performance\_test.py

preprocessing\_benchmark.py

cpu\_test.py

recovery\_performance.py

recovery\_realistic\_test.py

resource\_monitor.py

end\_to\_end\_test.py



Benchmark results can vary depending on the test configuration and computer environment.



Fault Recovery



The project includes fault-recovery test scripts that simulate failures in the monitoring system and evaluate the recovery behavior.



The recovery tests evaluate whether a simulated failure can be detected and whether the required monitoring process can be restored.



Relevant scripts include:



fault\_recovery\_test.py

recovery\_performance.py

recovery\_realistic\_test.py

Project Structure

ECG\_healthmonitoring/

│

├── app.py

├── ecg\_simulator.py

├── ecg\_cnn\_inference.py

├── clinical\_analysis.py

├── alert\_logging.py

├── network\_check.py

│

├── ai\_inference\_benchmark.py

├── performance\_test.py

├── preprocessing\_benchmark.py

├── cpu\_test.py

│

├── fault\_recovery\_test.py

├── recovery\_performance.py

├── recovery\_realistic\_test.py

├── resource\_monitor.py

├── end\_to\_end\_test.py

│

├── realtime\_ecg.py

├── no\_qnx\_results.txt

├── ecg\_log.txt

│

├── ecg\_model.tflite

│

├── templates/

│   └── index.html

│

└── .gitignore

Project Scope



This repository contains the No-QNX software baseline of the ECG health monitoring project.



The implementation focuses on:



ECG monitoring

ECG preprocessing

Clinical ECG analysis

Edge-AI inference

Normal/Arrhythmia classification

Heart-rate based risk monitoring

Alert generation

Logging

Fault-recovery testing

Performance evaluation

Real-time dashboard monitoring



This baseline can be used for comparison with an RTOS-based implementation.



Future Scope



The following features can be explored in future versions:



Predictive risk forecasting

Multi-sensor fusion using SpO2 and motion sensors

Voice-based alerts

Multi-vital monitoring

Low-power AI acceleration using FPGA or ASIC

On-device federated learning

Wearable ECG monitoring

Cloud-based long-term health monitoring

Built For



Global Innovation Hackathon 2026 — Build for a Better Future



Organized by Bharat Academix



Author



Pavithra K



Electronics and Communication Engineering



Disclaimer



This project is an academic/software prototype and is not intended for medical diagnosis, treatment, or clinical decision-making.



