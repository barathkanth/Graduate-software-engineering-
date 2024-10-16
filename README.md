# Efficient Data Stream Anomaly Detection

## Project Description
This project implements real-time anomaly detection using a combination of Z-scores and Exponential Weighted Moving Average (EWMA). The data stream simulates real-world conditions with seasonal components, noise, and random anomalies.

## Algorithm Overview
- **Z-scores** are used to detect extreme values.
- **EWMA** helps handle concept drift by adapting to changes in the mean and variance of the data.

## Requirements
- Python 3.x
- Required libraries: `numpy`, `matplotlib`

## How to Run
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the script: `python anomaly_detection.py`
