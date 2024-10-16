import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
from matplotlib.animation import FuncAnimation

# Set parameters for anomaly detection
window_size = 10  # The size of the moving window for calculating statistics
z_thresh = 3  # Z-score threshold for flagging anomalies
alpha = 0.3  # Smoothing factor for the Exponential Weighted Moving Average (EWMA)

def generate_seasonal_component(t):
    # Define a simple repeating pattern: values go from 0 to 10, then back to 0.
    cycle = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2]
    return cycle[t % len(cycle)]  # Use the remainder operator to cycle through the pattern

def generate_data_stream():
    """
    Simulates a continuous data stream by generating values with seasonal patterns, noise,
    and occasional anomalies. 
    """
    t = 0
    while True:
        # Seasonal component (cyclical pattern)
        seasonal_component = generate_seasonal_component(t)

        # Random noise (simulates real-world fluctuations)
        noise = np.random.normal(0, 2)  

        # Anomaly: 1% chance of introducing a random anomaly between 20 and 100
        anomaly = np.random.choice([0, np.random.uniform(20, 100)], p=[0.99, 0.01])

        # Yield the data point which combines seasonal component, noise, and anomaly
        yield seasonal_component + noise + anomaly  
        t += 1

class AnomalyDetector:
    """
    AnomalyDetector class detects anomalies using a combination of Z-scores and EWMA.
    Z-scores detect extreme values, while EWMA handles concept drift by adapting to changes in the mean and variance.
    """
    def __init__(self, window_size, alpha, z_thresh):
        self.window_size = window_size
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.data_window = deque(maxlen=window_size)  # Deque for storing data points within the window
        self.ewma_mean = 0  # Initialize EWMA mean
        self.ewma_var = 0  # Initialize EWMA variance

    def update_ewma(self, value):
        """
        Update the EWMA mean and variance based on the new value.
        """
        if len(self.data_window) == 0:
            self.ewma_mean = value  # Initialize mean if this is the first value
            self.ewma_var = 0  # Initialize variance
        else:
            # Update EWMA mean
            self.ewma_mean = (self.alpha * value) + ((1 - self.alpha) * self.ewma_mean)
            # Update EWMA variance
            self.ewma_var = (self.alpha * (value - self.ewma_mean) ** 2) + ((1 - self.alpha) * self.ewma_var)

    def detect_anomaly(self, value):
        """
        Detect anomalies using a combination of Z-scores and EWMA. If the Z-score exceeds the threshold,
        the value is flagged as an anomaly.
        """
        self.data_window.append(value)  # Add new value to the data window
        if len(self.data_window) < self.window_size:
            return False  # Not enough data points yet

        # Calculate the rolling mean and standard deviation
        mean = np.mean(self.data_window)
        std = np.std(self.data_window)
        z_score = (value - mean) / std if std > 0 else 0  # Calculate Z-score, handle division by zero

        # Update EWMA statistics
        self.update_ewma(value)

        # Flag as anomaly if Z-score is above the threshold
        return abs(z_score) > self.z_thresh

# Set up for real-time visualization
fig, ax = plt.subplots()
x_vals = []
y_vals = []
anomaly_vals = []

# Data stream generator and detector
stream = generate_data_stream()
detector = AnomalyDetector(window_size=window_size, alpha=alpha, z_thresh=z_thresh)

def update(frame):
    """
    This function updates the plot in real-time by adding new data points and flagging anomalies.
    """
    try:
        # Get the next value from the data stream
        new_val = next(stream)

        # Update the lists for plotting
        x_vals.append(frame)
        y_vals.append(new_val)
        is_anomaly = detector.detect_anomaly(new_val)
        anomaly_vals.append(is_anomaly)

        # Clear the previous frame and plot the new data
        ax.clear()
        ax.plot(x_vals, y_vals, label='Data Stream')

        # Highlight anomalies in red
        ax.scatter([x_vals[i] for i, anomaly in enumerate(anomaly_vals) if anomaly],
                   [y_vals[i] for i, anomaly in enumerate(anomaly_vals) if anomaly], color='r', label='Anomalies')

        ax.legend()
    except Exception as e:
        print(f"Error encountered during update: {e}")

# Real-time animation
ani = FuncAnimation(plt.gcf(), update, interval=100)  # Update every 100ms
plt.show()
