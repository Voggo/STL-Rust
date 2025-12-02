import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import chirp

def generate_chirp_signal(duration, sampling_rate, start_frequency, end_frequency, filename):
    """
    Generates a chirp signal with linearly changing frequency and saves it to a CSV file.

    Args:
        duration (float): The duration of the signal in seconds.
        sampling_rate (int): The number of samples per second.
        start_frequency (float): The starting frequency of the sine wave.
        end_frequency (float): The ending frequency of the sine wave.
        filename (str): The name of the CSV file to save the signal to.
    """
    filename = "signal_generation/" + filename
    signal_size = int(duration * sampling_rate)
    
    # Create a time vector from 0 to duration.
    t = np.linspace(0, duration, signal_size, endpoint=False)

    # Generate chirp signal
    signal_values = chirp(t, f0=start_frequency, f1=end_frequency, t1=duration, method='linear')
    
    # Plot the signal before saving
    print("Displaying plot. Close the plot window to save the file.")
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal_values)
    plt.title("Generated Chirp Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

    # Write the signal to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestep', 'value'])
        for i in range(signal_size):
            writer.writerow([t[i], signal_values[i]])

if __name__ == '__main__':
    # --- Configuration ---
    DURATION = 10000         # seconds
    SAMPLING_RATE = 1  # Hz
    START_FREQUENCY = 0.1   # Hz
    END_FREQUENCY = 0.0001   # Hz
    OUTPUT_FILENAME = 'signal.csv'
    # ---------------------

    generate_chirp_signal(DURATION, SAMPLING_RATE, START_FREQUENCY, END_FREQUENCY, OUTPUT_FILENAME)
    print(f"Signal generated and saved to {OUTPUT_FILENAME}")
