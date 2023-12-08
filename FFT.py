import math
import matplotlib.pyplot as plt
import numpy as np

def create_sin_wave(cycles, resolution):
    length = np.pi * 2 * cycles
    return np.sin(np.arange(0, length, length/resolution))

start_time = 0
end_time = 1
sample_rate = 1000
time = np.arange(start_time, end_time, 1/sample_rate)
theta = 0
frequency = 100
amplitude = 1
sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
figure(figsize=(20, 6), dpi=80)
plt.plot(sinewave)