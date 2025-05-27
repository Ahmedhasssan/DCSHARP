import numpy as np
import matplotlib.pyplot as plt

# Load data from the .txt file
data = np.loadtxt('/home/ah2288/gs_baseline/gaussian-splatting/scale_gradient/gradient14900.txt')

# Assuming the .txt file has two columns: x and y
# x = data[:, 0]  # First column
# y = data[:, 1]  # Second column

# Create a plot
plt.plot(data)

# Add labels and title
plt.xlabel('Samples')
plt.ylabel('Scale Gradient')
plt.title('Iteration 14900: Gradient Profile of Scale Parameters')

# Show grid
plt.grid(True)

y_min, y_max = min(data), max(data)  # Find min and max of y-values
# plt.yticks(np.arange(y_min, y_max, (y_max - y_min) / 10))

plt.savefig('Gradient_14900.png', dpi=300)

# Display the plot
plt.show()