import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("points.csv")

plt.plot(data["x"], data["net"], label="neural network")
plt.plot(data["x"], data["target"], label="target")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Compare neural net to target")
plt.legend()
plt.grid(True)

plt.show()