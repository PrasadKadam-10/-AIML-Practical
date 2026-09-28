import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 45, 52, 58, 65, 70, 78, 85, 92]
}

df = pd.DataFrame(data)

# Display dataset
print("Dataset:")
print(df)

# Calculate correlation
correlation = df["Hours_Studied"].corr(df["Marks"])

print("\nCorrelation between Hours Studied and Marks:")
print(correlation)

# Plot relationship
sns.scatterplot(
    x="Hours_Studied",
    y="Marks",
    data=df
)

plt.title("Association Between Hours Studied and Marks")
plt.xlabel("Hours Studied (Independent Variable)")
plt.ylabel("Marks (Dependent Variable)")
plt.show()
