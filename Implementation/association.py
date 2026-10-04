# Practical: Association Analysis
# Objective: Study the relationship between two variables using
#            Pearson correlation and visualize it with a scatter plot.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Step 1: Prepare the dataset
# We record how many hours of sleep students got the night before
# an exam and the score they achieved.
# ------------------------------------------------------------------
sleep_hours = [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
exam_scores  = [48, 55, 52, 61, 63, 70, 74, 80, 83, 91]

student_data = pd.DataFrame({
    "Sleep_Hours": sleep_hours,
    "Exam_Score" : exam_scores
})

print("=" * 40)
print("       Student Performance Dataset")
print("=" * 40)
print(student_data.to_string(index=False))
print()

# ------------------------------------------------------------------
# Step 2: Compute Pearson Correlation Coefficient
# A value close to +1 means a strong positive association.
# ------------------------------------------------------------------
r = student_data["Sleep_Hours"].corr(student_data["Exam_Score"])

print(f"Pearson Correlation Coefficient (r) : {r:.4f}")

if r >= 0.8:
    strength = "strong positive"
elif r >= 0.5:
    strength = "moderate positive"
elif r > 0:
    strength = "weak positive"
else:
    strength = "negative or no"

print(f"Interpretation: There is a {strength} association")
print(f"between sleep hours and exam scores.\n")

# ------------------------------------------------------------------
# Step 3: Visualize the association
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=student_data,
    x="Sleep_Hours",
    y="Exam_Score",
    color="steelblue",
    s=80,
    edgecolor="black"
)

# Add a trend line using seaborn regplot (linear fit)
sns.regplot(
    data=student_data,
    x="Sleep_Hours",
    y="Exam_Score",
    scatter=False,
    color="tomato",
    label="Trend Line"
)

plt.title("Association Between Sleep Hours and Exam Score", fontsize=13)
plt.xlabel("Sleep Hours (per night before exam)")
plt.ylabel("Exam Score (out of 100)")
plt.legend()
plt.tight_layout()
plt.show()
