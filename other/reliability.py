import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("tkagg")

# Load CSV files, skipping the first row
predictions_csv = pd.read_csv("timestep_predictions_Mel_IP.csv", header=None, skiprows=1)
true_labels = pd.read_csv("true_labels_Mel_IP.csv", header=None, skiprows=1)

timestep_predictions = []

# Convert each row to a numpy array
for i, row in predictions_csv.iterrows():
    timestep_predictions.append(row.values)

most_common_percentages = []
final_preds = []
y_true = []

for pred_per_timestep in timestep_predictions:
    # Remove silence (label 10)
    non_silence_preds = pred_per_timestep[pred_per_timestep != 10]

    # Majority vote
    final_label = mode(non_silence_preds, keepdims=False).mode
    final_preds.append(final_label)

    # Percentage of timesteps voting for majority label
    counts = np.bincount(non_silence_preds)
    percent = (counts[final_label] / len(non_silence_preds)) * 100
    most_common_percentages.append(percent)

most_common_percentages = np.array(most_common_percentages)
final_preds = np.array(final_preds)

for i, row in true_labels.iterrows():
    non_silence = row.values[row.values != 10]
    y_true.append(int(non_silence[0]))

# Check if majority vote prediction is correct
correct = (final_preds == y_true)

# Combine into DataFrame for analysis
df = pd.DataFrame({
    "final_pred": final_preds,
    "true_label": y_true,
    "percent_majority": most_common_percentages,
    "correct": correct
})

df.to_csv("majority_confidence_analysis_per_timestep.csv", index=False)

# Make sure 'correct' is boolean
df['correct'] = df['correct'].astype(bool)

# Boxplot: percent_majority vs correctness
plt.figure(figsize=(6, 5))
sns.boxplot(x='correct', y='percent_majority', data=df)
plt.xticks([0, 1], ['Incorrect', 'Correct'])
plt.ylabel('Percent Majority (Confidence)')
plt.xlabel('Prediction Correct?')
plt.title('Relation between Confidence and Correctness')
plt.show()

# Bin percent_majority into ranges
df['confidence_bin'] = pd.cut(df['percent_majority'], bins=11)

# Compute fraction correct in each bin
bin_stats = df.groupby('confidence_bin')['correct'].mean()

plt.figure(figsize=(7,5))
bin_stats.plot(marker='o')
plt.ylabel('Fraction Correct')
plt.xlabel('Confidence Bin (%)')
plt.title('Accuracy vs Confidence')
plt.show()