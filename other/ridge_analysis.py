import pandas as pd

# Load your file
df = pd.read_csv("../csvs/ridge_cv_results_full.csv")

df['accuracy'] = df['accuracy'] - 0.081

# Group by 'ridge' and calculate mean and standard deviation
accuracy_stats = df.groupby("ridge")["accuracy"].agg(['mean', 'std']).reset_index()

print(accuracy_stats)
