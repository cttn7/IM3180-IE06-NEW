import pandas as pd

# Load dataset
df = pd.read_csv('data/trump_slow_meltdown.csv', encoding='ISO-8859-1')  # Ensure this file exists in the specified location
df.fillna({'comment': 'No Comment'}, inplace=True)

# Select the relevant columns and rename them for clarity
df = df[['comment', 'OPR']].rename(columns={'comment': 'Document', 'OPR': 'Label'})

# Save processed data
df.to_csv('data/processed_data.csv', index=False)
print("Data preprocessing complete. Processed data saved to 'data/processed_data.csv'.")
