#%%
import pandas as pd

# Replace 'path_to_file' with the actual path to your CSV file
file_path = '/Users/samikshaburkul/Downloads/games (1).csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify it has been read properly
print(data.head())

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' contains your DataFrame from the NBA dataset

# Access the column containing points scored by players (replace 'Points' with the actual column name)
points_scored = data['HOME_TEAM_WINS']

# Summary statistics to understand the data
print(points_scored.describe())

# Visualize the distribution of points scored
plt.figure(figsize=(8, 6))
plt.hist(points_scored, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Points Scored by Players')
plt.xlabel('HOME_TEAM_WINS')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
