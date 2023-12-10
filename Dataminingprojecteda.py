

# %%
import pandas as pd

# Read the CSV file into a DataFrame
file_path = '/Users/samikshaburkul/Downloads/games (1).csv' # Replace with your file path
data = pd.read_csv('/Users/samikshaburkul/Downloads/games (1).csv')

#%%
# Check if the columns exist in the dataset
if 'PTS_home' in data.columns and 'PTS_away' in data.columns:
    # Create a new column 'Total_Points_Scored' by summing 'PTS_home' and 'PTS_away'
    data['Total_Points_Scored'] = data['PTS_home'] + data['PTS_away']

    # Displaying the updated DataFrame with the new column
    print(data.head())
else:
    print("Columns PTS_home and/or PTS_away not found in the dataset.")

#%%
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assuming 'data' contains your DataFrame with columns: GAME_ID, Date, Total_Points_Scored

# Sort the data by GAME_ID and Date to ensure it's in the right order
data = data.sort_values(by=['GAME_ID', 'Date'])

# Create a function to generate features and target for prediction
def create_features_and_target(data):
    features = []
    target = []
    for i in range(5, len(data)):
        last_five_games = data.iloc[i-5:i]  # Extract the last 5 games
        next_game = data.iloc[i]  # Next game to predict
        if last_five_games['GAME_ID'].nunique() == 1 and last_five_games['GAME_ID'].iloc[0] == next_game['GAME_ID']:
            # Ensure the last five games are for the same player
            features.append(last_five_games['Total_Points_Scored'].values.reshape(1, -1))  # Reshape to 2D array
            target.append(next_game['Total_Points_Scored'])
    return features, target

# Generate features and target for prediction
features, target = create_features_and_target(data)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(features, target)

# Example: Predict points scored in the next game based on the last 5 games
# Replace [...] with the actual last five games' total points scored for a specific game to predict the next game's score
player_last_five_games = [...]  
predicted_points = model.predict([player_last_five_games])

print("Predicted points for the next game:", predicted_points)


# %%
