import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection impo rt train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


games_df = pd.read_csv('games.csv')
games_details_df = pd.read_csv('games_details.csv', low_memory=False)
players_df = pd.read_csv('players.csv')
ranking_df = pd.read_csv('ranking.csv')
teams_df = pd.read_csv('teams.csv')

games_details_df = games_details_df[['FG_PCT', 'FG3_PCT', 'FT_PCT']]

imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Perform imputation
games_details_df = pd.DataFrame(imputer.fit_transform(games_details_df), columns=games_details_df.columns)

print(len(games_details_df))



#games_details_df.values.any()
#games_details_df.isnull().sum()



df = pd.read_csv('games_details.csv', low_memory=False)

# Select relevant features and target variable
features = df[['FG_PCT', 'FG3_PCT', 'FT_PCT']]
target = df['PLUS_MINUS'] > 0  # Binary classification: 1 if the team won, 0 if the team lost

# Handling missing values missing values
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

#  logistic regression model
model = LogisticRegression()

# Traing the model
model.fit(X_train, y_train)

# Making predictions 
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# Print metrics
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Lost', 'Won'], yticklabels=['Lost', 'Won'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()