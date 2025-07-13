import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv('player_stats.csv')

# Features and target
X = df[['matches', 'innings', 'strike_rate', 'average']]
y = df['runs']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model trained and saved as model.pkl")
