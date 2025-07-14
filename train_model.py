import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('winequality-red.csv', sep=',')

# Binarize quality: good (>=7) vs bad (<7)
df['quality_label'] = (df['quality'] >= 7).astype(int)
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
pickle.dump(clf, open('model.pkl', 'wb'))
