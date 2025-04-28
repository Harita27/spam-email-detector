import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print(data.head())  # 👈 Check if data is loaded

# Encode labels
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Split into features and labels
X = data['message']
y = data['label_num']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create and train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Try your own email
your_email = ["Congratulations! You've won a free ticket to Bahamas!"]
your_email_vec = vectorizer.transform(your_email)
prediction = model.predict(your_email_vec)
print("Spam" if prediction[0] == 1 else "Not Spam")
