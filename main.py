import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
import string
import nltk
from nltk.corpus import stopwords
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt








fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Display the first few rows of the datasets
print(fake.head())
print("This is real:")
print(real.head())



fake['label'] = 0  # Assign label 0 for fake news
real['label'] = 1  # Assign label 1 for real news

# Concate the datasets
data = pd.concat([fake, real], ignore_index=True)

# Checking for null values
print('Null values in the dataset:', data.isnull().sum(), data.isna().sum())

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Plotting the distribution of labels
data['label'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Distribution of Fake and Real News')
plt.xlabel('Label (0: Fake, 1: Real)')
plt.ylabel('Count')
plt.show()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') # For better converage of words


# Function to clean text data
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    #Remove punctuation
    text = text.translate(str.maketrans('','', string.punctuation))
    # Remove numbers
    text =re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text
    tokens = re.findall(r'\b\w+\b', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply the cleaning function to the 'text' column
data['text'] = data['text'].apply(clean_text)

# Display the cleaned data
# print(data.head(10))


# Lemetizing the text
lemetizer = WordNetLemmatizer()
data['lematized_text'] = data['text'].apply(lambda x: " ".join([lemetizer.lemmatize(word) for word in x.split()]))

print(data[['text', 'lematized_text']].head(5))
    

# Splitting the dataset into training and testing sets
X = data['lematized_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the models

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': LinearSVC(),
    'Naive Bayes': MultinomialNB()
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f'{name}: Accuracy = {accuracy_score(y_test, y_pred)}')
    print(f'{name}: Precision = {precision_score(y_test, y_pred)}')
    print(f'{name}: F1 Score = {f1_score(y_test, y_pred)}')
    print(f'{name}: Classification Report:\n{classification_report(y_test, y_pred)}')
    print(f'{name}: Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n')
    
    # Plotting confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    # plt.title(f'Confusion Matrix for {name}')
    # plt.xlabel('Predicted') 
    # plt.ylabel('Actual')
    # plt.show()
    
    
    
    
# Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(models['Logistic Regression'], model_file)
    
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
    
    