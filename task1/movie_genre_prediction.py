import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# for reading the txt file 
def read_and_parse_file(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            data = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            data = file.read()
    lines = data.strip().split('\n')
    parsed_data = [line.split(' ::: ') for line in lines]
    return parsed_data

train_data_file = 'train_data.txt'
test_data_file = 'test_data.txt'
test_data_solution_file = 'test_data_solution.txt'

train_data = read_and_parse_file(train_data_file)
test_data = read_and_parse_file(test_data_file)
test_data_solution = read_and_parse_file(test_data_solution_file)

train_columns = ['ID', 'Title', 'Genre', 'Plot']
test_columns = ['ID', 'Title', 'Plot']
solution_columns = ['ID', 'Title', 'Genre', 'Plot']

train_df = pd.DataFrame(train_data, columns=train_columns)
test_df = pd.DataFrame(test_data, columns=test_columns)
solution_df = pd.DataFrame(test_data_solution, columns=solution_columns)

# data preprocessing
train_df['Plot'] = train_df['Plot'].str.lower()
test_df['Plot'] = test_df['Plot'].str.lower()

# feature extraction : Using TF-IDF to convert the text data into numerical features.
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['Plot'])
X_test = vectorizer.transform(test_df['Plot'])
y_train = train_df['Genre']
y_test = solution_df['Genre']

# Training the Model using logistic regression 
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)