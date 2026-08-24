# Implement Bagging Classifier for Prompt Type Prediction

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
import pandas as pd


def question1():

    print("Question 1, Bagging classifier for Prompt Type Prediction")

    def load_data():

        df = pd.read_csv('ambi_dataset.csv')

        X = df['Prompt']
        Y = df['Label']

        return X, Y

    X, Y = load_data()


    def preprocessing(X):

        '''
        Since the input feature is text, numerical preprocessing
        such as StandardScaler is not directly applicable.

        TF-IDF converts the text into numerical features that
        can be given to the ML model.
        '''

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000
        )

        X_tfidf = vectorizer.fit_transform(X)

        return X_tfidf


    x_tfidf = preprocessing(X)


    def split_data(x_tfidf, Y):

        x_train, x_test, y_train, y_test = train_test_split(
            x_tfidf,
            Y,
            test_size=0.2,
            random_state=42,
            stratify=Y
        )

        return x_train, x_test, y_train, y_test


    x_train, x_test, y_train, y_test = split_data(x_tfidf, Y)


    def model(x_train, x_test, y_train, y_test):

        bagging_classifier = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=42
        )

        bagging_classifier.fit(x_train, y_train)

        y_pred = bagging_classifier.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

        return accuracy


    print(
        "The accuracy of the bagging classifier is =",
        model(x_train, x_test, y_train, y_test)
    )


question1()