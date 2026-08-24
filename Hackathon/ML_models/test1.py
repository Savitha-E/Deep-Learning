from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import joblib


# ============================================================
# FUNCTION 1
# TRAIN + TEST + SAVE MODEL
# ============================================================

def train_test_save_model():

    print("Training Bagging Classifier")
    print("Prompt Type + Ambiguity Category")

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = pd.read_csv("ambi_dataset_3000.csv")

    X = df["Prompt"]
    Y_label = df["Label"]

    Y_granular = df["Granular_Cause_Labels"].fillna("")


    # --------------------------------------------------
    # 2. Convert granular labels into multiple labels
    # --------------------------------------------------

    granular_lists = []

    for labels in Y_granular:

        if labels.strip() == "":
            granular_lists.append([])

        elif labels.strip() == "VALID":
            granular_lists.append([])

        else:
            labels = labels.split(";")
            labels = [label.strip() for label in labels]

            granular_lists.append(labels)


    # --------------------------------------------------
    # 3. MultiLabelBinarizer
    # --------------------------------------------------

    mlb = MultiLabelBinarizer()

    Y_granular_encoded = mlb.fit_transform(
        granular_lists
    )


    # --------------------------------------------------
    # 4. TF-IDF
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )

    X_tfidf = vectorizer.fit_transform(X)


    # --------------------------------------------------
    # 5. Train/test split
    # --------------------------------------------------

    (
        X_train,
        X_test,
        y_label_train,
        y_label_test,
        y_granular_train,
        y_granular_test
    ) = train_test_split(
        X_tfidf,
        Y_label,
        Y_granular_encoded,
        test_size=0.2,
        random_state=42,
        stratify=Y_label
    )


    # --------------------------------------------------
    # 6. Main classifier
    # --------------------------------------------------

    label_classifier = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42
    )

    label_classifier.fit(
        X_train,
        y_label_train
    )


    # --------------------------------------------------
    # 7. Test main classifier
    # --------------------------------------------------

    label_prediction = label_classifier.predict(
        X_test
    )

    label_accuracy = accuracy_score(
        y_label_test,
        label_prediction
    )

    print("\n====================================")
    print("MODEL TEST RESULTS")
    print("====================================")

    print(
        "Main Label Accuracy:",
        round(label_accuracy, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_label_test,
            label_prediction
        )
    )


    # --------------------------------------------------
    # 8. Train granular classifiers
    # --------------------------------------------------

    granular_classifiers = []

    for i in range(
        Y_granular_encoded.shape[1]
    ):

        classifier = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=42
        )

        classifier.fit(
            X_train,
            y_granular_train[:, i]
        )

        granular_classifiers.append(
            classifier
        )


    # --------------------------------------------------
    # 9. SAVE EVERYTHING
    # --------------------------------------------------

    joblib.dump(
        label_classifier,
        "label_classifier.pkl"
    )

    joblib.dump(
        granular_classifiers,
        "granular_classifiers.pkl"
    )

    joblib.dump(
        vectorizer,
        "tfidf_vectorizer.pkl"
    )

    joblib.dump(
        mlb,
        "multilabel_binarizer.pkl"
    )


    print("\n====================================")
    print("MODEL SAVED SUCCESSFULLY")
    print("====================================")

    print("label_classifier.pkl")
    print("granular_classifiers.pkl")
    print("tfidf_vectorizer.pkl")
    print("multilabel_binarizer.pkl")


# ============================================================
# FUNCTION 2
# LOAD TRAINED MODEL + CLASSIFY NEW PROMPT
# ============================================================

def classify_prompt(user_prompt):

    # --------------------------------------------------
    # 1. LOAD SAVED FILES
    # --------------------------------------------------

    label_classifier = joblib.load(
        "label_classifier.pkl"
    )

    granular_classifiers = joblib.load(
        "granular_classifiers.pkl"
    )

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    mlb = joblib.load(
        "multilabel_binarizer.pkl"
    )


    # --------------------------------------------------
    # 2. Convert prompt to TF-IDF
    # --------------------------------------------------

    user_tfidf = vectorizer.transform(
        [user_prompt]
    )


    # --------------------------------------------------
    # 3. Main classification
    # --------------------------------------------------

    main_prediction = label_classifier.predict(
        user_tfidf
    )[0]


    # --------------------------------------------------
    # 4. Main confidence
    # --------------------------------------------------

    main_probabilities = (
        label_classifier.predict_proba(
            user_tfidf
        )[0]
    )

    main_confidence = main_probabilities.max()


    print("\n====================================")
    print("USER PROMPT")
    print(user_prompt)

    print("\nMAIN CLASSIFICATION")

    print(
        "Label      :",
        main_prediction
    )

    print(
        "Confidence :",
        round(main_confidence, 4)
    )


    # --------------------------------------------------
    # 5. Granular classification
    # --------------------------------------------------

    if main_prediction == "AMBIGUOUS":

        predicted_categories = []

        for i, classifier in enumerate(
            granular_classifiers
        ):

            prediction = classifier.predict(
                user_tfidf
            )[0]

            if prediction == 1:

                predicted_categories.append(
                    mlb.classes_[i]
                )


        print("\nAMBIGUITY CATEGORIES")

        if predicted_categories:

            for category in predicted_categories:

                print(
                    "-",
                    category
                )

        else:

            print(
                "No specific ambiguity category "
                "was predicted."
            )


    else:

        print(
            "\nThe prompt contains sufficient "
            "information and is classified as VALID."
        )


    print("====================================")


    # Return results as well
    return {
        "label": main_prediction,
        "confidence": main_confidence,
        "categories": (
            predicted_categories
            if main_prediction == "AMBIGUOUS"
            else []
        )
    }

train_test_save_model()
classify_prompt("I want to know everything about headache symptoms.")
