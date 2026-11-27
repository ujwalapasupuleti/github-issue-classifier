import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TRAINING_DATA = [
    ("Application crashes during login", "Bug"),
    ("Server returns internal server error", "Bug"),
    ("Database connection fails", "Bug"),
    ("Page displays incorrect information", "Bug"),
    ("API endpoint returns 404 error", "Bug"),
    ("Memory leak in application", "Bug"),
    ("Button does not respond", "Bug"),
    ("Application freezes during checkout", "Bug"),
    ("Users cannot reset their password", "Bug"),
    ("Login authentication fails", "Bug"),

    ("Add dark mode to dashboard", "Feature Request"),
    ("Implement user authentication", "Feature Request"),
    ("Add export to PDF functionality", "Feature Request"),
    ("Allow users to upload profile pictures", "Feature Request"),
    ("Add notification system", "Feature Request"),
    ("Support multiple languages", "Feature Request"),
    ("Add payment gateway integration", "Feature Request"),
    ("Create analytics dashboard", "Feature Request"),
    ("Add search functionality", "Feature Request"),
    ("Implement two factor authentication", "Feature Request"),

    ("Update installation instructions", "Documentation"),
    ("Correct API documentation", "Documentation"),
    ("Add examples to README", "Documentation"),
    ("Fix spelling mistakes in user guide", "Documentation"),
    ("Document authentication endpoints", "Documentation"),
    ("Explain database configuration", "Documentation"),
    ("Update deployment guide", "Documentation"),
    ("Add setup instructions", "Documentation"),
    ("Improve developer documentation", "Documentation"),
    ("Document environment variables", "Documentation"),
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


class IssueClassifier:
    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=5000
            )),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

        texts = [clean_text(x[0]) for x in TRAINING_DATA]
        labels = [x[1] for x in TRAINING_DATA]

        self.model.fit(texts, labels)

    def predict(self, title, description=""):
        text = clean_text(title + " " + description)

        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.classes_

        best_index = probabilities.argmax()

        return {
            "category": classes[best_index],
            "confidence": round(float(probabilities[best_index]) * 100, 2),
            "probabilities": {
                label: round(float(prob) * 100, 2)
                for label, prob in zip(classes, probabilities)
            }
        }


if __name__ == "__main__":
    classifier = IssueClassifier()

    result = classifier.predict(
        "Application crashes during login",
        "Users cannot access their accounts."
    )

    print("NLP CLASSIFIER TEST")
    print(result)
