from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ISSUES = [
    {
        "id": 101,
        "title": "Application crashes during user login",
        "description": "The application crashes when users enter valid credentials."
    },
    {
        "id": 102,
        "title": "Database connection timeout",
        "description": "Database connections fail after several requests."
    },
    {
        "id": 103,
        "title": "Password reset email not received",
        "description": "Users do not receive password recovery emails."
    },
    {
        "id": 104,
        "title": "Dashboard loading very slowly",
        "description": "Dashboard takes too long to load information."
    },
    {
        "id": 105,
        "title": "Login page freezes after submitting password",
        "description": "The login screen becomes unresponsive after entering credentials."
    },
    {
        "id": 106,
        "title": "Payment processing returns server error",
        "description": "Checkout fails with a 500 internal server error."
    },
    {
        "id": 107,
        "title": "Application crashes when logging in",
        "description": "Users experience a crash after submitting their login details."
    },
    {
        "id": 108,
        "title": "Profile image upload fails",
        "description": "Uploading profile pictures produces an error."
    },
    {
        "id": 109,
        "title": "API documentation is outdated",
        "description": "The API guide contains old endpoint information."
    },
    {
        "id": 110,
        "title": "Application freezes when user logs in",
        "description": "Users cannot access the dashboard because login freezes."
    }
]


class DuplicateDetector:

    def __init__(self):
        self.documents = [
            issue["title"] + " " + issue["description"]
            for issue in ISSUES
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )

        self.matrix = self.vectorizer.fit_transform(self.documents)

    def find_duplicates(self, title, description="", top_k=3):

        query = title + " " + description

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        results = []

        ranked = similarities.argsort()[::-1][:top_k]

        for index in ranked:
            issue = ISSUES[index]

            results.append({
                "id": issue["id"],
                "title": issue["title"],
                "description": issue["description"],
                "similarity": round(float(similarities[index]) * 100, 2),
                "duplicate": bool(similarities[index] >= 0.55)
            })

        return results


if __name__ == "__main__":
    detector = DuplicateDetector()

    matches = detector.find_duplicates(
        "Application crashes when logging in",
        "The user cannot access the account."
    )

    for match in matches:
        print(match)
