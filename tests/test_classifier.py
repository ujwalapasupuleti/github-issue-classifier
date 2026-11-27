from backend.nlp.classifier import IssueClassifier
from backend.nlp.duplicate_detector import DuplicateDetector
from backend.fluid.flow_engine import calculate_flow_metrics


def test_classifier():
    model = IssueClassifier()

    result = model.predict(
        "Application crashes during login"
    )

    assert result["category"] in [
        "Bug",
        "Feature Request",
        "Documentation"
    ]


def test_duplicate_detector():
    detector = DuplicateDetector()

    results = detector.find_duplicates(
        "Application crashes during login",
        "Users cannot log into their account."
    )

    assert len(results) == 3
    assert results[0]["similarity"] > 0


def test_fluid_engine():
    result = calculate_flow_metrics(
        total_issues=100,
        pending_issues=35,
        duplicate_issues=15,
        processing_capacity=50
    )

    assert result["flow_rate"] == 1.667
    assert result["pressure"] == 0.7
    assert result["recirculation"] == 15.0
    assert result["efficiency"] == 65.0
