from pathlib import Path

import pytest

from app import create_app


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "model" / "spam_classifier.joblib").exists(),
    reason="Run python train_model.py before running prediction tests",
)
def test_prediction_page_returns_result():
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()
    response = client.post(
        "/",
        data={
            "subject": "You won a prize",
            "message": "Congratulations! You have won a free prize. Call now!",
        },
    )
    assert response.status_code == 200
    assert b"MODEL RESULT" in response.data


def test_home_page_loads_with_separate_fields():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b'name="subject"' in response.data
    assert b'name="message"' in response.data
