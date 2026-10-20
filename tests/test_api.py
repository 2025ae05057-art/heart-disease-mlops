import joblib
import pandas as pd
from fastapi.testclient import TestClient
from app import main

def test_health_reports_model_loaded(monkeypatch):
    monkeypatch.setitem(main.state, "model", object())
    response = main.health()
    assert response["status"] == "ok" and response["model_loaded"] is True

def test_patient_schema_rejects_invalid_age():
    from pydantic import ValidationError
    try:
        main.PatientData(age=0, sex=1, cp=3, trestbps=140, chol=241, fbs=0, restecg=1, thalach=123, exang=1, oldpeak=.2, slope=2, ca=0, thal=7)
    except ValidationError:
        return
    assert False, "age=0 should be rejected"

def test_predict_returns_prediction_and_confidence(monkeypatch):
    class FakeModel:
        def predict_proba(self, frame):
            assert isinstance(frame, pd.DataFrame) and len(frame) == 1
            return [[0.2, 0.8]]
    monkeypatch.setitem(main.state, "model", FakeModel())
    patient=main.PatientData(age=57,sex=1,cp=3,trestbps=140,chol=241,fbs=0,restecg=1,thalach=123,exang=1,oldpeak=.2,slope=2,ca=0,thal=7)
    result=main.predict(patient)
    assert result["prediction"] == 1
    assert 0 <= result["confidence"] <= 1
    assert result["probability_disease"] == .8


def test_load_data_cleans_missing_values_and_binarizes_target(tmp_path):
    from app.train import load_data, COLUMNS
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [57, 0, 2, 120, "?", 0, 0, 174, 0, 0.0, 2, 0, 3, 1],
    ]
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(csv_path, header=False, index=False)
    df = load_data(str(csv_path))
    assert list(df.columns) == COLUMNS
    assert set(df["target"].unique()) == {0, 1}
    assert pd.isna(df.loc[2, "chol"])
