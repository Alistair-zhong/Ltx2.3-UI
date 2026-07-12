from fastapi.testclient import TestClient

from ltx23_ui.app import app


def test_health_and_frame_calculator() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True

        response = client.post("/api/frames", json={"duration": 16, "fps": 25})
        assert response.status_code == 200
        assert response.json() == {"num_frames": 393, "video_duration": 15.72}


def test_static_ui_is_served() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "A2V LoRA Lab" in response.text

