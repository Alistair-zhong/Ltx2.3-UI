from fastapi.testclient import TestClient

from ltx23_ui.app import app


def test_health_and_frame_calculator() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["version"] == "0.1.5"

        response = client.post("/api/frames", json={"duration": 16, "fps": 25})
        assert response.status_code == 200
        assert response.json() == {"num_frames": 393, "video_duration": 15.72}


def test_static_ui_is_served() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "A2V LoRA Lab" in response.text


def test_upload_saves_file(tmp_path, monkeypatch) -> None:
    import ltx23_ui.app as app_module

    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(app_module, "UPLOAD_DIR", upload_dir)
    with TestClient(app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("example image.jpg", b"fake-image-data", "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "example image.jpg"
    assert body["size"] == len(b"fake-image-data")
    assert (upload_dir / body["name"]).read_bytes() == b"fake-image-data"
