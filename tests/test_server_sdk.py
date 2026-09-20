"""The served Jevified model must be a drop-in for TypeSafe's official SDK."""
import threading
import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
typesafe_sdk = pytest.importorskip("typesafe_sdk")

from jevify.server import build_app, make_service

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"
PORT = 8765


@pytest.fixture(scope="module")
def server():
    svc = make_service(TINY, device="cpu", permutations=1)
    config = uvicorn.Config(build_app(svc), host="127.0.0.1", port=PORT, log_level="warning")
    srv = uvicorn.Server(config)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{PORT}"
    srv.should_exit = True


def test_official_sdk_against_jevified_server(server, monkeypatch):
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    monkeypatch.setenv("TYPESAFE_BASE_URL", server)
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-needed")
    with TypeSafeClient() as client:
        models = client.models.list().models
        assert any(m.name == "jevify-latest" for m in models)
        r = client.system_one(
            state={"ticket": "I was charged twice for order A-104, please refund the duplicate."},
            model="jevify-latest",
            questions={
                "dept": Choice(instructions="Which team should handle `ticket`?",
                               criteria={"billing": "Payments and refunds", "shipping": "Delivery problems", "other": None}),
                "anger": Score(instructions="How angry is the customer?", criteria=["calm", "annoyed", "furious"]),
                "refund": Noul(instructions="Does `ticket` ask for a refund?"),
            },
        )
    dept, anger, refund = r.answers["dept"], r.answers["anger"], r.answers["refund"]
    assert dept.choice in ("billing", "shipping", "other")
    assert abs(sum(dept.probabilities.values()) - 1) < 1e-3 and 0 <= dept.confidence <= 1
    assert 0 <= anger.score <= 2 and set(anger.legend) == {0, 1, 2} or set(anger.legend) == {"0", "1", "2"}
    assert 0 <= refund.noul <= 1
    assert r.usage.input_tokens > 0 and r.usage.output_tokens > 0


def test_validation_error_shape(server):
    import httpx

    resp = httpx.post(f"{server}/v1/systemone", json={"state": "x", "model": "jevify-latest", "questions": {"q": {"type": "choice", "criteria": {"only": None}}}})
    assert resp.status_code == 422 and isinstance(resp.json()["detail"], list)
