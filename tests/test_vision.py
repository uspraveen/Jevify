"""Vision-language System One: images in the state, same typed answers out."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL")

from PIL import Image

from jevify.engine.vision import VisionScorer, split_images

TINY_VLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _img(color):
    return Image.new("RGB", (64, 64), color)


def test_split_images_finds_and_placeholders():
    red, blue = _img("red"), _img("blue")
    state = {"photo": red, "notes": "a note", "extras": [{"figure": blue}, {"caption": "text only"}]}
    text_state, images = split_images(state)
    assert len(images) == 2
    assert text_state["photo"] == "<image 1>"
    assert text_state["extras"][0]["figure"] == "<image 2>"
    assert text_state["notes"] == "a note" and text_state["extras"][1]["caption"] == "text only"
    assert all(im.mode == "RGB" for im in images)


def test_split_images_ignores_non_image_strings():
    state = {"image": "not-a-path", "url": "https://example.com/page.html"}
    text_state, images = split_images(state)
    assert images == [] and text_state == state


@pytest.fixture(scope="module")
def scorer():
    return VisionScorer(TINY_VLM, device="cpu", dtype=torch.float32, batch_size=2)


def test_identifiers_are_single_token(scorer):
    ids = scorer.identifiers(30)
    assert ids[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_scores_an_image_question(scorer):
    q = {"type": "choice", "instructions": "What colour fills the image?",
         "criteria": {"red": None, "blue": None, "green": None}}
    item, rd = scorer.item({"image": _img("red")}, q)
    assert "<image 1>" in item.prefix or "image" in item.prefix.lower()
    assert len(item.images) == 1 and rd.keys == ["red", "blue", "green"]
    scores = scorer.score_many([item])[0]
    assert len(scores) == 3 and all(s < 0 for s in scores)


def test_noul_and_score_over_an_image(scorer):
    noul = {"type": "noul", "instructions": "Is the image red?"}
    item, rd = scorer.item({"image": _img("red")}, noul)
    s = scorer.score_many([item])[0]
    assert len(s) == 2 and rd.keys == ["1", "0"]
    sc = {"type": "score", "instructions": "How bright is the image?", "criteria": ["dark", "medium", "bright"]}
    item2, rd2 = scorer.item({"image": _img("white")}, sc)
    s2 = scorer.score_many([item2])[0]
    assert len(s2) == 3 and rd2.keys == ["0", "1", "2"]


def test_batches_mixed_questions(scorer):
    q = {"type": "choice", "instructions": "What colour?", "criteria": {"red": None, "blue": None}}
    items = [scorer.item({"image": _img(c)}, q)[0] for c in ("red", "blue", "red")]
    scores = scorer.score_many(items)
    assert len(scores) == 3 and all(len(s) == 2 for s in scores)


def test_write_vision_records_strips_images(tmp_path):
    """Regression: a PIL image on the state must not reach ``json.dumps``.

    ``BenchRecord.to_row`` encodes the state itself, so replacing the ``state`` column
    on the row afterwards is too late. This exact ordering mistake killed a scored
    A100 vision run after the predictions were already on disk.
    """
    import json

    from jevify.bench.record import BenchRecord
    from jevify.runners.vision_runner import write_vision_records

    recs = [
        BenchRecord(id="pope/test/0", source="pope", primitive="noul", split="test",
                    state={"image": _img("red"), "question": "Is there a cat?"},
                    question={"type": "noul", "instructions": "yes?"}, label=1,
                    meta={"category": "adversarial"}),
        BenchRecord(id="ai2d/test/1", source="ai2d", primitive="choice", split="test",
                    state={"image": _img("blue"), "question": "Which part?"},
                    question={"type": "choice", "instructions": "pick", "criteria": {"A": "root", "B": "stem"}},
                    label="A"),
    ]
    out = tmp_path / "test_records.jsonl"
    n = write_vision_records(out, recs)
    assert n == 2

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row, rec in zip(rows, recs):
        state = json.loads(row["state"])
        assert state["image"] == "<image 1>"           # placeholder, not an object
        assert state["question"] == rec.state["question"]
        assert BenchRecord.from_row(row).id == rec.id  # and it round-trips back
    assert rows[0]["label"] == "1" and rows[1]["label"] == "A"


def test_write_vision_records_handles_nested_and_multiple_images(tmp_path):
    import json

    from jevify.bench.record import BenchRecord
    from jevify.runners.vision_runner import write_vision_records

    rec = BenchRecord(id="x/test/0", source="x", primitive="choice", split="test",
                      state={"panels": [_img("red"), _img("blue")], "caption": "two panels"},
                      question={"type": "choice", "instructions": "?", "criteria": {"A": 1, "B": 2}}, label="A")
    out = tmp_path / "r.jsonl"
    write_vision_records(out, [rec])
    state = json.loads(json.loads(out.read_text(encoding="utf-8"))["state"])
    assert state["panels"] == ["<image 1>", "<image 2>"] and state["caption"] == "two panels"


def test_ask_answers_questions_about_an_image(scorer):
    """The served path: images in `state`, the same typed questions, the same wire format."""
    from jevify.load import JevifiedModel

    model = JevifiedModel(scorer, {"backbone": TINY_VLM, "chat": True, "modality": "vision",
                                   "recipe": {"mode": "index", "permutations": 1}})
    assert model.vision and "vision" in repr(model)
    out = model.ask({"image": _img("red"), "note": "a plain square"},
                    {"q1": {"type": "noul", "instructions": "Is the image red?"},
                     "q2": {"type": "choice", "instructions": "What colour is it?",
                            "criteria": {"red": "red", "blue": "blue", "green": "green"}},
                     "q3": {"type": "score", "instructions": "How bright?", "criteria": ["dark", "medium", "bright"]}})
    assert set(out) == {"q1", "q2", "q3"}
    assert out["q1"]["type"] == "noul" and 0.0 <= out["q1"]["noul"] <= 1.0
    assert out["q2"]["choice"] in ("red", "blue", "green") and abs(sum(out["q2"]["probabilities"].values()) - 1) < 1e-3
    assert set(out["q3"]["legend"]) == {"0", "1", "2"} and 0.0 <= out["q3"]["score"] <= 2.0
    assert scorer.last_input_tokens > 0                     # the processor's count, for `usage`


def test_resolve_images_loads_each_reference_once(tmp_path, monkeypatch):
    from jevify.engine import vision as V

    path = tmp_path / "a.png"; _img("red").save(path)
    calls = {"n": 0}
    real = V.load_image
    def counting(x):
        calls["n"] += 1
        return real(x)
    monkeypatch.setattr(V, "load_image", counting)
    resolved = V.resolve_images({"image": str(path), "note": "text", "extras": [{"figure": str(path)}]})
    assert calls["n"] == 2 and V._is_image(resolved["image"]) and V._is_image(resolved["extras"][0]["figure"])
    text_state, images = V.split_images(resolved)         # a later split finds loaded images: no more loads
    assert calls["n"] == 2 and len(images) == 2 and text_state["image"] == "<image 1>"
