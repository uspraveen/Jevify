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
