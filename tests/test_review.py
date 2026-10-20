"""Human-in-the-loop review: reviews.jsonl (append-only), /review + /api/review/*, panel / gallery coloured by review state, and the YOLO export of reviewed events."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.web.review import ReviewStore, clean_note, model_of
from apps.web.routes import Context, create_app, read_events
from sentinel.config import REPO

sys.path.insert(0, str(REPO / "tools"))
import export_reviews as ex  # noqa: E402
import import_exp020_reviews as imp  # noqa: E402

SHA_H, SHA_C = "a" * 64, "b" * 64
MODELS = {"helmet": {"file": "h.pt", "sha256": SHA_H}, "coco": {"file": "c.pt", "sha256": SHA_C}, "plate": {"file": "p.pt", "sha256": "c" * 64}}


class FakeWorker:
    def status(self):
        return {"running": False, "run": 1}

    def set_source(self, s):
        return str(s)

    def next_jpeg(self, last_id=0, timeout=2.0):
        return None


def event(i, type="no_helmet", status="closed", models=MODELS, **kw):
    e = {"id": f"vid-r1-e{i:04d}", "type": type, "status": status, "track_id": f"M_{i}", "run": 1, "frame_start": i * 10, "frame_end": i * 10 + 5, "created_at": f"2026-10-20T10:00:{i:02d}",
         "video_id": "vid", "camera_id": "cam", "key_box": [100, 100, 200, 260], "trigger_frame": i * 10, "light": "RED",
         "evidence": {"frame": f"vid-r1-e{i:04d}/frame.jpg", "vehicle": f"vid-r1-e{i:04d}/vehicle.jpg", "plate": None}, **kw}
    if models:
        e["models"] = models
    return e


@pytest.fixture
def env(tmp_path):
    (tmp_path / "templates").mkdir()
    for n in ("index.html", "review.html"):
        shutil.copy(REPO / "templates" / n, tmp_path / "templates" / n)
    ev = tmp_path / "static" / "violations"
    ev.mkdir(parents=True)
    evs = [event(1), event(2, "overload"), event(3, "red_light", models=None), event(4, status="open"), {"id": "gone", "status": "evicted"}]
    (ev / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    web = SimpleNamespace(max_upload_mb=1, upload_dir="static/uploads", allowed_extensions=[".mp4"], allowed_source_dirs=["static/uploads"], allow_rtsp=False, allow_camera_index=False)
    records = ["/static/violations/vid-r1-e0001/plate.jpg", "/static/violations/vid-r1-e0002/plate.jpg"]
    ctx = Context(worker=FakeWorker(), records=records, web=web, root=tmp_path, evidence_dir=ev, evidence_url="/static/violations")
    return TestClient(create_app(ctx)), ev, tmp_path


def post(c, eid, decision, note="", **kw):
    return c.post(f"/api/review/{eid}", json={"decision": decision, "note": note}, **kw)


def lines(ev):
    return [json.loads(l) for l in (ev / "reviews.jsonl").read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------- queue
def test_next_gives_the_oldest_closed_pending_event_with_urls_and_the_model_that_made_it(env):
    c, ev, _ = env
    d = c.get("/api/review/next").json()
    e = d["event"]
    assert e["id"] == "vid-r1-e0001" and d["pending"] == 4 and e["review_status"] == "pending"
    assert e["urls"]["frame"] == "/static/violations/vid-r1-e0001/frame.jpg" and e["urls"]["plate"] is None
    assert e["model"] == {"role": "helmet", "file": "h.pt", "sha256": SHA_H, "source": "event"}


def test_open_events_are_offered_only_when_no_closed_event_waits(env):
    c, ev, _ = env
    for n in (1, 2, 3):
        assert post(c, f"vid-r1-e000{n}", "unclear").status_code == 200
    d = c.get("/api/review/next").json()
    assert d["event"]["id"] == "vid-r1-e0004" and d["event"]["status"] == "open" and d["pending"] == 1
    post(c, "vid-r1-e0004", "rejected")
    assert c.get("/api/review/next").json() == {"event": None, "pending": 0}


def test_an_event_without_a_model_stamp_has_an_unknown_model_not_a_guess(env):
    c, _, _ = env
    for n in (1, 2):
        post(c, f"vid-r1-e000{n}", "unclear")
    e = c.get("/api/review/next").json()["event"]                                    # e3: red_light, no stamp -> unknown (the model running now may not be the one that made it)
    assert e["id"] == "vid-r1-e0003" and e["model"]["role"] == "coco" and e["model"]["sha256"] is None and e["model"]["file"] is None and "no model stamp" in e["model"]["source"]
    assert model_of({"type": "no_helmet", "models": MODELS})["file"] == "h.pt" and model_of({"type": "red_light", "models": MODELS})["file"] == "c.pt"


# ---------------------------------------------------------------- storage
def test_a_decision_is_appended_with_event_model_time_and_a_short_clean_note(env):
    c, ev, _ = env
    r = post(c, "vid-r1-e0001", "confirmed", "  no helmet\n\tfor sure \x00 ")
    assert r.status_code == 200
    rec = lines(ev)[0]
    assert rec["event_id"] == "vid-r1-e0001" and rec["decision"] == "confirmed" and rec["note"] == "no helmet for sure" and rec["schema"] == "sentinel.review/1"
    assert rec["event_type"] == "no_helmet" and rec["track_id"] == "M_1" and rec["run"] == 1 and rec["video_id"] == "vid"
    assert rec["model"]["file"] == "h.pt" and rec["model"]["sha256"] == SHA_H and len(rec["reviewed_at"]) == 19
    assert c.get("/api/review/next").json()["event"]["id"] == "vid-r1-e0002"           # decided events leave the queue


def test_reviewing_again_appends_a_new_line_and_never_edits_the_old_one(env):
    c, ev, _ = env
    post(c, "vid-r1-e0001", "confirmed", "first")
    first = (ev / "reviews.jsonl").read_bytes()
    post(c, "vid-r1-e0001", "rejected", "second thoughts")
    data = (ev / "reviews.jsonl").read_bytes()
    assert data.startswith(first) and len(lines(ev)) == 2                              # the old line is byte-for-byte unchanged
    st = c.get("/api/review/stats").json()
    assert st["rejected"] == 1 and st["confirmed"] == 0 and st["review_lines"] == 2 and st["reviewed_events"] == 1      # the last line wins
    store = ReviewStore(ev, read_events)
    assert [r["decision"] for r in store.history()] == ["confirmed", "rejected"] and store.status_of("vid-r1-e0001") == "rejected"


def test_a_damaged_line_in_reviews_jsonl_is_skipped(env):
    c, ev, _ = env
    post(c, "vid-r1-e0001", "confirmed")
    with open(ev / "reviews.jsonl", "a", encoding="utf-8") as f:
        f.write("{not json\n[1,2]\n")
    assert c.get("/api/review/stats").json()["confirmed"] == 1


# ---------------------------------------------------------------- validation
def test_unknown_event_is_404_and_bad_input_is_422_and_nothing_is_written(env):
    c, ev, _ = env
    assert post(c, "nope", "confirmed").status_code == 404
    assert post(c, "vid-r1-e0001", "maybe").status_code == 422
    assert post(c, "vid-r1-e0001", "confirmed", "x" * 301).status_code == 422
    assert c.post("/api/review/vid-r1-e0001", json={"note": "no decision"}).status_code == 422
    assert c.post("/api/review/vid-r1-e0001", data={"decision": "confirmed"}).status_code == 422          # a plain HTML form cannot post a review
    assert c.post("/api/review/vid-r1-e0001", content=b"not json", headers={"content-type": "application/json"}).status_code == 422
    assert post(c, "bad%20id%3B", "confirmed").status_code == 422
    assert not (ev / "reviews.jsonl").exists()
    assert post(c, "gone", "confirmed").status_code == 404                              # an evicted event no longer exists


def test_a_post_from_another_site_is_refused_but_a_local_page_is_accepted(env):
    c, ev, _ = env
    assert post(c, "vid-r1-e0001", "confirmed", headers={"origin": "https://evil.example"}).status_code == 403
    assert not (ev / "reviews.jsonl").exists()
    assert post(c, "vid-r1-e0001", "confirmed", headers={"origin": "http://127.0.0.1:8000"}).status_code == 200
    assert post(c, "vid-r1-e0002", "confirmed", headers={"origin": "http://localhost:8000"}).status_code == 200


def test_clean_note_limits_and_strips():
    assert clean_note(None) == "" and clean_note("a\x00b\n c") == "a b c" and len(clean_note("y" * 1000)) == 300


# ---------------------------------------------------------------- statistics
def test_stats_counts_and_precision_come_from_the_decisions_only(env):
    c, ev, _ = env
    post(c, "vid-r1-e0001", "confirmed")
    post(c, "vid-r1-e0002", "rejected")
    post(c, "vid-r1-e0003", "confirmed")
    post(c, "vid-r1-e0004", "unclear")
    with open(ev / "reviews.jsonl", "a", encoding="utf-8") as f:                       # an imported review (not an event of this folder), and a second no_helmet on the helmet model
        for i, d in enumerate(("rejected", "rejected", "confirmed")):
            f.write(json.dumps({"event_id": f"x/{i}", "decision": d, "event_type": "no_helmet", "model": {"file": "h.pt", "sha256": SHA_H}, "source": "exp020"}) + "\n")
    st = c.get("/api/review/stats").json()
    assert (st["pending"], st["confirmed"], st["rejected"], st["unclear"]) == (0, 3, 3, 1) and st["imported"] == 3
    nh = st["by_type"]["no_helmet"]
    assert (nh["confirmed"], nh["rejected"], nh["unclear"], nh["decided"], nh["precision"]) == (2, 2, 1, 4, 0.5)           # unclear is left out of the ratio
    assert st["by_type"]["overload"]["precision"] == 0.0 and st["by_type"]["red_light"]["precision"] == 1.0
    assert st["by_model"][f"h.pt ({SHA_H[:12]})"]["confirmed"] == 2 and st["by_model"]["? (no sha256)"]["confirmed"] == 1


def test_precision_is_none_when_nothing_is_decided_and_pending_counts_events_without_a_review(env):
    c, _, _ = env
    st = c.get("/api/review/stats").json()
    assert st["pending"] == 4 and st["confirmed"] == st["rejected"] == st["unclear"] == 0 and st["by_type"] == {} and st["by_model"] == {}


def test_the_23_exp020_conclusions_are_imported_as_rejected_and_give_real_stats(env, tmp_path):
    c, ev, _ = env
    src = REPO / "experiments" / "EXP-020-regression-diag" / "reviews_exp020.jsonl"
    recs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 50 and {r["decision"] for r in recs} == {"rejected"} and len({r["row"] for r in recs}) == 23
    assert imp.append_missing(ev / "reviews.jsonl", recs) == 50 and imp.append_missing(ev / "reviews.jsonl", recs) == 0         # run twice: no duplicates
    st = c.get("/api/review/stats").json()
    assert st["rejected"] == 50 and st["imported"] == 50 and st["by_type"]["no_helmet"]["precision"] == 0.0
    per_version = {}
    for r in recs:
        per_version[r["version"]] = per_version.get(r["version"], 0) + 1
    assert per_version == {"v0": 14, "current+old model": 14, "EXP-015 3cls": 7, "EXP-016 4cls": 5, "current+old model @10FPS": 10}
    assert sum(v["rejected"] for k, v in st["by_model"].items() if k.startswith("helmet_best.pt")) == 38           # v0 + current + @10 FPS use the old model
    assert sum(v["rejected"] for k, v in st["by_model"].items() if "EXP-015" in k) == 7


# ---------------------------------------------------------------- panel and gallery
def test_events_carry_their_review_state_and_the_gallery_shows_confirmed_events_only(env):
    c, ev, _ = env
    assert c.get("/api/violations").json() == {"images": []}                           # nothing is a violation before a person says so
    states = {e["id"]: e["review_status"] for e in c.get("/api/events?scope=all").json()["events"]}
    assert set(states.values()) == {"pending"}
    post(c, "vid-r1-e0001", "confirmed")
    post(c, "vid-r1-e0002", "rejected", "wears one")
    states = {e["id"]: (e["review_status"], e["review_note"]) for e in c.get("/api/events?scope=all").json()["events"]}
    assert states["vid-r1-e0001"] == ("confirmed", "") and states["vid-r1-e0002"] == ("rejected", "wears one") and states["vid-r1-e0003"][0] == "pending"
    assert c.get("/api/violations").json() == {"images": ["/static/violations/vid-r1-e0001/plate.jpg"]}


def test_pages_exist_use_text_content_and_show_pending_in_another_colour(env):
    c, _, root = env
    r = c.get("/review")
    assert r.status_code == 200 and "/api/review/next" in r.text and "innerHTML" not in r.text
    for key in ("'c'", "'r'", "'u'", "confirmed", "rejected", "unclear", "note"):
        assert key in r.text
    idx = (root / "templates" / "index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in idx and "s-pending" in idx and "s-confirmed" in idx and "/review" in idx and "confirmed by a reviewer" in idx


# ---------------------------------------------------------------- model stamp in the evidence
def test_events_are_stamped_with_the_models_that_ran(tmp_path):
    from test_evidence import make, raw_frame, records, veh
    rec, _ = make(tmp_path)
    rec.models = MODELS
    rec.update(0, raw_frame(), [veh()], "UNKNOWN")
    rec.flush()
    assert records(tmp_path)[0]["models"] == MODELS


# ---------------------------------------------------------------- export
def synthetic_evidence(tmp_path):
    ev = tmp_path / "ev"
    for i, color in ((1, (40, 200, 40)), (2, (200, 40, 40)), (3, (40, 40, 200))):
        d = ev / f"vid-r1-e000{i}"
        d.mkdir(parents=True)
        img = np.full((720, 1280, 3), 60, np.uint8)
        img[100:260, 100:200] = color
        cv2.imwrite(str(d / "frame.jpg"), img)
    (ev / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in (event(1), event(2, "overload"), event(3, "red_light"))), encoding="utf-8")
    return ev


def write_video(path, n=5):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    for i in range(n):
        f = np.full((720, 1280, 3), 80, np.uint8)
        f[200:400, 300:420] = (0, 200, 200)
        vw.write(f)
    vw.release()


def test_export_writes_yolo_positives_hard_negatives_and_a_manifest_and_skips_unclear(tmp_path):
    ev = synthetic_evidence(tmp_path)
    write_video(tmp_path / "clip.mp4")
    rv = tmp_path / "reviews.jsonl"
    rows = [{"event_id": "vid-r1-e0001", "decision": "confirmed", "event_type": "no_helmet", "model": {"file": "h.pt", "sha256": SHA_H}, "reviewed_at": "t", "track_id": "M_1"},
            {"event_id": "vid-r1-e0002", "decision": "rejected", "event_type": "overload", "model": {"file": "h.pt", "sha256": SHA_H}},
            {"event_id": "vid-r1-e0003", "decision": "unclear", "event_type": "red_light"},
            {"event_id": "vid-r1-e0001", "decision": "confirmed", "event_type": "no_helmet", "note": "later line wins", "model": {"file": "h.pt", "sha256": SHA_H}},
            {"event_id": "exp/x/M_9/no_helmet", "decision": "rejected", "event_type": "no_helmet", "video": "clip.mp4", "frame": 2, "box": [300, 200, 420, 400], "video_id": "clip", "source": "exp020",
             "model": {"file": "old.pt", "sha256": "d" * 64}},
            {"event_id": "ghost", "decision": "confirmed", "event_type": "no_helmet"}]
    rv.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "review_export"
    s = ex.export(rv, ev, out, video_root=tmp_path)
    assert (s["positives"], s["hard_negatives"], s["skipped"]) == (1, 2, {"unclear": 1, "no_image": 1})                  # ghost: no evidence, no video
    img = cv2.imread(str(out / "images" / "positives" / "vid-r1-e0001.jpg"))
    label = (out / "labels" / "positives" / "vid-r1-e0001.txt").read_text(encoding="utf-8").split()
    assert label[0] == "0" and len(label) == 5 and all(0 <= float(v) <= 1 for v in label[1:])
    cx, cy, w, h = (float(v) for v in label[1:])
    H, W = img.shape[:2]
    patch = img[int((cy - h / 2) * H) + 2:int((cy + h / 2) * H) - 2, int((cx - w / 2) * W) + 2:int((cx + w / 2) * W) - 2]
    assert patch.size and (np.abs(patch.astype(int) - np.array([40, 200, 40])).max() < 30)                                 # the label box is exactly the coloured vehicle
    assert (out / "labels" / "hard_negatives" / "vid-r1-e0002.txt").read_text(encoding="utf-8") == ""                      # hard negative = empty label file
    assert (out / "images" / "hard_negatives" / "exp_x_M_9_no_helmet.jpg").exists() and not (out / "images" / "positives" / "vid-r1-e0003.jpg").exists()
    import csv
    manifest = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    by = {m["event_id"]: m for m in manifest}
    assert by["vid-r1-e0001"]["note"] == "later line wins" and by["vid-r1-e0001"]["group"] == "vid" and by["exp/x/M_9/no_helmet"]["group"] == "clip"
    assert by["vid-r1-e0001"]["model_sha256"] == SHA_H and by["exp/x/M_9/no_helmet"]["source"] == "exp020" and by["vid-r1-e0002"]["decision"] == "rejected"
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "Split by video / camera" in readme and "never by crop" in readme and "names:" in (out / "dataset.yaml").read_text(encoding="utf-8")


def test_export_refuses_a_non_empty_folder_and_deletes_nothing(tmp_path):
    ev = synthetic_evidence(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ex.export(tmp_path / "none.jsonl", ev, out)
    assert (out / "keep.txt").read_text(encoding="utf-8") == "x"
    s = ex.export(tmp_path / "none.jsonl", ev, tmp_path / "empty_ok")                   # no reviews: an empty but valid export
    assert s["positives"] == s["hard_negatives"] == 0


def test_export_of_the_committed_exp020_reviews_makes_50_hard_negatives_when_the_video_is_there(tmp_path):
    video = REPO / "test_video.mp4"
    if not video.exists():
        pytest.skip("test_video.mp4 not available")
    s = ex.export(REPO / "experiments" / "EXP-020-regression-diag" / "reviews_exp020.jsonl", tmp_path / "ev", tmp_path / "o")
    assert s["hard_negatives"] == 50 and s["positives"] == 0 and s["groups"] == ["test_video"]
    assert len(list((tmp_path / "o" / "images" / "hard_negatives").glob("*.jpg"))) == 50
