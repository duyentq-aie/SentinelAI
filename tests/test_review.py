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


# ---------------------------------------------------------------- the head box and what a decision means
def test_recorder_stores_the_triggering_head_box_and_class_of_a_no_helmet_event_and_null_otherwise(tmp_path):
    from test_evidence import make, raw_frame, records, veh
    rec, _ = make(tmp_path)
    v = veh("M_1", "NO-HELMET")
    v.heads = [((10, 20, 30, 40), "helmet"), ((100, 110, 140, 160), "no-helmet"), ((50, 50, 60, 60), "no-helmet")]
    rec.update(0, raw_frame(), [v], "UNKNOWN")
    o = veh("M_2", "OVERLOAD")
    o.heads = [((1, 1, 5, 5), "no-helmet")]
    rec.update(0, raw_frame(), [o], "UNKNOWN")
    rec.flush()
    by = {r["track_id"]: r for r in records(tmp_path)}
    assert by["M_1"]["head_box"] == [100, 110, 140, 160] and by["M_1"]["head_class"] == "no-helmet"          # the biggest no-helmet head
    assert by["M_2"]["head_box"] is None and by["M_2"]["head_class"] is None                                   # overload: no head label
    rec2, _ = make(tmp_path / "second")
    rec2.update(0, raw_frame(), [veh("M_3", "NO-HELMET")], "UNKNOWN")                                         # no head attached this frame: null, not a guess
    rec2.flush()
    assert records(tmp_path / "second")[0]["head_box"] is None


def test_the_rules_hand_the_attached_heads_to_the_vehicle():
    from sentinel.config import load_camera_config, load_pipeline_config
    from sentinel.perception import Perception, Track
    from sentinel.rules.windowed import make_rules
    cfg, cam = load_pipeline_config(), load_camera_config(REPO / "configs" / "cameras" / "test_video.yaml")
    r = make_rules(cfg.rules, cam)
    res = r.step(Perception(helmet=[Track(1, "motorcycle", (100, 100, 300, 400), 0.9), Track(2, "no-helmet", (150, 110, 200, 170), 0.9), Track(3, "helmet", (500, 10, 520, 30), 0.9)]))
    m = [v for v in res.vehicles if v.id == "M_1"][0]
    assert m.heads == [((150, 110, 200, 170), "no-helmet")]


def test_corrected_label_semantics():
    from apps.web.review import corrected_label
    assert corrected_label("no_helmet", "confirmed") == "no-helmet"
    assert corrected_label("no_helmet", "rejected") == "helmet"                                   # the model was wrong: the head wears a helmet, NOT background
    assert corrected_label("no_helmet", "rejected", not_head=True) == "background"
    assert corrected_label("no_helmet", "unclear") is None
    assert corrected_label("overload", "confirmed") is None and corrected_label("red_light", "rejected") is None


def test_review_records_the_corrected_label_and_not_head_is_only_for_a_rejected_no_helmet(env):
    c, ev, _ = env
    assert post(c, "vid-r1-e0001", "confirmed").status_code == 200
    assert post(c, "vid-r1-e0002", "rejected").status_code == 200                                      # overload
    r = c.post("/api/review/vid-r1-e0003", json={"decision": "unclear"})
    assert r.status_code == 200
    got = {l["event_id"]: l for l in lines(ev)}
    assert got["vid-r1-e0001"]["corrected_label"] == "no-helmet" and got["vid-r1-e0001"]["not_head"] is False
    assert got["vid-r1-e0002"]["corrected_label"] is None and got["vid-r1-e0003"]["corrected_label"] is None
    assert c.post("/api/review/vid-r1-e0004", json={"decision": "rejected", "not_head": True}).status_code == 200            # e4: no_helmet
    assert lines(ev)[-1]["corrected_label"] == "background" and lines(ev)[-1]["not_head"] is True
    n = len(lines(ev))
    assert c.post("/api/review/vid-r1-e0001", json={"decision": "confirmed", "not_head": True}).status_code == 422             # not_head needs a rejection
    assert c.post("/api/review/vid-r1-e0002", json={"decision": "rejected", "not_head": True}).status_code == 422             # ... of a no_helmet event
    assert len(lines(ev)) == n
    st = c.get("/api/review/stats").json()
    assert st["corrected_labels"] == {"no-helmet": 1, "helmet": 0, "background": 1}


def test_review_page_offers_the_not_a_head_option_for_no_helmet_events():
    html = (REPO / "templates" / "review.html").read_text(encoding="utf-8")
    assert "not a head" in html and "not_head" in html and "it wears a helmet" in html and "innerHTML" not in html


def test_the_exp020_import_reconstructs_the_head_from_the_fixtures_and_marks_it():
    recs = [json.loads(l) for l in (REPO / "experiments" / "EXP-020-regression-diag" / "reviews_exp020.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 50 and {r["import_rev"] for r in recs} == {2}
    nh = [r for r in recs if r["event_type"] == "no_helmet"]
    assert len(nh) == 46 and all(r["head_box"] and r["corrected_label"] == "helmet" and r["head_class"] == "no-helmet" for r in nh)
    methods = {r["head_reconstructed"]["method"].split(":")[0] for r in nh}
    assert methods == {"fixture", "model_rerun"} and all(r["head_reconstructed"]["frame"] == r["head_frame"] for r in nh)
    ov = [r for r in recs if r["event_type"] == "overload"]
    assert len(ov) == 4 and all(r["head_box"] is None and r["corrected_label"] is None for r in ov)
    for r in nh:                                                                                       # the head lies inside the vehicle box of the event
        hx, hy = (r["head_box"][0] + r["head_box"][2]) / 2, (r["head_box"][1] + r["head_box"][3]) / 2
        assert r["box"][0] <= hx <= r["box"][2] and r["box"][1] <= hy <= r["box"][3]


def test_head_from_perceptions_prefers_the_event_frame_then_the_closest_one_and_ignores_other_boxes():
    from sentinel.perception import Perception, Track
    P = [Perception(helmet=[]) for _ in range(10)]
    P[4].helmet = [Track(1, "no-helmet", (110, 110, 150, 150), 0.6), Track(2, "helmet", (110, 110, 150, 150), 0.9), Track(3, "no-helmet", (900, 900, 950, 950), 0.99)]
    P[6].helmet = [Track(4, "no-helmet", (112, 112, 152, 152), 0.9)]
    assert imp.head_from_perceptions(P, 4, (100, 100, 300, 300)) == ([110, 110, 150, 150], 4)         # event frame wins; helmet class and far boxes ignored
    assert imp.head_from_perceptions(P, 5, (100, 100, 300, 300)) == ([110, 110, 150, 150], 4)         # equally close (4 and 6): the earlier one
    assert imp.head_from_perceptions(P, 9, (100, 100, 300, 300)) == ([112, 112, 152, 152], 6)         # +-3 frames
    assert imp.head_from_perceptions(P, 0, (100, 100, 300, 300)) is None


# ---------------------------------------------------------------- export
def synthetic_evidence(tmp_path, with_head=True):
    ev = tmp_path / "ev"
    heads = {1: [120, 110, 160, 150], 2: [130, 120, 170, 160], 3: [125, 115, 165, 155], 4: None, 5: [140, 130, 180, 170]}
    recs = []
    for i in (1, 2, 3, 4, 5):
        d = ev / f"vid-r1-e000{i}"
        d.mkdir(parents=True)
        img = np.full((720, 1280, 3), 60, np.uint8)
        img[100:260, 100:200] = (40, 200, 40)
        cv2.imwrite(str(d / "frame.jpg"), img)
        e = event(i, "overload" if i == 3 else "no_helmet", head_box=heads[i] if with_head else None, head_class="no-helmet" if heads[i] and with_head else None)
        e["video_id"] = f"vid{i % 2}"
        recs.append(e)
    (ev / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in recs), encoding="utf-8")
    return ev


def write_video(path, n=8):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    for i in range(n):
        f = np.full((720, 1280, 3), 80, np.uint8)
        f[200:400, 300:420] = (0, 200, 200)
        vw.write(f)
    vw.release()


def rv_lines(tmp_path, rows):
    p = tmp_path / "reviews.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def label_of(out, name):
    return (out / "labels" / f"{name}.txt").read_text(encoding="utf-8")


def test_rejected_no_helmet_is_exported_with_the_helmet_label_confirmed_with_no_helmet_on_the_whole_frame(tmp_path):
    import csv
    ev = synthetic_evidence(tmp_path)
    rows = [{"event_id": "vid-r1-e0001", "decision": "rejected", "event_type": "no_helmet", "model": {"file": "h.pt", "sha256": SHA_H}, "track_id": "M_1"},
            {"event_id": "vid-r1-e0002", "decision": "confirmed", "event_type": "no_helmet", "model": {"file": "h.pt", "sha256": SHA_H}}]
    out = tmp_path / "export"
    s = ex.export(rv_lines(tmp_path, rows), ev, out)
    assert (s["images"], s["helmet"], s["no-helmet"]) == (2, 1, 1)
    img = cv2.imread(str(out / "images" / "vid-r1-e0001.jpg"))
    assert img.shape[:2] == (720, 1280)                                                                 # the whole frame, not a vehicle crop
    cls, cx, cy, w, h = label_of(out, "vid-r1-e0001").split()
    assert cls == "0" and (float(cx), float(cy), float(w), float(h)) == pytest.approx(((120 + 160) / 2 / 1280, (110 + 150) / 2 / 720, 40 / 1280, 40 / 720), abs=1e-5)       # rejected = helmet
    assert label_of(out, "vid-r1-e0002").split()[0] == "2"                                              # confirmed = no-helmet
    m = {r["event_id"]: r for r in csv.DictReader(open(out / "manifest.csv", encoding="utf-8"))}
    assert m["vid-r1-e0001"]["label"] == "helmet" and m["vid-r1-e0001"]["head_source"] == "event" and m["vid-r1-e0001"]["partial_labels"] == "yes" and m["vid-r1-e0001"]["group"] == "vid1"
    assert m["vid-r1-e0002"]["label"] == "no-helmet" and m["vid-r1-e0002"]["model_sha256"] == SHA_H
    assert "names:" in (out / "dataset.yaml").read_text(encoding="utf-8")


def test_not_a_head_is_listed_as_background_and_never_exported_as_an_empty_label(tmp_path):
    import csv
    ev = synthetic_evidence(tmp_path)
    rows = [{"event_id": "vid-r1-e0001", "decision": "rejected", "event_type": "no_helmet", "not_head": True}]
    out = tmp_path / "export"
    s = ex.export(rv_lines(tmp_path, rows), ev, out)
    assert (s["images"], s["background"]) == (0, 1)
    assert not list((out / "labels").iterdir()) and not list((out / "images").iterdir())                # no image, no (empty) label file
    assert [r["event_id"] for r in csv.DictReader(open(out / "background.csv", encoding="utf-8"))] == ["vid-r1-e0001"]


def test_unclear_is_not_exported_and_events_without_a_head_box_go_to_needs_head_box(tmp_path):
    import csv
    ev = synthetic_evidence(tmp_path)
    rows = [{"event_id": "vid-r1-e0001", "decision": "unclear", "event_type": "no_helmet"},
            {"event_id": "vid-r1-e0004", "decision": "rejected", "event_type": "no_helmet"},                # e4 has head_box null
            {"event_id": "vid-r1-e0005", "decision": "confirmed", "event_type": "no_helmet"}]
    out = tmp_path / "export"
    s = ex.export(rv_lines(tmp_path, rows), ev, out)
    assert s["skipped"]["unclear"] == 1 and (s["images"], s["needs_head_box"]) == (1, 1)
    assert [p.name for p in (out / "images").iterdir()] == ["vid-r1-e0005.jpg"] and not (out / "labels" / "vid-r1-e0001.txt").exists() and not (out / "labels" / "vid-r1-e0004.txt").exists()
    nh = list(csv.DictReader(open(out / "needs_head_box.csv", encoding="utf-8")))
    assert [(r["event_id"], r["corrected_label"]) for r in nh] == [("vid-r1-e0004", "helmet")]


def test_overload_is_metadata_only_and_has_no_detector_label(tmp_path):
    import csv
    ev = synthetic_evidence(tmp_path)
    rows = [{"event_id": "vid-r1-e0003", "decision": "confirmed", "event_type": "overload", "model": {"file": "h.pt", "sha256": SHA_H}},
            {"event_id": "vid-r1-e0002", "decision": "rejected", "event_type": "red_light"}]
    out = tmp_path / "export"
    s = ex.export(rv_lines(tmp_path, rows), ev, out)
    assert (s["images"], s["metadata_only"]) == (0, 2) and not list((out / "labels").iterdir()) and not list((out / "images").iterdir())
    assert {r["event_id"]: r["event_type"] for r in csv.DictReader(open(out / "metadata_only.csv", encoding="utf-8"))} == {"vid-r1-e0003": "overload", "vid-r1-e0002": "red_light"}


def test_no_exported_image_ever_has_an_empty_label_file(tmp_path):
    ev = synthetic_evidence(tmp_path)
    rows = [{"event_id": f"vid-r1-e000{i}", "decision": d, "event_type": "overload" if i == 3 else "no_helmet", "not_head": nh}
            for i, d, nh in ((1, "rejected", False), (2, "confirmed", False), (3, "rejected", False), (4, "rejected", False), (5, "rejected", True))]
    out = tmp_path / "export"
    ex.export(rv_lines(tmp_path, rows), ev, out)
    for img in (out / "images").iterdir():
        lab = out / "labels" / (img.stem + ".txt")
        assert lab.exists() and lab.read_text(encoding="utf-8").strip() != ""
    assert sorted(p.stem for p in (out / "images").iterdir()) == sorted(p.stem for p in (out / "labels").iterdir())


def test_the_same_head_labelled_two_ways_is_a_conflict_and_is_not_exported(tmp_path):
    write_video(tmp_path / "clip.mp4")
    base = {"decision": "rejected", "event_type": "no_helmet", "video": "clip.mp4", "video_id": "clip", "source": "exp020", "corrected_label": "helmet", "not_head": False,
            "box": [300, 200, 420, 400], "head_box": [330, 210, 370, 250], "head_frame": 3, "frame": 3, "head_reconstructed": {"method": "fixture:t.jsonl", "frame": 3, "event_frame": 3}}
    rows = [{**base, "event_id": "exp020/v0/M_1/no_helmet"}, {**base, "event_id": "exp020/exp015/M_9/no_helmet", "head_box": [332, 211, 371, 251], "head_frame": 4, "frame": 4},
            {**base, "event_id": "exp020/exp016/M_5/no_helmet", "head_box": None, "head_frame": None, "head_reconstructed": None},
            {**base, "event_id": "exp020/x/M_6/no_helmet", "head_box": [331, 211, 371, 251], "head_frame": 5, "frame": 5, "decision": "confirmed", "corrected_label": "no-helmet"}]
    out = tmp_path / "export"
    s = ex.export(rv_lines(tmp_path, rows), tmp_path / "none", out, video_root=tmp_path)
    # v0 + exp015 agree (helmet): the second is a duplicate; the 'confirmed' row disagrees about the same head -> a conflict, and NOTHING is exported for that head
    assert (s["images"], s["duplicates"], s["conflicts"], s["needs_head_box"]) == (0, 1, 2, 1)
    assert not list((out / "images").iterdir()) and not list((out / "labels").iterdir())


def test_imported_events_without_a_conflict_are_exported_from_the_video_with_the_head_source(tmp_path):
    import csv
    write_video(tmp_path / "clip.mp4")
    base = {"decision": "rejected", "event_type": "no_helmet", "video": "clip.mp4", "video_id": "clip", "source": "exp020", "corrected_label": "helmet", "not_head": False,
            "box": [300, 200, 420, 400], "head_box": [330, 210, 370, 250], "head_frame": 3, "frame": 3, "head_reconstructed": {"method": "fixture:t.jsonl", "frame": 3, "event_frame": 3}}
    rows = [{**base, "event_id": "exp020/v0/M_1/no_helmet"}, {**base, "event_id": "exp020/exp015/M_9/no_helmet", "head_box": [332, 211, 371, 251], "head_frame": 4, "frame": 4},
            {**base, "event_id": "exp020/exp016/M_5/no_helmet", "head_box": None, "head_frame": None, "head_reconstructed": None}]
    out = tmp_path / "export"
    s = ex.export(rv_lines(tmp_path, rows), tmp_path / "none", out, video_root=tmp_path)
    assert (s["images"], s["duplicates"], s["needs_head_box"], s["conflicts"]) == (1, 1, 1, 0)
    m = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert m[0]["event_id"] == "exp020/exp015/M_9/no_helmet" and m[0]["head_source"]   # event ids are processed in sorted order: the first one is kept == "reconstructed:fixture:t.jsonl" and m[0]["label"] == "helmet" and m[0]["group"] == "clip"
    assert cv2.imread(str(out / "images" / "exp020_exp015_M_9_no_helmet.jpg")).shape[:2] == (720, 1280)
    assert label_of(out, "exp020_exp015_M_9_no_helmet").split()[0] == "0"
    assert [r["event_id"] for r in csv.DictReader(open(out / "duplicates.csv", encoding="utf-8"))] == ["exp020/v0/M_1/no_helmet"]


def test_export_refuses_a_non_empty_folder_and_deletes_nothing(tmp_path):
    ev = synthetic_evidence(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ex.export(tmp_path / "none.jsonl", ev, out)
    assert (out / "keep.txt").read_text(encoding="utf-8") == "x"
    s = ex.export(tmp_path / "none.jsonl", ev, tmp_path / "empty_ok")                   # no reviews: an empty but valid export
    assert s["images"] == 0


def test_the_committed_exp020_reviews_export_with_every_no_helmet_as_a_helmet_label(tmp_path):
    video = REPO / "test_video.mp4"
    if not video.exists():
        pytest.skip("test_video.mp4 not available")
    s = ex.export(REPO / "experiments" / "EXP-020-regression-diag" / "reviews_exp020.jsonl", tmp_path / "ev", tmp_path / "o")
    assert s["no-helmet"] == 0 and s["helmet"] == s["images"] > 0 and s["metadata_only"] == 4 and s["needs_head_box"] == 0 and s["background"] == 0
    assert s["images"] + s["duplicates"] == 46                                                             # every reconstructed head is either an image or a duplicate of one
    for lab in (tmp_path / "o" / "labels").iterdir():
        assert lab.read_text(encoding="utf-8").split()[0] == "0"
