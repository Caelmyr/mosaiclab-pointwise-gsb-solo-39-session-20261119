"""End-to-end checks for the gray-preserving color matrix fit (exploration script)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from rawbench.development import DevelopSettings
from rawbench.storage import Catalog
from rawbench.synthetic import BLUEPRINTS, chart_swatches


def main():
    data_dir = Path(tempfile.mkdtemp(prefix="mosaiclab-e2e-"))
    app = create_app(data_dir)
    client = app.test_client()

    project = client.post("/api/projects", json={"demo": "color-target"}).get_json()
    pid = project["id"]
    frame_id = next(f["id"] for f in project["frames"] if f["role"] == "scene")
    swatches = chart_swatches()

    def inspection(**overrides):
        settings = DevelopSettings().as_dict()
        settings.update(overrides)
        return {"frame_id": frame_id, "settings": settings}

    # 1. Baseline color audit with identity matrix
    audit = client.post(f"/api/projects/{pid}/color-audit", json={
        "inspection": inspection(), "patches": swatches,
    }).get_json()
    print(f"[1] baseline audit: median={audit['median_delta_e']:.3f} max={audit['maximum_delta_e']:.3f}")

    # 2. Matrix fit: built-in neutral references auto-attached for the demo
    fit_resp = client.post(f"/api/projects/{pid}/matrix-fit", json={
        "inspection": inspection(), "patches": swatches,
    })
    assert fit_resp.status_code == 200, fit_resp.get_data(as_text=True)
    payload = fit_resp.get_json()
    report = payload["report"]
    print("[2] decision:", report["decision"], "-", report["reason"])
    assert report["decision"] == "accepted", report["decision"]
    assert report["matrix_replaced"] is True
    assert len(report["custom_matrix"]) == 9
    assert report["saved_recipe"]["title"] == "Fitted color matrix"
    assert payload["settings"]["matrix_name"] == "identity"  # request settings stay untouched
    b, f = report["baseline"]["holdout"], report["final"]["holdout"]
    print(f"    holdout chroma median {b['chroma_median']:.2f} -> {f['chroma_median']:.2f}")
    print(f"    holdout neutral median {b['neutral_median']:.2f} -> {f['neutral_median']:.2f}")
    print(f"    holdout neutral max    {b['neutral_max']:.2f} -> {f['neutral_max']:.2f}")
    assert b["chroma_median"] - f["chroma_median"] >= 0.5
    assert f["neutral_median"] <= b["neutral_median"]
    assert f["neutral_max"] <= b["neutral_max"] + 0.5

    # 3. Recipe persisted and applies the custom matrix
    detail = client.get(f"/api/projects/{pid}").get_json()
    recipes = detail["recipes"]
    assert len(recipes) == 1 and recipes[0]["title"] == "Fitted color matrix"
    saved_settings = recipes[0]["settings"]
    assert saved_settings["matrix_name"] == "custom"
    assert np.allclose(saved_settings["custom_matrix"], report["custom_matrix"], atol=1e-9)
    print("[3] recipe persisted with custom matrix; white balance still",
          saved_settings["white_balance"], "exposure", saved_settings["exposure_ev"])

    # 4. Re-running the audit with the fitted matrix shows the improvement
    fitted = inspection(matrix_name="custom", custom_matrix=report["custom_matrix"])
    audit2 = client.post(f"/api/projects/{pid}/color-audit", json={
        "inspection": fitted, "patches": swatches,
    }).get_json()
    print(f"[4] fitted audit: median={audit2['median_delta_e']:.3f} max={audit2['maximum_delta_e']:.3f}")
    assert audit2["median_delta_e"] < audit["median_delta_e"]
    assert audit2["maximum_delta_e"] < audit["maximum_delta_e"]

    # 5. Neutral card itself improved (the extra neutral regions used only as holdout)
    names = {row["name"]: row for row in audit["patches"]}
    names2 = {row["name"]: row for row in audit2["patches"]}
    for name in ("Patch 13", "Patch 18"):
        e1, e2 = names[name]["delta_e2000"], names2[name]["delta_e2000"]
        print(f"    {name}: {e1:.2f} -> {e2:.2f}")
        assert e2 < e1

    # 6. Degenerate fit (3 chroma patches, no neutrals) must be rejected
    degenerate = client.post(f"/api/projects/{pid}/matrix-fit", json={
        "inspection": inspection(), "patches": swatches[:3],
    })
    assert degenerate.status_code == 400, degenerate.get_data(as_text=True)
    print("[6] too-few patches rejected:", degenerate.get_json()["error"])

    # 7. Explicit empty neutral refs + enough patches: still works using chart grays only
    fit3 = client.post(f"/api/projects/{pid}/matrix-fit", json={
        "inspection": inspection(), "patches": swatches, "neutral_references": [],
    }).get_json()["report"]
    print("[7] chart-only neutral fit decision:", fit3["decision"],
          "holdout neutrals:", [p["name"] for p in fit3["final"]["holdout_patches"] if p["kind"] == "neutral"])

    # 8. Custom explicit neutral references are honored
    custom_refs = [{"name": "Neutral card", "region": {"x": 71, "y": 280, "width": 118, "height": 29},
                   "srgb": [0.517, 0.517, 0.517]}]
    fit4 = client.post(f"/api/projects/{pid}/matrix-fit", json={
        "inspection": inspection(), "patches": swatches, "neutral_references": custom_refs,
    }).get_json()["report"]
    holdout_names = fit4["split"]["holdout"]
    assert "Neutral card (neutral)" in holdout_names
    print("[8] explicit neutral ref placed in holdout:", "Neutral card (neutral)" in holdout_names)

    # 9. Invalid payload fields produce 400s
    bad = client.post(f"/api/projects/{pid}/matrix-fit", json={
        "inspection": inspection(), "patches": swatches, "neutral_references": [{}],
    })
    assert bad.status_code == 400
    print("[9] malformed neutral reference rejected:", bad.get_json()["error"])

    # 10. Re-fitting from the already-fitted matrix yields no further gain:
    # current matrix is kept and no extra recipe is created.
    fit5 = client.post(f"/api/projects/{pid}/matrix-fit", json={
        "inspection": fitted, "patches": swatches,
    }).get_json()["report"]
    detail2 = client.get(f"/api/projects/{pid}").get_json()
    print("[10] second fit decision:", fit5["decision"],
          "recipes now:", len(detail2["recipes"]))
    assert fit5["decision"] == "kept-current"
    assert fit5["matrix_replaced"] is False
    assert len(detail2["recipes"]) == 1

    # 11. A different white balance is preserved by the fit; a new matrix is found.
    wb_settings = inspection(white_balance=(1.08, 1.0, 0.82))
    fit6 = client.post(f"/api/projects/{pid}/matrix-fit", json={
        "inspection": wb_settings, "patches": swatches,
    }).get_json()["report"]
    print("[11] re-fit under new WB decision:", fit6["decision"],
          "WB kept:", fit6["white_balance"], "EV:", fit6["exposure_ev"])
    assert fit6["decision"] == "accepted"
    assert np.allclose(fit6["white_balance"], [1.08, 1.0, 0.82], atol=1e-9)
    assert fit6["custom_matrix"] != report["custom_matrix"]
    detail3 = client.get(f"/api/projects/{pid}").get_json()
    assert detail3["recipes"][-1]["settings"]["white_balance"] == [1.08, 1.0, 0.82]

    # 12. Studio preset as the current matrix: fit must compare against it, not identity.
    studio = inspection(matrix_name="studio")
    fit7 = client.post(f"/api/projects/{pid}/matrix-fit", json={
        "inspection": studio, "patches": swatches,
    }).get_json()["report"]
    print("[12] fit from studio preset decision:", fit7["decision"],
          "baseline holdout chroma:", fit7["baseline"]["holdout"]["chroma_median"])
    assert fit7["decision"] == "accepted"

    print("\nALL E2E CHECKS PASSED")


if __name__ == "__main__":
    main()
