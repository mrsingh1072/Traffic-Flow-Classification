"""Real-API contract checks for every frontend workflow. Uses synthetic test inputs only."""
import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
checks = 0

def request(path, payload=None):
    req = urllib.request.Request(BASE + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:5173"})
    with urllib.request.urlopen(req, timeout=60) as response:
        assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
        return json.load(response)

def check(condition, message):
    global checks
    assert condition, message
    checks += 1

catalog = request("/datasets")["datasets"]
check(len(catalog) == 5, "Five configured datasets")
for item in catalog:
    ds = item["dataset"]
    schema = request("/schema?dataset=" + ds)
    check(schema["required_feature_count"] == len(schema["features"]), ds + " schema count")
    record = {key: 0 if kind == "numeric" else None for key, kind in schema["features"].items()}
    single = request("/predict", {"dataset": ds, "features": record})
    batch = request("/predict/batch", {"dataset": ds, "records": [record, record]})
    check(batch["count"] == 2 and len(batch["predictions"]) == 2, ds + " batch length")
    check(all(p == single for p in batch["predictions"]), ds + " batch matches single inference")
    check(single["predicted_class"] in single["probabilities"], ds + " class in distribution")
    check(abs(sum(single["probabilities"].values()) - 1) < 0.0001, ds + " normalized softmax")
    metric = request("/metrics?dataset=" + ds)["metrics"][0]
    classes = request("/metrics/" + ds + "/classes")["classes"]
    matrix = request("/metrics/" + ds + "/confusion-matrix")
    check(set(matrix["labels"]) == {c["class_name"] for c in classes}, ds + " label consistency")
    check(len(matrix["matrix"]) == len(matrix["labels"]) and all(len(r) == len(matrix["labels"]) for r in matrix["matrix"]), ds + " square matrix")
    check(sum(sum(r) for r in matrix["matrix"]) == metric["test_samples"], ds + " matrix sample count")
    check(sum(c["support"] for c in classes) == metric["test_samples"], ds + " class support")
    print(f"PASS | {ds}: {schema['required_feature_count']} raw features, {len(classes)} classes; single, batch, metrics, matrix, CORS")

for records in ([], [{}] * 101):
    try:
        request("/predict/batch", {"dataset": catalog[0]["dataset"], "records": records})
        raise AssertionError("Invalid batch size accepted")
    except urllib.error.HTTPError as exc:
        check(exc.code == 422, "Batch size validation")
print(f"{checks}/{checks} contract checks passed.")
