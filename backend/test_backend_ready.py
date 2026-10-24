import json
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
DATASETS = ["UNSW_NB15", "ISCX_15s", "ISCX_30s", "ISCX_60s", "ISCX_120s"]


def request(method, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode())
        except Exception:
            body = {}
        return exc.code, body


def main():
    checks = []

    status, body = request("GET", "/health")
    checks.append(("health", status == 200 and body.get("status") == "healthy"))

    status, body = request("GET", "/datasets")
    rows = body.get("datasets", [])
    checks.append(("all model/preprocessor files", status == 200 and len(rows) == 5 and all(
        r.get("model_available") and r.get("preprocessor_available") for r in rows
    )))

    status, body = request("GET", "/metrics")
    checks.append(("dashboard metrics", status == 200 and len(body.get("metrics", [])) == 5))

    for dataset in DATASETS:
        status, schema = request("GET", f"/schema?dataset={dataset}")
        checks.append((f"schema {dataset}", status == 200 and bool(schema.get("features"))))
        if status == 200:
            features = {
                name: (0.0 if kind == "numeric" else None)
                for name, kind in schema["features"].items()
            }
            status, result = request("POST", "/predict", {
                "dataset": dataset, "features": features
            })
            checks.append((
                f"prediction {dataset}",
                status == 200 and result.get("dataset") == dataset
                and bool(result.get("predicted_class"))
            ))

    status, _ = request("GET", "/schema?dataset=INVALID")
    checks.append(("reject invalid dataset schema", status == 400))

    status, _ = request("POST", "/predict", {
        "dataset": "INVALID", "features": {"x": 1}
    })
    checks.append(("reject invalid prediction dataset", status == 400))

    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} | {name}")
    print(f"\\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed.")
    if not all(ok for _, ok in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
