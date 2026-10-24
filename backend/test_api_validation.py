
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"


def request_json(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"detail": body}
        return exc.code, parsed


def check(name, expected_status, path, payload=None):
    status, body = request_json(path, payload)

    if status == expected_status:
        print(f"PASS | {name} | HTTP {status}")
        return True

    print(
        f"FAIL | {name} | expected HTTP {expected_status}, "
        f"received HTTP {status} | {body}"
    )
    return False


def main():
    passed = 0
    total = 0

    def run(name, expected_status, path, payload=None):
        nonlocal passed, total
        total += 1
        if check(name, expected_status, path, payload):
            passed += 1

    # 1. Health endpoint
    status, health = request_json("/health")
    total += 1
    if status == 200 and health.get("status") == "healthy":
        print("PASS | Health endpoint | HTTP 200")
        passed += 1
    else:
        print(f"FAIL | Health endpoint | HTTP {status} | {health}")

    # 2. Dataset artifact availability
    status, data = request_json("/datasets")
    total += 1
    datasets = data.get("datasets", []) if status == 200 else []
    artifacts_ok = (
        len(datasets) == 5
        and all(
            item.get("preprocessor_available")
            and item.get("model_available")
            for item in datasets
        )
    )
    if artifacts_ok:
        print("PASS | All five dataset artifacts | HTTP 200")
        passed += 1
    else:
        print(f"FAIL | Dataset artifacts | HTTP {status} | {data}")

    # 3. Invalid dataset in schema endpoint
    run(
        "Unknown dataset in schema",
        400,
        "/schema?dataset=NOT_A_DATASET",
    )

    # 4. Unknown dataset in prediction endpoint
    run(
        "Unknown dataset in prediction",
        400,
        "/predict",
        {"dataset": "NOT_A_DATASET", "features": {"x": 1}},
    )

    # Build a valid schema-shaped input for UNSW-NB15.
    status, schema = request_json("/schema?dataset=UNSW_NB15")
    if status != 200:
        raise RuntimeError(f"Could not load schema: {schema}")

    feature_types = schema["features"]
    valid_features = {
        name: (0.0 if kind == "numeric" else None)
        for name, kind in feature_types.items()
    }

    # 5. Missing feature must be rejected.
    missing_features = valid_features.copy()
    missing_features.pop(next(iter(missing_features)))

    run(
        "Missing required feature",
        422,
        "/predict",
        {"dataset": "UNSW_NB15", "features": missing_features},
    )

    # 6. Unknown feature must be rejected.
    unknown_features = valid_features.copy()
    unknown_features["unexpected_feature"] = 1

    run(
        "Unknown feature name",
        422,
        "/predict",
        {"dataset": "UNSW_NB15", "features": unknown_features},
    )

    # 7. All-null input must be rejected.
    null_features = {
        name: None for name in feature_types
    }

    run(
        "All-null input",
        422,
        "/predict",
        {"dataset": "UNSW_NB15", "features": null_features},
    )

    # 8. Valid request must still work after invalid requests.
    run(
        "Valid prediction after validation errors",
        200,
        "/predict",
        {"dataset": "UNSW_NB15", "features": valid_features},
    )

    print(f"\nValidation result: {passed}/{total} checks passed.")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
