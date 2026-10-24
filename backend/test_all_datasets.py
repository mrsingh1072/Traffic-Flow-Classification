
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

DATASETS = [
    "UNSW_NB15",
    "ISCX_15s",
    "ISCX_30s",
    "ISCX_60s",
    "ISCX_120s",
]


def get_json(url):
    request = urllib.request.Request(
        url, headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("Testing prediction API for all five datasets\n")
    passed = 0

    for dataset in DATASETS:
        try:
            schema = get_json(
                f"{BASE_URL}/schema?dataset={dataset}"
            )

            # Synthetic smoke-test input, not a real traffic record.
            features = {
                name: (
                    0.0 if kind == "numeric" else None
                )
                for name, kind in schema["features"].items()
            }

            result = post_json(
                f"{BASE_URL}/predict",
                {
                    "dataset": dataset,
                    "features": features,
                },
            )

            if (
                result.get("dataset") != dataset
                or not result.get("predicted_class")
                or not isinstance(result.get("probabilities"), dict)
            ):
                raise RuntimeError("Unexpected prediction response.")

            probability_sum = sum(result["probabilities"].values())

            if abs(probability_sum - 1.0) > 0.01:
                raise RuntimeError(
                    f"Probabilities do not sum to 1: {probability_sum}"
                )

            print(
                f"PASS | {dataset:12s} | "
                f"Prediction: {result['predicted_class']} | "
                f"Confidence: {result['confidence']:.4f}"
            )
            passed += 1

        except Exception as exc:
            print(f"FAIL | {dataset}: {exc}")

            if isinstance(exc, urllib.error.HTTPError):
                print(exc.read().decode("utf-8"))

    print(f"\nResult: {passed}/{len(DATASETS)} datasets passed.")

    if passed != len(DATASETS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
