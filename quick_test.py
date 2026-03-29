import pandas as pd
import requests
import time

API_URL = "https://spam-detection-api-lyart.vercel.app/predict"

df = pd.read_csv("community_spam_dataset.csv")

results = []

for i, row in df.iterrows():
    message_text = str(row["text"])
    actual_label = str(row["label"]).upper()

    payload = {
        "message": message_text
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=(5, 60))

        if response.status_code == 200:
            data = response.json()
            predicted_label = str(data.get("label", "")).upper()

            results.append({
                "text": message_text,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "status_code": 200,
                "correct": predicted_label == actual_label
            })
        else:
            results.append({
                "text": message_text,
                "actual_label": actual_label,
                "predicted_label": None,
                "status_code": response.status_code,
                "correct": False
            })

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1} rows")

        time.sleep(0.5)

    except requests.exceptions.ConnectTimeout:
        results.append({
            "text": message_text,
            "actual_label": actual_label,
            "predicted_label": None,
            "status_code": "CONNECT_TIMEOUT",
            "correct": False
        })
    except requests.exceptions.ReadTimeout:
        results.append({
            "text": message_text,
            "actual_label": actual_label,
            "predicted_label": None,
            "status_code": "READ_TIMEOUT",
            "correct": False
        })
    except requests.exceptions.RequestException as e:
        results.append({
            "text": message_text,
            "actual_label": actual_label,
            "predicted_label": None,
            "status_code": "REQUEST_ERROR",
            "correct": False,
            "error": str(e)
        })

results_df = pd.DataFrame(results)
results_df.to_csv("api_test_results.csv", index=False)

success_df = results_df[results_df["status_code"] == 200]

print("Total dataset rows:", len(df))
print("Successful 200 responses:", len(success_df))
print("Failed rows:", len(results_df) - len(success_df))

if len(success_df) > 0:
    accuracy = success_df["correct"].mean() * 100
    print(f"Accuracy on successful responses: {accuracy:.2f}%")