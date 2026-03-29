import pandas as pd
import requests

# Replace this with your actual Vercel endpoint
API_URL = "https://spam-detection-api-lyart.vercel.app/predict"

# Load dataset
df = pd.read_csv("community_spam_dataset.csv")

results = []

for i, row in df.iterrows():
    message_text = str(row["text"])
    actual_label = str(row["label"]).upper()

    payload = {
        "message": message_text
    }

    # If your API expects sender too, use this instead:
    # payload = {
    #     "sender": "test-user",
    #     "message": message_text
    # }

    try:
        response = requests.post(API_URL, json=payload, timeout=20)

        status_code = response.status_code

        if status_code == 200:
            data = response.json()

            # Change "label" below if your API uses another field name
            predicted_label = str(data.get("label", "")).upper()

            is_correct = predicted_label == actual_label

            results.append({
                "text": message_text,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "status_code": status_code,
                "correct": is_correct
            })
        else:
            results.append({
                "text": message_text,
                "actual_label": actual_label,
                "predicted_label": None,
                "status_code": status_code,
                "correct": False
            })

    except Exception as e:
        results.append({
            "text": message_text,
            "actual_label": actual_label,
            "predicted_label": None,
            "status_code": "ERROR",
            "correct": False,
            "error": str(e)
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("api_test_results.csv", index=False)

# Keep only successful API calls
success_df = results_df[results_df["status_code"] == 200]

print(f"Total rows in dataset: {len(df)}")
print(f"Successful API responses (200): {len(success_df)}")
print(f"Failed responses: {len(results_df) - len(success_df)}")

if len(success_df) > 0:
    accuracy = success_df["correct"].mean() * 100
    print(f"Accuracy on successful requests: {accuracy:.2f}%")
else:
    print("No successful 200 responses found.")