import os
import requests

token = os.getenv("GITHUB_TOKEN")

if not token:
    raise ValueError("GITHUB_TOKEN is not set")

response = requests.get(
    "https://models.inference.ai.azure.com/models",
    headers={"Authorization": f"Bearer {token}"},
)

response.raise_for_status()

for model in response.json():
    print(model["id"])