from google.oauth2 import service_account
import google.auth.transport.requests
import requests
import os

SA_FILE = os.path.abspath(r'C:\Users\LiveComp\AppData\Roaming\gcloud\gen-lang-client-0529006318-635f9584a30b.json')

credentials = service_account.Credentials.from_service_account_file(
    SA_FILE,
    scopes=['https://www.googleapis.com/auth/generative-language'],
)

# Получаем OAuth токен
auth_req = google.auth.transport.requests.Request()
credentials.refresh(auth_req)
print('===============================')
print('Access token:', credentials.token)
print('===============================')
headers = {
    'Authorization': f'Bearer {credentials.token}',
    'Content-Type': 'application/json',
}

# # Список доступных моделей
# resp = requests.get(
#     'https://generativelanguage.googleapis.com/v1beta/models',
#     headers=headers,
# )
# print('Status:', resp.status_code)
# print('Models:', [m['name'] for m in resp.json().get('models', [])])

# # Тест генерации
resp = requests.post(
    'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent',
    headers=headers,
    json={'contents': [{'parts': [{'text': 'Say hello in one sentence.'}]}]},
)
print('Response:', resp.json())
