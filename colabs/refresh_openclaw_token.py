"""
Refreshes Google SA token for openclaw google-gemini-cli provider.
Updates auth.json and auth-profiles.json with a fresh Bearer token.
Run every ~50 minutes (e.g. via Windows Task Scheduler).
"""
from google.oauth2 import service_account
import google.auth.transport.requests
import json
import time
import os

SA_FILE = r'C:\Users\LiveComp\AppData\Roaming\gcloud\gen-lang-client-0529006318-635f9584a30b.json'
PROJECT_ID = 'gen-lang-client-0529006318'
PROFILE_KEY = 'google-gemini-cli:andrey@veriga.ru'

AUTH_JSON      = r'C:\Users\LiveComp\.openclaw\agents\main\agent\auth.json'
PROFILES_JSON  = r'C:\Users\LiveComp\.openclaw\agents\main\agent\auth-profiles.json'

# SA token expires in 3600s; we set expires 3500s ahead so openclaw
# doesn't try to self-refresh (which would use the old OAuth refresh token)
EXPIRE_SECONDS = 3500


def get_sa_token():
    credentials = service_account.Credentials.from_service_account_file(
        SA_FILE,
        scopes=['https://www.googleapis.com/auth/generative-language'],
    )
    req = google.auth.transport.requests.Request()
    credentials.refresh(req)
    return credentials.token


def patch_json(path, patcher):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    patcher(data)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def main():
    token = get_sa_token()
    expires_ms = int((time.time() + EXPIRE_SECONDS) * 1000)
    print(f'New token: {token[:20]}... expires_ms={expires_ms}')

    # auth.json — упрощённый формат (ключ без email)
    def patch_auth(data):
        entry = data.get('google-gemini-cli', {})
        entry['access'] = token
        entry['expires'] = expires_ms
        # type и refresh не трогаем
        data['google-gemini-cli'] = entry

    # auth-profiles.json — полный формат с projectId
    def patch_profiles(data):
        profiles = data.setdefault('profiles', {})
        entry = profiles.get(PROFILE_KEY, {})
        entry['access'] = token
        entry['expires'] = expires_ms
        entry['projectId'] = PROJECT_ID
        profiles[PROFILE_KEY] = entry

    patch_json(AUTH_JSON, patch_auth)
    patch_json(PROFILES_JSON, patch_profiles)

    print('Done. Token valid for ~58 minutes.')


if __name__ == '__main__':
    main()
