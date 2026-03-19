# SSL/TLS 트러블슈팅 가이드 (Anaconda + Windows)

## 증상

```
ERROR: Could not install packages due to an OSError: Could not find a suitable TLS CA certificate bundle, invalid path:
```

`pip install`이나 `conda install` 시 SSL 인증서를 찾지 못해 패키지 설치 실패.

---

## 원인

Anaconda 환경에서 `REQUESTS_CA_BUNDLE`이나 `SSL_CERT_FILE` 환경변수가 비어있거나 잘못된 경로를 가리킴.

확인 방법:
```bash
python -c "import certifi; print(certifi.where())"
# 출력 예: C:\Users\<user>\anaconda3\lib\site-packages\certifi\cacert.pem
```

---

## 해결법

### 방법 1: 명령 실행 시 환경변수 설정 (임시)

```bash
# bash (Git Bash / WSL)
SSL_CERT_FILE="C:/Users/<user>/anaconda3/lib/site-packages/certifi/cacert.pem" \
REQUESTS_CA_BUNDLE="C:/Users/<user>/anaconda3/lib/site-packages/certifi/cacert.pem" \
python -m pip install <package>
```

```powershell
# PowerShell
$env:SSL_CERT_FILE = "C:\Users\<user>\anaconda3\lib\site-packages\certifi\cacert.pem"
$env:REQUESTS_CA_BUNDLE = "C:\Users\<user>\anaconda3\lib\site-packages\certifi\cacert.pem"
python -m pip install <package>
```

### 방법 2: pip 설정으로 영구 적용

```bash
pip config set global.cert "C:/Users/<user>/anaconda3/lib/site-packages/certifi/cacert.pem"
```

또는 `pip.ini` 파일 직접 편집 (`%APPDATA%\pip\pip.ini`):
```ini
[global]
cert = C:/Users/<user>/anaconda3/lib/site-packages/certifi/cacert.pem
```

### 방법 3: conda 설정

```bash
conda config --set ssl_verify C:/Users/<user>/anaconda3/lib/site-packages/certifi/cacert.pem
```

### 방법 4: trusted-host 우회 (보안 낮음, 비추천)

```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org <package>
```

### 방법 5: 시스템 환경변수 영구 설정

1. Windows 설정 → 시스템 → 고급 시스템 설정 → 환경 변수
2. 사용자 변수에 추가:
   - `SSL_CERT_FILE` = `C:\Users\<user>\anaconda3\lib\site-packages\certifi\cacert.pem`
   - `REQUESTS_CA_BUNDLE` = (동일 경로)

---

## 이 프로젝트에서의 적용

현재 프로젝트의 Anaconda 경로:
```
C:\Users\gaeba\anaconda3
```

인증서 경로:
```
C:\Users\gaeba\anaconda3\lib\site-packages\certifi\cacert.pem
```

실행 예:
```bash
SSL_CERT_FILE="C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem" \
REQUESTS_CA_BUNDLE="C:/Users/gaeba/anaconda3/lib/site-packages/certifi/cacert.pem" \
/c/Users/gaeba/anaconda3/python.exe -m pip install <패키지>
```

---

## conda run 주의사항

- `conda run`은 멀티라인 인자를 지원하지 않음 → 한 줄로 작성
- `--no-banner` 플래그는 일부 버전에서 미지원
- 직접 `anaconda3/python.exe`를 호출하는 것이 더 안정적
