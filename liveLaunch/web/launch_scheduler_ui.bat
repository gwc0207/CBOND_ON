@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
set "PYW_EXE=C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\pythonw.exe"
set "APP_MODULE=liveLaunch.web.app"
set "HEALTH_URL=http://127.0.0.1:5002/api/config"
set "UI_URL=http://127.0.0.1:5002"

cd /d "%PROJECT_ROOT%" || exit /b 1

rem If UI is already healthy, do not spawn extra pythonw; just open browser.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%HEALTH_URL%' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% neq 0 (
  powershell -NoProfile -Command "$ps = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'pythonw.exe' -and $_.CommandLine -like '*-m liveLaunch.web.app*' }; $ps | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
  timeout /t 1 >nul
  if exist "%PYW_EXE%" (
    start "CBOND_ON Scheduler UI" "%PYW_EXE%" -m %APP_MODULE%
  ) else (
    where py >nul 2>nul && (
      start "CBOND_ON Scheduler UI" py -3 -m %APP_MODULE%
    ) || (
      start "CBOND_ON Scheduler UI" python -m %APP_MODULE%
    )
  )
  timeout /t 2 >nul
)

start "" "%UI_URL%"
endlocal
