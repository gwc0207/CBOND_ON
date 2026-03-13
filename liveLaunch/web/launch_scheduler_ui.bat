@echo off
setlocal
cd /d C:\Users\BaiYang\CBOND_ON

rem If UI is already healthy, do not spawn extra pythonw; just open browser.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:5002/api/config' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% neq 0 (
  powershell -NoProfile -Command "$ps = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'pythonw.exe' -and $_.CommandLine -like '*-m liveLaunch.web.app*' }; $ps | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
  timeout /t 1 >nul
  start "CBOND_ON Scheduler UI" "C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\pythonw.exe" -m liveLaunch.web.app
  timeout /t 2 >nul
)

start "" http://127.0.0.1:5002
endlocal
