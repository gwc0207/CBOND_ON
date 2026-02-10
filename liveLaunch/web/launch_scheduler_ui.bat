@echo off
cd /d C:\Users\BaiYang\CBOND_ON
start "CBOND_ON Scheduler UI" "C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\pythonw.exe" -m cbond_on.run.scheduler_ui
ping 127.0.0.1 -n 2 >nul
start http://127.0.0.1:5002
