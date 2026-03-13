Set WshShell = CreateObject("WScript.Shell")
cmd = "cmd /c ""cd /d C:\Users\BaiYang\CBOND_ON\liveLaunch\web && launch_scheduler_ui.bat"""
WshShell.Run cmd, 0, False
