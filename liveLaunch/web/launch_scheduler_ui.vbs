Set WshShell = CreateObject("WScript.Shell")
cmd = "cmd /c ""cd /d C:\Users\BaiYang\CBOND_ON && C:\Users\BaiYang\AppData\Local\Programs\Python\Python311\pythonw.exe -m cbond_on.run.scheduler_ui"""
WshShell.Run cmd, 0, False
WScript.Sleep 1200
WshShell.Run "http://127.0.0.1:5002", 1, False
