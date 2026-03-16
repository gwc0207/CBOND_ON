Option Explicit

Dim WshShell, Fso
Dim scriptDir, batPath, cmd

Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")

scriptDir = Fso.GetParentFolderName(WScript.ScriptFullName)
batPath = scriptDir & "\launch_scheduler_ui.bat"

If Not Fso.FileExists(batPath) Then
  MsgBox "launch_scheduler_ui.bat not found: " & batPath, 16, "CBOND_ON Scheduler UI"
  WScript.Quit 1
End If

cmd = "cmd /c """ & batPath & """"
WshShell.Run cmd, 0, False
