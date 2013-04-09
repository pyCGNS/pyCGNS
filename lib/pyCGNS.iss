; -- UninstallCodeExample1.iss --
;
; This script shows various things you can achieve using a [Code] section for Uninstall

[Setup]
AppName=pyCGNS
AppVersion=4.3
DefaultDirName={pf}\pyCGNS
DefaultGroupName=pyCGNS
UninstallDisplayIcon={app}\CGNS.exe
OutputDir="..\build"
SetupIconFile="..\doc\images\pyCGNS-logo-large.ico"
PrivilegesRequired=none
Compression=lzma2
SolidCompression=yes

[Files]
Source: "..\build\exe.win32-2.7\*"; DestDir: "{app}"

[Run]
Filename: "{app}\CGNS.exe"; Description: "Launch My Program"; Flags: nowait postinstall skipifsilent
