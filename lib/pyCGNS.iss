; #  -------------------------------------------------------------------------
; #  pyCGNS.NAV - Python package for CFD General Notation System - NAVigater
; #  See license.txt file in the root directory of this Python module source  
; #  -------------------------------------------------------------------------

; Inno Setup Script - to be run into the pyCGNS/lib directory

[Setup]
AppName=pyCGNS
AppVersion=4.3
DefaultDirName={src}\pyCGNS
DefaultGroupName=pyCGNS
UninstallDisplayIcon={app}\CGNSNAV.exe
OutputDir="..\build"
SetupIconFile=pyCGNS-small.ico
PrivilegesRequired=none
Compression=lzma2
SolidCompression=yes
WizardImageFile=pyCGNS.bmp
WizardImageStretch=no
WizardSmallImageFile=pyCGNS-small.bmp
LicenseFile=..\license.txt

[Files]
Source: "..\build\exe.win32-2.7\*"; DestDir: "{app}"
Source: "..\build\exe.win32-2.7\lib\*"; DestDir: "{app}"
Source: "..\build\exe.win32-2.7\demo\*"; DestDir: "{app}"

[Run]
Filename: "{app}\CGNSNAV.exe"; Description: "Launch CGNS.NAV"; Flags: nowait postinstall skipifsilent

[Icons]
Name: {userdesktop}\pyCGNS; FileName: {app}\CGNSNAV.exe; WorkingDir: {app}; IconFilename:  {app}\pyCGNS-small.ico

; --- last line