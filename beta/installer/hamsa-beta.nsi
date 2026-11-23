; Hamsa Beta installer (NSIS). Build with:
;   makensis beta\installer\hamsa-beta.nsi
; after `cargo build -p hamsa-beta --release` from the hamsa folder.
; Installs for the current user only (no admin prompt). Settings, models and history stay in
; %LOCALAPPDATA%\app.hamsa.desktop, shared with the regular Hamsa, and are kept on uninstall.

Unicode true
!include "MUI2.nsh"

!define NAME "Hamsa Beta"
!define VERSION "0.1.0"
!define EXE "hamsa-beta.exe"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\HamsaBeta"

Name "${NAME}"
OutFile "..\..\target\release\Hamsa_Beta_${VERSION}_x64-setup.exe"
InstallDir "$LOCALAPPDATA\Programs\Hamsa Beta"
InstallDirRegKey HKCU "Software\Hamsa Beta" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
BrandingText "${NAME} ${VERSION}"

!define MUI_ICON "..\assets\app-icon.ico"
!define MUI_UNICON "..\assets\app-icon.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\${EXE}"
!define MUI_FINISHPAGE_RUN_TEXT "Start Hamsa Beta"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Install"
  ; Close a running copy so its file can be replaced.
  nsExec::Exec 'taskkill /F /IM ${EXE}'
  Sleep 300
  SetOutPath "$INSTDIR"
  File "..\..\target\release\${EXE}"
  File "..\assets\fonts\OFL.txt"
  WriteUninstaller "$INSTDIR\uninstall.exe"

  CreateShortCut "$SMPROGRAMS\Hamsa Beta.lnk" "$INSTDIR\${EXE}"
  CreateShortCut "$DESKTOP\Hamsa Beta.lnk" "$INSTDIR\${EXE}"

  WriteRegStr HKCU "Software\Hamsa Beta" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${NAME}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "Hamsa"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\${EXE}"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  nsExec::Exec 'taskkill /F /IM ${EXE}'
  Sleep 300
  Delete "$INSTDIR\${EXE}"
  Delete "$INSTDIR\OFL.txt"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\Hamsa Beta.lnk"
  Delete "$DESKTOP\Hamsa Beta.lnk"
  ; Start with Windows is switched on inside the app; remove it here. Never added at install time.
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "HamsaBeta"
  DeleteRegKey HKCU "${UNINST_KEY}"
  DeleteRegKey HKCU "Software\Hamsa Beta"
SectionEnd
