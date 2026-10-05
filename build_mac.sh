#!/bin/bash
# Builds $BUILD/dist/VoiceKey.app (menu-bar app with a Dock icon, ad-hoc signed).
set -e
cd "$(dirname "$0")"
# Build outside iCloud-synced folders (Desktop/Documents): syncing makes PyInstaller hang.
BUILD=${BUILD_DIR:-$HOME/.voicekey-build}
mkdir -p "$BUILD"
PY=${PY:-.venv/bin/python}
$PY -m PyInstaller --noconfirm --clean --windowed --workpath "$BUILD/work" --distpath "$BUILD/dist" --specpath "$BUILD" --name VoiceKey \
  --osx-bundle-identifier com.voicekey.app \
  --collect-all faster_whisper --collect-all ctranslate2 --collect-all onnxruntime \
  --collect-all sounddevice --collect-data av \
  "$PWD/voicekey.py"
PLIST=$BUILD/dist/VoiceKey.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string 'VoiceKey records your voice while you hold the hotkey to turn it into text.'" $PLIST 2>/dev/null || true
DR='designated => identifier "com.voicekey.app"'
codesign --force --deep --sign - $BUILD/dist/VoiceKey.app
# iCloud (Desktop/Documents) adds xattrs that invalidate the signature, so install a clean copy.
mkdir -p ~/Applications && rm -rf ~/Applications/VoiceKey.app
ditto --norsrc --noextattr --noqtn $BUILD/dist/VoiceKey.app ~/Applications/VoiceKey.app
xattr -cr ~/Applications/VoiceKey.app
codesign --force --deep --sign - ~/Applications/VoiceKey.app
# top-level only (no --deep): keeps nested libraries valid, makes the app match by identifier
codesign --force --sign - -r="$DR" ~/Applications/VoiceKey.app
echo "Built $BUILD/dist/VoiceKey.app and installed ~/Applications/VoiceKey.app"
