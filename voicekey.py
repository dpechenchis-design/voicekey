#!/usr/bin/env python3
"""VoiceKey — hold a key, speak, release, and the text is pasted at the cursor.

Runs on macOS and Windows as a tray / menu-bar app. Transcription is local
(faster-whisper), so there is no API key and no cost.

  python voicekey.py                 # tray app
  python voicekey.py --file a.wav    # transcribe a file and exit (debug)
"""
import argparse
import json
import multiprocessing
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyperclip
import pystray
import sounddevice as sd
from PIL import Image, ImageDraw
from pynput import keyboard

IS_MAC = platform.system() == "Darwin"
IS_WIN = platform.system() == "Windows"

SAMPLE_RATE = 16000
MIN_SECONDS = 0.3  # ignore accidental taps
CONFIG_PATH = Path.home() / ".voicekey" / "config.json"

LANGUAGES = {"en": "English", "uk": "Українська", "ru": "Русский", "auto": "Auto-detect"}
KEYS = {
    "alt_r": "Right Option" if IS_MAC else "Right Alt",
    "ctrl_r": "Right Control",
    "f13": "F13",
    **({"cmd_r": "Right Command"} if IS_MAC else {}),
}
DEFAULTS = {"lang": "en", "key": "alt_r" if IS_MAC else "ctrl_r"}


# ---------- config ----------
def load_config() -> dict:
    try:
        return {**DEFAULTS, **json.loads(CONFIG_PATH.read_text())}
    except Exception:
        return dict(DEFAULTS)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------- platform bits ----------
LOG_PATH = CONFIG_PATH.parent / "app.log"


def log(msg: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def play(sound: str) -> None:
    """sound: 'start' or 'stop'."""
    try:
        if IS_MAC:
            name = {"start": "Tink", "stop": "Pop"}[sound]
            subprocess.Popen(
                ["afplay", f"/System/Library/Sounds/{name}.aiff"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif IS_WIN:
            import winsound

            freq = {"start": 880, "stop": 660}[sound]
            threading.Thread(target=winsound.Beep, args=(freq, 80), daemon=True).start()
    except Exception:
        pass


def paste_text(text: str) -> None:
    """Put text on the clipboard, press Cmd/Ctrl+V, then restore the old clipboard."""
    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text)
    time.sleep(0.05)
    kb = keyboard.Controller()
    mod = keyboard.Key.cmd if IS_MAC else keyboard.Key.ctrl
    with kb.pressed(mod):
        kb.press("v")
        kb.release("v")
    time.sleep(0.25)  # let the target app read the clipboard before restoring
    if old is not None:
        pyperclip.copy(old)


def tell_user(message: str) -> None:
    """Show a self-closing dialog so the user can see the app has started."""
    if not IS_MAC:
        return
    script = (
        f'display dialog "{message}" with title "VoiceKey" buttons {{"OK"}} '
        'default button "OK" giving up after 20'
    )
    subprocess.Popen(["osascript", "-e", script],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_permissions(prompt: bool) -> list[str]:
    """macOS only: return the missing permissions; with prompt=True, ask the system for them."""
    if not IS_MAC:
        return []
    missing = []
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
        from Quartz import CGPreflightListenEventAccess, CGRequestListenEventAccess

        ax = bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt}))
        listen = bool(CGPreflightListenEventAccess())
        log(f"permissions: accessibility={ax} input_monitoring={listen} prompt={prompt} exe={sys.executable}")
        if not ax:
            missing.append("Accessibility")
        if not listen:
            if prompt:
                CGRequestListenEventAccess()
            missing.append("Input Monitoring")
    except Exception as e:
        log(f"permission check failed: {e!r}")
    return missing


def make_icon(color: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), fill=color)
    d.rounded_rectangle((26, 16, 38, 36), radius=6, fill="white")  # mic body
    d.arc((20, 24, 44, 44), 0, 180, fill="white", width=3)         # mic cradle
    d.line((32, 44, 32, 50), fill="white", width=3)
    return img


# ---------- transcription ----------
def model_for(lang: str) -> str:
    # *.en models are English-only and faster; other languages need multilingual.
    return "base.en" if lang == "en" else "small"


class Transcriber:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.lock = threading.Lock()

    def load(self, lang: str) -> None:
        name = model_for(lang)
        with self.lock:
            if self.model_name == name:
                return
            from faster_whisper import WhisperModel

            self.model = WhisperModel(name, device="cpu", compute_type="int8")
            self.model_name = name

    def __call__(self, audio: np.ndarray, lang: str) -> str:
        self.load(lang)
        segments, _ = self.model.transcribe(
            audio,
            language=None if lang == "auto" else lang,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class Recorder:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None

    def start(self):
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=lambda indata, *_: self.frames.append(indata.copy()),
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.frames:
            return np.zeros(0, dtype="float32")
        return np.concatenate(self.frames).flatten()


# ---------- the app ----------
class App:
    def __init__(self):
        self.cfg = load_config()
        self.transcriber = Transcriber()
        self.recorder = Recorder()
        self.missing = check_permissions(prompt=True)
        self.status = "Loading model..."
        self.recording = False
        self.busy = False
        self.icon = pystray.Icon(
            "VoiceKey", make_icon("#888888"), "VoiceKey", menu=self.build_menu()
        )

    # --- tray ---
    def build_menu(self) -> pystray.Menu:
        def radio(table, field):
            return pystray.Menu(*[
                pystray.MenuItem(
                    label, self._setter(field, value),
                    checked=lambda item, v=value: self.cfg[field] == v, radio=True,
                )
                for value, label in table.items()
            ])

        return pystray.Menu(
            pystray.MenuItem(lambda item: self.status, None, enabled=False),
            pystray.MenuItem("Language", radio(LANGUAGES, "lang")),
            pystray.MenuItem("Hold-to-talk key", radio(KEYS, "key")),
            pystray.MenuItem("Re-check permissions", self._recheck),
            pystray.MenuItem("Quit", lambda icon, item: icon.stop()),
        )

    def _recheck(self, icon, item):
        self.missing = check_permissions(prompt=False)
        self.set_status(self.ready_text())

    def _setter(self, field, value):
        def set_value(icon, item):
            self.cfg[field] = value
            save_config(self.cfg)
            if field == "lang":
                threading.Thread(target=self.load_model, daemon=True).start()
            self.set_status(self.ready_text())
        return set_value

    def ready_text(self) -> str:
        if self.missing:
            return "Needs permission: " + " + ".join(self.missing)
        return f"Ready — hold {KEYS.get(self.cfg['key'], self.cfg['key'])}"

    def set_status(self, text: str, color: str | None = None) -> None:
        self.status = text
        if color:
            self.icon.icon = make_icon(color)
        self.icon.update_menu()
        self.icon.title = f"VoiceKey: {text}"

    def load_model(self) -> None:
        self.busy = True
        self.set_status("Loading model (first run downloads it)...", "#888888")
        try:
            self.transcriber.load(self.cfg["lang"])
            self.set_status(self.ready_text(), "#2e7d32")
        except Exception as e:
            self.set_status(f"Model error: {e}", "#c62828")
        finally:
            self.busy = False

    # --- hotkey ---
    def is_hotkey(self, key) -> bool:
        return key == getattr(keyboard.Key, self.cfg["key"], None)

    def on_press(self, key):
        log(f"key press: {key} hotkey={self.cfg['key']} match={self.is_hotkey(key)}")
        if self.is_hotkey(key) and not self.recording and not self.busy:
            try:
                self.recorder.start()
            except Exception as e:
                self.set_status(f"Mic error: {e}", "#c62828")
                return
            self.recording = True
            play("start")
            self.set_status("Recording...", "#d32f2f")

    def on_release(self, key):
        if self.is_hotkey(key) and self.recording:
            self.recording = False
            audio = self.recorder.stop()
            play("stop")
            threading.Thread(target=self.process, args=(audio,), daemon=True).start()

    def process(self, audio: np.ndarray) -> None:
        self.busy = True
        try:
            if len(audio) < SAMPLE_RATE * MIN_SECONDS:
                return
            self.set_status("Transcribing...", "#f9a825")
            text = self.transcriber(audio, self.cfg["lang"])
            if text:
                print(text)
                paste_text(text + " ")
        except Exception as e:
            print(f"error: {e}")
        finally:
            self.busy = False
            self.set_status(self.ready_text(), "#2e7d32")

    # --- run ---
    def run(self) -> None:
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        def setup(icon):
            icon.visible = True
            if self.missing:
                tell_user(
                    "VoiceKey needs permission to hear the hotkey. In System Settings > Privacy "
                    "& Security, turn ON VoiceKey under: " + ", ".join(self.missing)
                    + ". Then choose Quit from the menu icon and open VoiceKey again."
                )
            else:
                tell_user(
                    "VoiceKey is running. Look for the microphone icon in the menu bar "
                    "(top right, near the clock).\\n\\nHold " + KEYS.get(self.cfg["key"], self.cfg["key"])
                    + " and speak; release to paste the text. Choose the language from the menu icon."
                )
            threading.Thread(target=self.load_model, daemon=True).start()

        self.icon.run(setup)  # blocks on the main thread until Quit
        listener.stop()


def transcribe_file(path: str, lang: str | None = None) -> None:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-"],
        capture_output=True, check=True,
    ).stdout
    t0 = time.time()
    text = Transcriber()(np.frombuffer(raw, dtype="float32"), lang or load_config()["lang"])
    print(f"[{time.time() - t0:.1f}s] {text}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--file", help="transcribe an audio file and exit")
    p.add_argument("--lang", help="override saved language for --file (en, uk, ru, auto)")
    args = p.parse_args()

    if args.file:
        transcribe_file(args.file, args.lang)
        return
    App().run()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed once the app is frozen by PyInstaller
    main()
