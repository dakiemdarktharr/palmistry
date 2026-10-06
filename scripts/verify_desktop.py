"""Render the real template and test a download without camera or user data."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")

from flask import Flask, jsonify, render_template
from werkzeug.serving import make_server
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtWebEngineCore import QWebEngineProfile
from palmistry_desktop import AppWindow, save_download
from palm_keypoints import LINES


def main():
    web = Flask(__name__, template_folder=str(ROOT / "templates"))

    @web.get("/keypoints")
    def page():
        return render_template("keypoints.html", csrf="test-token", nonce="test-nonce", lines=LINES)

    @web.get("/api/keypoints/state")
    def state():
        return jsonify(ok=True, projects=[], runs=[], job={"status": "idle"})

    @web.get("/download")
    def download():
        return web.response_class('{"test_only":true}', mimetype="application/json",
            headers={"Content-Disposition": 'attachment; filename="test.json"'})

    server = make_server("127.0.0.1", 0, web, threaded=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    application = QApplication(sys.argv)
    window = AppWindow()
    result = {"ok": False}

    with tempfile.TemporaryDirectory(prefix="palm-desktop-check-") as folder:
        output = Path(folder) / "test.json"
        QFileDialog.getSaveFileName = lambda *args: (str(output), "")

        def finished(download):
            if not download.isFinished():
                return
            try:
                assert json.loads(output.read_text()) == {"test_only": True}
                result.update(ok=True, download=True)
            except Exception as exc:
                result["error"] = str(exc)
            application.quit()

        def requested(download):
            download.isFinishedChanged.connect(lambda: finished(download))
            save_download(window, download)

        def inspect(value):
            try:
                doc = json.loads(value)
                assert "Keypoint" in doc["heading"] and doc["canvas"] and doc["fileInput"]
                assert doc["lineButtons"] == 4, "template JavaScript did not run"
                popup = window.view.createWindow(None)
                assert popup.window().windowTitle() == "Palmistry Live Camera"
                popup.window().close()
                result.update(template=True, popup=True)
                window.view.page().runJavaScript("location.href='/download'")
            except Exception as exc:
                result["error"] = str(exc)
                application.quit()

        def loaded(ok):
            window.view.loadFinished.disconnect(loaded)
            if not ok:
                result["error"] = "local template did not load"
                application.quit()
                return
            QTimer.singleShot(1000, lambda: window.view.page().runJavaScript(
                "JSON.stringify({heading:document.querySelector('h1')?.textContent,"
                "canvas:!!document.getElementById('canvas'),"
                "fileInput:!!document.querySelector('input[type=file]'),"
                "lineButtons:document.querySelectorAll('[data-line]').length})", inspect))

        QWebEngineProfile.defaultProfile().downloadRequested.connect(requested)
        window.view.loadFinished.connect(loaded)
        window.view.setUrl(QUrl(f"http://127.0.0.1:{server.server_port}/keypoints"))
        window.show()
        QTimer.singleShot(20000, application.quit)
        try:
            application.exec()
        finally:
            server.shutdown()
            worker.join(timeout=5)
            server.server_close()
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
