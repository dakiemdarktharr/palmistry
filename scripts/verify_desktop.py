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

from flask import Flask, jsonify, render_template, request
from werkzeug.serving import make_server
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtWebEngineCore import QWebEngineProfile
from palmistry_desktop import AppWindow, save_download
from palm_keypoints import LINES, SCHEMA
from palm_keypoints.data import empty_annotation, write_json, read_json, annotation_path
from palm_keypoints.web import register_keypoint_ui
from PIL import Image


def main():
    web = Flask(__name__, template_folder=str(ROOT / "templates"))
    fixture = tempfile.TemporaryDirectory(prefix="palm-desktop-data-")
    project = Path(fixture.name) / "artifacts/keypoints/desktop_smoke"
    rows = []
    for i in range(2):
        image_id = f"{i:020x}"
        image_path = project / "images" / (image_id + ".jpg")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (500,500), "#ab9478").save(image_path)
        a = empty_annotation(image_id)
        a['palm_width_points'] = [[.15,.8],[.85,.8]]
        for j,name in enumerate(LINES):
            a['lines'][name] = {'status':'present','points':[[.2+k*.12,.25+j*.12] for k in range(6)]}
        a['lines'][LINES[0]] = {'status':'unreviewed','points':[]}
        write_json(annotation_path(project,image_id),a)
        rows.append({'image_id':image_id,'image_path':'images/'+image_id+'.jpg','width':500,'height':500})
    write_json(project/'project.json',{'schema':SCHEMA,'images':rows})
    register_keypoint_ui(web,fixture.name,'test-token','test-nonce')
    unloaded = threading.Event()

    @web.post("/close-proof")
    def close_proof():
        if request.get_data() == b"2":
            unloaded.set()
        return "", 204

    @web.get("/keypoints")
    def page():
        return render_template("keypoints.html", csrf="test-token", nonce="test-nonce", lines=LINES)

    @web.get("/download")
    def download():
        return web.response_class('{"test_only":true}', mimetype="application/json",
            headers={"Content-Disposition": 'attachment; filename="test.json"'})

    server = make_server("127.0.0.1", 0, web, threaded=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    application = QApplication(sys.argv)
    application.setQuitOnLastWindowClosed(False)
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
                result.update(download=True)
                window.close()
                QTimer.singleShot(1000, verify_close)
                return
            except Exception as exc:
                result["error"] = str(exc)
            application.quit()

        def verify_close():
            result.update(closeGuard=unloaded.is_set(), ok=unloaded.is_set() and not window.isVisible())
            application.quit()

        def requested(download):
            download.isFinishedChanged.connect(lambda: finished(download))
            save_download(window, download)

        def inspect(value):
            try:
                doc = json.loads(value)
                assert "100" in doc["heading"] and doc["canvas"] and doc["fileInput"]
                assert not doc["handFields"] and doc["steps"], "simplified annotation controls missing"
                assert doc["lineButtons"] == 4, "template JavaScript did not run"
                window.view.page().runJavaScript("""
                    (()=>{
                      const canvas=document.getElementById('canvas'),r=canvas.getBoundingClientRect();
                      canvas.setPointerCapture=()=>{};
                      for(let k=0;k<6;k++){
                        canvas.onpointerdown({clientX:r.left+(.2+k*.12)*r.width,clientY:r.top+.25*r.height,pointerId:1});
                        canvas.onpointerup();
                      }
                      document.querySelector('[data-line="fate_line"]').click();
                      document.getElementById('absent').click();
                      document.getElementById('approve').click();
                    })()
                """)
                QTimer.singleShot(1500, check_annotation)
            except Exception as exc:
                result["error"] = str(exc)
                application.quit()

        annotation_checks = 0
        def check_annotation():
            nonlocal annotation_checks
            annotation_checks += 1
            try:
                a=read_json(annotation_path(project,rows[0]['image_id']))
                if a['status']=='pending' and annotation_checks<30:
                    QTimer.singleShot(200, check_annotation)
                    return
                assert a['status']=='approved' and a['mirrored']=='unknown' and a['handedness']=='unknown'
                assert len(a['lines'][LINES[0]]['points'])==6 and a['lines']['fate_line']['status']=='absent'
                assert read_json(annotation_path(project,rows[1]['image_id']))['status']=='pending'
                window.view.page().runJavaScript("document.getElementById('index').value", check_next)
            except Exception as exc:
                result['error']=str(exc)
                application.quit()

        def check_next(value):
            try:
                assert str(value)=='2', 'approval did not advance to the next image'
                result.update(annotation=True, nextImage=True)
                popup = window.view.createWindow(None)
                assert popup.window().windowTitle() == "Palmistry Live Camera"
                popup.window().close()
                result.update(template=True, popup=True)
                window.view.page().runJavaScript(
                    "window.closeProofCount=0;"
                    "window.addEventListener('beforeunload',()=>navigator.sendBeacon('/close-proof',String(++window.closeProofCount)));"
                    "location.href='/download'")
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
                "lineButtons:document.querySelectorAll('[data-line]').length,"
                "handFields:!!document.querySelector('#hand,#mirror,#predictMirror'),"
                "steps:!!document.getElementById('stepHint')?.textContent})", inspect))

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
            fixture.cleanup()
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
