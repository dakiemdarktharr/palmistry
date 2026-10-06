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
    web = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    fixture = tempfile.TemporaryDirectory(prefix="palm-desktop-data-")
    project = Path(fixture.name) / "artifacts/keypoints/desktop_smoke"
    rows = []
    for i in range(3):
        image_id = f"{i:020x}"
        image_path = project / "images" / (image_id + ".jpg")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (500,500), "#ab9478").save(image_path)
        a = empty_annotation(image_id)
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

    fail_next = {'value': True}
    @web.before_request
    def simulate_storage_failure():
        if request.method=='POST' and '/swipe/' in request.path and fail_next['value']:
            fail_next['value']=False
            return jsonify(ok=False,error='TEST write failure; retry safely'),503

    @web.after_request
    def csp(response):
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'nonce-test-nonce'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
        return response

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

        page = window.view.page()
        def fail(error):
            result['error']=str(error)
            application.quit()

        def wait_for(expression,callback,attempt=0):
            def checked(value):
                if value:
                    try:callback()
                    except Exception as exc:fail(exc)
                elif attempt<100:QTimer.singleShot(100,lambda:wait_for(expression,callback,attempt+1))
                else:fail('Timed out: '+expression)
            page.runJavaScript('Boolean('+expression+')',checked)

        def execute(script,callback):
            page.runJavaScript(script,lambda _:QTimer.singleShot(50,callback))

        def stored(index):return read_json(annotation_path(project,rows[index]['image_id']))

        def gesture(distance,cancel=False):
            return """(()=>{
              const h=document.getElementById('swipeHandle'),r=h.getBoundingClientRect();
              const x=r.left+r.width/2,y=r.top+r.height/2;
              h.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,button:0,buttons:1,pointerId:7,pointerType:'mouse',clientX:x,clientY:y}));
              window.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,button:0,buttons:1,pointerId:7,pointerType:'mouse',clientX:x+DISTANCE,clientY:y}));
              window.dispatchEvent(new PointerEvent('EVENT',{bubbles:true,button:0,buttons:0,pointerId:7,pointerType:'mouse',clientX:x+DISTANCE,clientY:y}));
            })()""".replace('DISTANCE',str(distance)).replace('EVENT','pointercancel' if cancel else 'pointerup')

        def canvas_point(x,y):
            return f"""(()=>{{
              const c=document.getElementById('canvas'),r=c.getBoundingClientRect();c.setPointerCapture=()=>{{}};
              for(const type of ['pointerdown','pointerup'])c.dispatchEvent(new PointerEvent(type,{{bubbles:true,button:0,pointerId:1,pointerType:'mouse',clientX:r.left+{x}*r.width,clientY:r.top+{y}*r.height}}));
            }})()"""

        def draw():
            assert len(LINES)==3
            result.update(template=True)
            execute("""(()=>{
              const c=document.getElementById('canvas'),r=c.getBoundingClientRect();c.setPointerCapture=()=>{};
              function event(type,x){c.dispatchEvent(new PointerEvent(type,{bubbles:true,button:0,buttons:type==='pointerup'?0:1,pointerId:1,pointerType:'mouse',clientX:r.left+x*r.width,clientY:r.top+.3*r.height}));}
              event('pointerdown',.2);for(let i=1;i<=20;i++)event('pointermove',.2+i*.03);event('pointerup',.8);
            })()""",lambda:wait_for("document.querySelector('[data-tool=heart_line] small')?.textContent==='Đã vẽ'",select_width))

        def select_width():
            execute("document.querySelector('[data-tool=width]').click()",lambda:wait_for("document.querySelector('[data-tool=width]')?.getAttribute('aria-pressed')==='true'",width_first))

        def width_first():
            execute(canvas_point(.15,.8),lambda:wait_for("document.querySelector('[data-tool=width] small')?.textContent==='1/2 điểm'",width_second))

        def width_second():
            execute(canvas_point(.85,.8),lambda:wait_for("document.querySelector('[data-tool=width] small')?.textContent==='Đã vẽ'",try_failure))

        def try_failure():
            execute("document.getElementById('chooseLeft').click()",lambda:wait_for("document.getElementById('message')?.textContent.includes('TEST write failure') && !document.getElementById('chooseLeft').disabled",verify_failure))

        def verify_failure():
            assert stored(0)['status']=='pending' and stored(0)['palm_width_points']==[]
            result['retryPreservesDraft']=True
            # A cancelled long gesture and a short gesture must not save.
            execute(gesture(-130,True),lambda:execute(gesture(-30),lambda:QTimer.singleShot(250,retry)))

        def retry():
            assert stored(0)['status']=='pending'
            result['cancelledSwipe']=True
            execute(gesture(-140),lambda:wait_for("document.getElementById('photoCard')?.dataset.imageId==='00000000000000000001' && !document.getElementById('chooseLeft').disabled",verify_left))

        def verify_left():
            a=stored(0)
            assert a['status']=='approved' and a['handedness']=='left'
            assert len(a['lines']['heart_line']['points'])==6 and len(a['palm_width_points'])==2
            assert a['lines']['head_line']=={'status':'unreviewed','points':[]}
            assert a['lines']['life_line']=={'status':'unreviewed','points':[]}
            result.update(freehand=True,swipe=True,partialAnnotations=True)
            execute("document.getElementById('undoSwipe').click()",lambda:wait_for("document.getElementById('photoCard')?.dataset.imageId==='00000000000000000000' && !document.getElementById('chooseLeft').disabled",verify_undo))

        def verify_undo():
            assert stored(0)['status']=='pending' and stored(0)['handedness']=='unknown'
            assert len(stored(0)['lines']['heart_line']['points'])==6
            result['undoSwipe']=True
            execute("document.getElementById('chooseLeft').click()",lambda:wait_for("document.getElementById('photoCard')?.dataset.imageId==='00000000000000000001' && !document.getElementById('chooseRight').disabled",save_empty))

        def save_empty():
            execute(gesture(140),lambda:wait_for("document.getElementById('photoCard')?.dataset.imageId==='00000000000000000002' && !document.getElementById('chooseLeft').disabled",verify_empty))

        def verify_empty():
            a=stored(1)
            assert a['status']=='approved' and a['handedness']=='right' and a['palm_width_points']==[]
            assert all(line=={'status':'unreviewed','points':[]} for line in a['lines'].values())
            result['emptySwipe']=True
            QTimer.singleShot(500, capture_and_download)

        def capture_and_download():
            output_dir=ROOT/'artifacts/desktop-swipe'
            output_dir.mkdir(parents=True,exist_ok=True)
            window.grab().save(str(output_dir/'labeler.png'))
            popup=window.view.createWindow(None)
            assert popup.window().windowTitle()=='Palmistry Live Camera'
            popup.window().close();result['popup']=True
            page.runJavaScript("window.closeProofCount=0;window.addEventListener('beforeunload',()=>navigator.sendBeacon('/close-proof',String(++window.closeProofCount)));location.href='/download'")

        def loaded(ok):
            window.view.loadFinished.disconnect(loaded)
            if not ok:fail('local template did not load');return
            wait_for("document.querySelectorAll('[data-line]').length===3 && document.getElementById('canvas') && !document.getElementById('chooseLeft').disabled && !document.querySelector('.photo-loading')",draw)

        QWebEngineProfile.defaultProfile().downloadRequested.connect(requested)
        window.view.loadFinished.connect(loaded)
        window.view.setUrl(QUrl(f"http://127.0.0.1:{server.server_port}/keypoints"))
        window.show()
        QTimer.singleShot(45000, application.quit)
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
