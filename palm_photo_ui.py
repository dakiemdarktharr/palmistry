"""Local single-photo workflow; ephemeral files, no personal-trait inference."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from flask import jsonify, request, render_template_string

PHOTO_PAGE = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đo đường bàn tay</title><style>body{font:16px system-ui;max-width:960px;margin:32px auto;padding:0 20px;background:#12151a;color:#eee}a{color:#8ed6fa}label{display:block;margin:18px 0}input,select,button{font:inherit;padding:10px;max-width:100%}button{cursor:pointer}img{max-width:100%;max-height:520px}table{border-collapse:collapse;width:100%;margin-top:20px}td,th{padding:12px;border-bottom:1px solid #555;text-align:left}.note{color:#bbc3ce}#status{white-space:pre-line}</style>
<a href="/">← Quay lại app</a><h1>Đo đường bàn tay</h1>
<p>Chụp một lòng bàn tay mở, ngón hướng lên, thấy rõ ngón cái và cổ tay.</p>
<form id="photoForm"><label>Ảnh bàn tay <input id="photo" name="image" type="file" accept="image/png,image/jpeg,image/webp,image/tiff" required></label>
<label>Ảnh có lật gương không? <select name="mirrored" id="mirrored"><option value="unknown">Chưa biết</option><option value="no">Không lật</option><option value="yes">Có lật</option></select></label>
<label>Cách nhận diện <select name="mode"><option value="classical">CV thử nghiệm — chưa nhận diện simian tự động</option><option value="model">Model đã cấu hình</option></select></label>
<button id="submit" type="submit">Phân tích ảnh</button></form>
<p class="note">Kết quả là phép đo hình học. Ảnh tải lên được xử lý cục bộ, không đưa vào dataset. Model simian cần mask simian đã được duyệt.</p>
<p id="status" role="status"></p><p id="hand"></p><img id="overlay" hidden alt="Overlay đường bàn tay">
<table id="result" hidden><thead><tr><th>Đường nhận diện được</th><th>Tỷ lệ chiều dài / bề rộng lòng bàn tay ước tính</th></tr></thead><tbody></tbody></table>
<script nonce="{{nonce}}">const names={life_line:'Life',head_line:'Head',heart_line:'Heart',simian_line:'Simian'};
document.getElementById('photoForm').addEventListener('submit',async event=>{event.preventDefault();const button=document.getElementById('submit');const status=document.getElementById('status');button.disabled=true;status.textContent='Đang kiểm tra ảnh và đo đường...';document.getElementById('result').hidden=true;document.getElementById('overlay').hidden=true;document.getElementById('hand').textContent='';document.querySelector('tbody').replaceChildren();try{const response=await fetch('/api/photo/analyze',{method:'POST',headers:{'X-Palm-CSRF':{{csrf|tojson}}},body:new FormData(event.target)});const data=await response.json();if(!response.ok||!data.ok)throw Error(data.error||'Không phân tích được ảnh.');const out=data.result;status.textContent=out.status==='success'?'Đã hoàn tất phép đo.':(out.reason_vi||'Chưa có đường đủ chắc chắn.')+'\n'+(out.fix_vi||'');const hand=out.handedness||{};document.getElementById('hand').textContent=hand.side==='left'?'Tay trái (ước lượng CV)':hand.side==='right'?'Tay phải (ước lượng CV)':'Tay: chưa xác định';if(data.overlay){document.getElementById('overlay').src=data.overlay;document.getElementById('overlay').hidden=false;}let count=0;for(const [name,item] of Object.entries(out.accepted_lines||{})){if(!names[name]||!item.detected)continue;const row=document.createElement('tr');for(const value of [names[name],Number(item.length_to_palm_ratio).toFixed(3)]){const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell);}document.querySelector('tbody').appendChild(row);count++;}document.getElementById('result').hidden=count===0;}catch(error){status.textContent=error.message;}finally{button.disabled=false;}});</script></html>'''


def register_photo_ui(app, *, project_dir, checkpoint, checkpoint_sha256, csrf, nonce):
    lock=threading.Lock()

    @app.get('/analyze')
    def photo_page():
        return render_template_string(PHOTO_PAGE,csrf=csrf,nonce=nonce)

    @app.post('/api/photo/analyze')
    def analyze_photo():
        if request.content_length and request.content_length>21*1024*1024:
            return jsonify(ok=False,error='Ảnh tối đa 20 MiB.'),413
        uploaded=request.files.get('image')
        if uploaded is None:
            return jsonify(ok=False,error='Chọn một ảnh bàn tay.'),400
        mirrored=request.form.get('mirrored','unknown');mode=request.form.get('mode','classical')
        if mirrored not in ('yes','no','unknown') or mode not in ('classical','model'):
            return jsonify(ok=False,error='Lựa chọn ảnh không hợp lệ.'),400
        if mode=='model' and (not checkpoint or not checkpoint_sha256):
            return jsonify(ok=False,error='Chưa cấu hình checkpoint v2 và SHA-256. Hãy train từ mask đã duyệt; hoặc chọn rõ CV thử nghiệm.'),409
        raw=uploaded.stream.read(20*1024*1024+1)
        if not raw or len(raw)>20*1024*1024:
            return jsonify(ok=False,error='Ảnh rỗng hoặc lớn hơn 20 MiB.'),413
        if not lock.acquire(blocking=False):
            return jsonify(ok=False,error='Đang phân tích ảnh khác. Thử lại sau khi hoàn tất.'),409
        try:
            with tempfile.TemporaryDirectory(prefix='palm-photo-') as temp:
                folder=Path(temp);source=folder/'input.img';source.write_bytes(raw)
                result_path=folder/'result.json';overlay=folder/'overlay.jpg'
                command=[sys.executable,str(project_dir/'palmistry_strict_auto_onefile.py'),'predict','--image',str(source),'--mirrored',mirrored,
                         '--out_size','512','--out_json',str(result_path),'--out_mask',str(folder/'mask.png'),'--out_overlay',str(overlay)]
                if mode=='model':command+=['--checkpoint',str(checkpoint),'--checkpoint-sha256',checkpoint_sha256,'--require_validation_scores']
                else:command+=['--allow-classical-fallback']
                env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
                proc=subprocess.run(command,cwd=str(project_dir),capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=60,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if not result_path.is_file():
                    return jsonify(ok=False,error='Tiến trình không tạo kết quả. Hãy thử ảnh JPG nhỏ hơn.'),500
                out=json.loads(result_path.read_text(encoding='utf-8'))
                if out.get('status')=='error' or (proc.returncode and out.get('status')!='need_retake'):
                    return jsonify(ok=False,error=out.get('error','Không xử lý được ảnh.')),422
                encoded='data:image/jpeg;base64,'+base64.b64encode(overlay.read_bytes()).decode('ascii') if overlay.is_file() else None
                # Source paths and training internals are not needed in the photo UI.
                result={key:out[key] for key in ('status','reason_vi','fix_vi','handedness','accepted_lines','inference_source','class_map_version','measurement_version') if key in out}
                return jsonify(ok=True,result=result,overlay=encoded)
        except subprocess.TimeoutExpired:
            return jsonify(ok=False,error='Phân tích quá 60 giây. Thử ảnh nhỏ hơn hoặc chụp lại rõ hơn.'),504
        finally:
            lock.release()
