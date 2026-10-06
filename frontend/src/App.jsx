import React, { useEffect, useRef, useState } from 'react';
import { ActionIcon, Alert, Badge, Button, Group, Loader, Progress, Select, SegmentedControl, Switch, Tooltip } from '@mantine/core';
import { ArrowLeft, ArrowRight, Check, CircleHelp, Download, Hand, History, Maximize2, PencilLine, RotateCcw, Settings2, Undo2 } from 'lucide-react';
import PhotoCanvas from './PhotoCanvas.jsx';
import SwipeTray from './SwipeTray.jsx';
import ToolsDrawer from './ToolsDrawer.jsx';
import { TOOLS, firstIncomplete, nextPending, pointsFor, request, withPoints } from './domain.js';

export default function App() {
  const [state,setState]=useState({projects:[],runs:[],job:{status:'idle'}});
  const [project,setProject]=useState(''),[images,setImages]=useState([]),[index,setIndex]=useState(0);
  const [annotation,setAnnotation]=useState(null),[dirty,setDirty]=useState(false),[busy,setBusy]=useState(true);
  const [tool,setTool]=useState('heart_line'),[mode,setMode]=useState('draw'),[traces,setTraces]=useState({});
  const [overlay,setOverlay]=useState(true),[magnify,setMagnify]=useState(false),[tools,setTools]=useState(false);
  const [message,setMessage]=useState(null),[distance,setDistance]=useState(0),[leaving,setLeaving]=useState(false);
  const [done,setDone]=useState(false),[history,setHistory]=useState([]),[undoCount,setUndoCount]=useState(0);
  const session=useRef({project:'',images:[],index:0,annotation:null,dirty:false});
  const busyRef=useRef(true),undo=useRef([]),previewSequence=useRef(0),awaitedProject=useRef('');
  const busyJob=state.job.status==='running'&&state.job.project===project;
  const locked=busy||busyJob;
  const notice=(text,bad=false)=>setMessage({text,bad});
  const setWorking=value=>{busyRef.current=value;setBusy(value);};

  async function refresh() {
    const result=await request('/api/keypoints/state');setState(result);return result;
  }
  function useAnnotation(value, imageIndex) {
    session.current={...session.current,index:imageIndex,annotation:value,dirty:false};
    setIndex(imageIndex);setAnnotation(value);setDirty(false);setTool(firstIncomplete(value));
    setTraces({});setDistance(0);setLeaving(false);setDone(false);undo.current=[];setUndoCount(0);
  }
  async function fetchImage(imageIndex) {
    const s=session.current,entry=s.images[imageIndex];if(!entry)return;
    const result=await request(`/api/keypoints/annotation/${s.project}/${entry.image_id}`);
    useAnnotation(result.annotation,imageIndex);
  }
  async function loadProject(name, explicitIndex=null) {
    const result=await request('/api/keypoints/project/'+name);
    session.current={project:name,images:result.images,index:0,annotation:null,dirty:false};
    setProject(name);setImages(result.images);setAnnotation(null);setDirty(false);setDone(false);setHistory([]);
    const start=explicitIndex??Math.max(0,result.images.findIndex(image=>image.status==='pending'));
    if(result.images.length)await fetchImage(start);
    try{localStorage.setItem('palm-project',name)}catch{}
  }
  async function run(task) {
    if(busyRef.current)return;
    setWorking(true);
    try{await task()}catch(error){setDistance(0);setLeaving(false);notice(error.message,true)}
    finally{setWorking(false)}
  }
  async function persistDraft() {
    const s=session.current;if(!s.dirty||!s.annotation)return;
    const result=await request(`/api/keypoints/annotation/${s.project}/${s.annotation.image_id}`,{...s.annotation,status:'pending'});
    session.current={...s,annotation:result.annotation,dirty:false};setAnnotation(result.annotation);setDirty(false);
    const changed=s.images.map((image,i)=>i===s.index?{...image,status:'pending'}:image);
    session.current.images=changed;setImages(changed);
  }
  const reload=()=>run(async()=>{await persistDraft();await refresh();if(session.current.annotation)await fetchImage(session.current.index);notice('Đã tải lại ảnh.');});
  const navigate=target=>run(async()=>{await persistDraft();await fetchImage(Math.max(0,Math.min(session.current.images.length-1,target)));await refresh()});
  const switchProject=name=>run(async()=>{await persistDraft();await loadProject(name);await refresh()});

  useEffect(()=>{
    let live=true;
    (async()=>{
      try{
        const result=await refresh();if(!live)return;
        let remembered='';try{remembered=localStorage.getItem('palm-project')||''}catch{}
        const query=new URLSearchParams(location.search).get('project');
        const selected=result.projects.find(p=>p.name===(query||remembered))||result.projects.find(p=>!p.tutorial_only)||result.projects[0];
        if(selected)await loadProject(selected.name);
      }catch(error){notice(error.message,true)}finally{if(live)setWorking(false)}
    })();
    const timer=setInterval(()=>{
      refresh().then(result=>{
        const n=awaitedProject.current;
        if(n&&result.job.status==='completed'&&result.projects.some(p=>p.name===n)&&!busyRef.current){
          awaitedProject.current='';run(async()=>{await persistDraft();await loadProject(n);notice('Đã tạo bộ ảnh mới.');});
        }
      }).catch(error=>notice(error.message,true));
    },5000);
    return ()=>{live=false;clearInterval(timer)};
  },[]);

  useEffect(()=>{
    const warn=e=>{if(session.current.dirty||busyRef.current){e.preventDefault();e.returnValue='';}};
    window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);
  },[]);

  useEffect(()=>{
    if(!annotation)return;
    const sequence=++previewSequence.current,s=session.current;
    const timer=setTimeout(()=>{
      request(`/api/keypoints/preview/${s.project}/${annotation.image_id}`,{...annotation,status:'pending'})
        .then(result=>{if(sequence===previewSequence.current)setTraces(result.lines)})
        .catch(error=>{if(sequence===previewSequence.current)notice(error.message,true)});
    },160);
    return()=>{clearTimeout(timer);previewSequence.current++};
  },[annotation,project]);

  function change(value, remember=true, before=null) {
    if(busyRef.current||busyJob)return;
    if(remember){undo.current=[...undo.current,structuredClone(before||session.current.annotation)].slice(-30);setUndoCount(undo.current.length)}
    session.current={...session.current,annotation:value,dirty:true};setAnnotation(value);setDirty(true);setTraces({});
  }
  function undoEdit(){if(locked||!undo.current.length)return;const previous=undo.current.pop();setUndoCount(undo.current.length);change(previous,false);}

  async function choose(hand) {
    if(busyRef.current||busyJob||!session.current.annotation||done)return;
    await run(async()=>{
      const s=session.current,before=structuredClone(s.annotation),imageIndex=s.index;
      const result=await request(`/api/keypoints/swipe/${s.project}/${before.image_id}`,{handedness:hand,annotation:before});
      // Only acknowledge and move the card after the server's atomic write succeeds.
      session.current={...s,annotation:result.annotation,dirty:false};setAnnotation(result.annotation);setDirty(false);
      const changed=s.images.map((image,i)=>i===imageIndex?{...image,status:'approved'}:image);
      session.current.images=changed;setImages(changed);
      setHistory(previous=>[...previous,{project:s.project,index:imageIndex,before,revision:result.annotation.revision}].slice(-30));
      setDistance(hand==='left'?-360:360);setLeaving(true);
      const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if(!reduce)await new Promise(resolve=>setTimeout(resolve,180));
      const next=nextPending(changed,imageIndex);
      if(next>=0)await fetchImage(next);else{setDone(true);setDistance(0);setLeaving(false)}
      notice(`Đã lưu tay ${hand==='left'?'trái':'phải'}.`);await refresh();
    });
  }
  const undoSwipe=()=>run(async()=>{
    const last=history.at(-1);if(!last)return;
    await persistDraft();
    await request(`/api/keypoints/annotation/${last.project}/${last.before.image_id}`,{...last.before,revision:last.revision});
    const changed=session.current.images.map((image,i)=>i===last.index?{...image,status:last.before.status}:image);
    session.current.images=changed;setImages(changed);setHistory(previous=>previous.slice(0,-1));
    await fetchImage(last.index);await refresh();notice('Đã hoàn tác quẹt. Có thể chỉnh và chọn lại bên tay.');
  });

  useEffect(()=>{
    const key=e=>{
      if(/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||locked||!annotation)return;
      if(e.ctrlKey&&e.key.toLowerCase()==='z'){e.preventDefault();undoEdit();return;}
      if(!e.ctrlKey&&!e.metaKey&&!e.altKey){if(['t','p'].includes(e.key.toLowerCase())){e.preventDefault();choose(e.key.toLowerCase()==='t'?'left':'right');return;}const n=Number(e.key);if(n>=1&&n<=4){e.preventDefault();setTool(TOOLS[n-1].id)}}
    };
    window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);
  },[locked,annotation,history,done]);

  function downloadDraft(){
    if(!annotation)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(session.current.annotation,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download=annotation.image_id+'-draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
  }

  const current=state.projects.find(p=>p.name===project),entry=images[index];
  const reviewed=current?(current.counts.approved+current.counts.rejected):0;
  const active=TOOLS.find(t=>t.id===tool);
  const labeled=annotation?TOOLS.filter(t=>t.id==='width'?annotation.palm_width_points.length===2:annotation.lines[t.id].status!=='unreviewed').length:0;
  return <main className="app-shell" aria-busy={locked}>
    <header className="app-header">
      <div className="brand"><div className="brand-mark"><Hand size={26}/></div><div><h1>Palm Studio</h1><span>Dán nhãn 3 đường chỉ tay</span></div></div>
      <div className="header-progress"><div><strong>{reviewed}<span> / {current?.count||100} ảnh</span></strong><span>{current?.counts.pending??0} ảnh chờ</span></div><Progress value={current?.count?reviewed/current.count*100:0} size={5} color="violet" aria-label="Tiến độ dán nhãn"/></div>
      <Button variant="white" color="dark" leftSection={<Settings2 size={18}/>} onClick={()=>setTools(true)}>Công cụ</Button>
    </header>
    <div className="workspace-bar">
      <Select id="project" aria-label="Bộ ảnh" data={state.projects.map(p=>({value:p.name,label:p.tutorial_only?'Ví dụ hướng dẫn':p.name}))} value={project||null} onChange={n=>n&&switchProject(n)} disabled={busy} searchable placeholder="Chọn bộ ảnh"/>
      <div className="save-indicator" role="status">{busy?<><Loader size={13}/> Đang xử lý</>:dirty?<><span className="unsaved-dot"/>Nháp chưa lưu</>:<><Check size={15}/>Đã đồng bộ</>}</div>
      <Button size="xs" variant="subtle" leftSection={<History size={15}/>} id="undoSwipe" onClick={undoSwipe} disabled={locked||!history.length}>Hoàn tác quẹt</Button>
    </div>
    {message&&<Alert id="message" className="notice" color={message.bad?'red':'violet'} withCloseButton onClose={()=>setMessage(null)} role={message.bad?'alert':'status'}>{message.text}</Alert>}
    {state.project_errors?.length>0&&<Alert color="yellow">Có bộ ảnh không đọc được: {state.project_errors.map(p=>p.name).join(', ')}. Chọn bộ khác hoặc khôi phục dữ liệu.</Alert>}
    {current?.tutorial_only&&<Alert color="blue">Đây là ảnh hướng dẫn, không được dùng làm dữ liệu huấn luyện.</Alert>}

    {!annotation?<section className="empty-state"><Hand size={46}/><h2>{busy?'Đang mở bộ ảnh…':'Bắt đầu với 100 ảnh'}</h2><p>Tạo một bộ ảnh trong Công cụ rồi vẽ 3 đường chính trên mỗi lòng bàn tay.</p><Button onClick={()=>setTools(true)}>Mở Công cụ</Button></section>:
    <div className="labeling-layout">
      <aside className="drawing-tools" aria-label="Công cụ dán nhãn">
        <div className="tool-heading"><PencilLine size={17}/><h2>Vẽ trên ảnh</h2><Badge variant="light" size="sm">{labeled}/4</Badge></div>
        <div id="lineButtons" className="line-tools">{TOOLS.map((t,i)=>{
          const count=pointsFor(annotation,t.id).length,total=t.id==='width'?2:6;
          return <button key={t.id} data-line={t.id==='width'?undefined:t.id} data-tool={t.id} className={`tool-button ${tool===t.id?'selected':''}`} onClick={()=>setTool(t.id)} disabled={locked} aria-pressed={tool===t.id} style={{'--line-color':t.color}}>
            <span className="tool-color"/>
            <span><strong>{t.name}</strong><small>{count===total?'Đã vẽ':count?`${count}/${total} điểm`:'Để trống'}</small></span>
            <kbd>{i+1}</kbd>
          </button>;
        })}</div>
        <SegmentedControl id="drawMode" value={mode} onChange={setMode} disabled={locked} fullWidth data={[{value:'draw',label:'Vẽ đường'},{value:'points',label:'Đặt điểm'}]}/>
        <p id="stepHint" className="tool-hint">{active.hint}</p>
        <Group gap={8}><Button variant="default" size="xs" leftSection={<Undo2 size={14}/>} disabled={locked||!undoCount} onClick={undoEdit}>Hoàn tác</Button><Button variant="subtle" color="gray" size="xs" onClick={()=>change(withPoints(annotation,tool,[]))} disabled={locked||!pointsFor(annotation,tool).length}>Để trống</Button></Group>
        <div className="drawing-options"><Switch label="Hiện đường và điểm" checked={overlay} onChange={e=>setOverlay(e.currentTarget.checked)} size="sm"/><Switch label="Kính phóng đại" checked={magnify} onChange={e=>setMagnify(e.currentTarget.checked)} size="sm"/></div>
        <div className="shortcuts"><CircleHelp size={15}/><p>Kéo để vẽ. Kéo điểm để sửa.<br/>1–4 chọn công cụ · Ctrl+Z hoàn tác.<br/>T / P lưu tay trái / phải.</p></div>
      </aside>
      <section className="deck" aria-label="Thẻ ảnh dán nhãn">
        <div className="deck-meta"><span>Ảnh {index+1} / {images.length}</span><div><Tooltip label="Ảnh trước"><ActionIcon variant="subtle" color="gray" aria-label="Ảnh trước" onClick={()=>navigate(index-1)} disabled={locked||index===0}><ArrowLeft size={17}/></ActionIcon></Tooltip><Tooltip label="Ảnh tiếp (lưu nháp)"><ActionIcon variant="subtle" color="gray" aria-label="Ảnh tiếp (lưu nháp)" onClick={()=>navigate(index+1)} disabled={locked||index===images.length-1}><ArrowRight size={17}/></ActionIcon></Tooltip></div></div>
        {done?<div className="finished-card"><div><Check size={42}/></div><h2>Đã duyệt hết bộ ảnh</h2><p>Nhãn và bên tay đã được lưu. Bạn có thể xem lại ảnh hoặc mở Công cụ để xuất nhãn.</p><Button variant="light" onClick={()=>navigate(0)}>Xem lại ảnh</Button></div>:
        <div className="card-stage">
          <div className="card-back" aria-hidden="true"/>
          <article key={project+'/'+entry.image_id} id="photoCard" data-image-id={annotation.image_id} className={`photo-card ${leaving?'leaving':''}`} style={{transform:`translateX(${distance}px) rotate(${distance/35}deg)`,opacity:leaving?0:1}}>
            {Math.abs(distance)>28&&<div className={`swipe-stamp ${distance<0?'left':'right'}`}>{distance<0?'Tay trái':'Tay phải'}</div>}
            <PhotoCanvas key={project+'/'+entry.image_id} annotation={annotation} entry={entry} src={`/keypoints/assets/${project}/${entry.image_id}`} tool={tool} mode={mode} traces={traces} visible={overlay} disabled={locked} change={change} notice={notice} magnify={magnify} compact={!!message||!!current?.tutorial_only}/>
            <div className="card-caption"><span>{annotation.handedness==='unknown'?'Chưa chọn bên tay':`Đã lưu tay ${annotation.handedness==='left'?'trái':'phải'}`}</span><Badge size="sm" variant="light" color="gray">{labeled===4?'Đủ 4 mục':`${4-labeled} mục để trống`}</Badge></div>
          </article>
        </div>}
        <SwipeTray disabled={locked||done} choose={choose} distance={distance} setDistance={setDistance}/>
        <p className="swipe-note">Quẹt trái = tay trái. Quẹt phải = tay phải.<br/>Cả hai đều lưu ngay; phần chưa vẽ giữ trống.</p>
      </section>
      <aside className="annotation-summary">
        <h2>Trên ảnh này</h2>
        {TOOLS.slice(0,3).map(t=>{
          const line=annotation.lines[t.id],trace=traces[t.id];
          return <div key={t.id} className="summary-line"><span className="tool-color" style={{background:t.color}}/><div><strong>{t.name}</strong><span>{line.status==='absent'?'Nhãn cũ: không thấy':line.points.length===6?'Đã vẽ spline':line.points.length?`${line.points.length} điểm, chưa hoàn tất`:'Chưa vẽ'}</span>{trace?.valid&&trace.length_to_palm_ratio!=null&&<small>{trace.length_to_palm_ratio.toFixed(2)} × bề rộng</small>}{trace?.reasons?.length>0&&line.points.length===6&&<small className="geometry-warning">Kiểm tra lại điểm hoặc hướng vẽ</small>}</div></div>;
        })}
        <div className="width-summary"><Maximize2 size={18}/><div><strong>Bề rộng lòng bàn tay</strong><span>{annotation.palm_width_points.length===2?'Đã đặt 2 mép':annotation.palm_width_points.length===1?'Còn thiếu một mép':'Chưa đặt'}</span></div></div>
        <p className="blank-note">Không cần đoán những đường khó thấy. Phần chưa có dữ liệu được giữ trống.</p>
        <Button variant="default" size="sm" leftSection={<Download size={15}/>} onClick={downloadDraft}>Tải bản nháp JSON</Button>
        <Button variant="subtle" size="sm" leftSection={<RotateCcw size={15}/>} disabled={locked} onClick={reload}>Tải lại ảnh</Button>
      </aside>
    </div>}
    <ToolsDrawer opened={tools} close={()=>setTools(false)} state={state} project={project} annotation={annotation} locked={locked} change={change} refresh={refresh} persistDraft={persistDraft} reloadData={()=>fetchImage(session.current.index)} run={run} notice={notice} created={n=>{awaitedProject.current=n}}/>
  </main>;
}
