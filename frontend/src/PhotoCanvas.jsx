import React, { useEffect, useRef, useState } from 'react';
import { useViewportSize } from '@mantine/hooks';
import { pointsFor, withPoints, sampleStroke, TOOLS } from './domain.js';

export default function PhotoCanvas({ annotation, entry, src, tool, mode, traces, visible, disabled, change, notice, magnify, compact }) {
  const canvas = useRef(null), image = useRef(null), gesture = useRef(null);
  const [stroke, setStroke] = useState([]);
  const [ready, setReady] = useState(false);
  const [cursor, setCursor] = useState(null);
  const zoomCanvas = useRef(null);
  const viewport=useViewportSize();
  const height=viewport.height||window.innerHeight;
  const photoHeight=Math.max(150,Math.min(height*.56,height-(compact?490:435)));
  useEffect(() => {
    setReady(false);gesture.current=null;setStroke([]);setCursor(null);
    const img = new Image();
    img.onload = () => { image.current=img;setReady(true); };
    img.onerror = () => notice('Không tải được ảnh. Nhấn Tải lại ảnh.');
    img.src=src;
    return () => { img.onload=null;img.onerror=null; };
  }, [src]);

  useEffect(() => {
    const c=canvas.current, ctx=c.getContext('2d');
    ctx.clearRect(0,0,c.width,c.height);
    if (!ready) return;
    ctx.drawImage(image.current,0,0,c.width,c.height);
    if (visible) TOOLS.forEach(t => {
      const points=pointsFor(annotation,t.id), curve=t.id==='width'?null:traces[t.id]?.curve_xy;
      const coordinates=curve || points.map(([x,y])=>[x*(c.width-1),y*(c.height-1)]);
      ctx.strokeStyle=t.color;ctx.lineWidth=Math.max(2,c.width/280);ctx.lineCap='round';ctx.lineJoin='round';
      ctx.setLineDash(curve?[]:[5,4]);
      if(coordinates.length>1){ctx.beginPath();coordinates.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();}
      ctx.setLineDash([]);
      points.forEach(([x,y],i)=>{
        x*=c.width-1;y*=c.height-1;ctx.fillStyle=t.color;ctx.strokeStyle='#fff';ctx.lineWidth=2;
        ctx.beginPath();ctx.arc(x,y,Math.max(4,c.width/100),0,Math.PI*2);ctx.fill();ctx.stroke();
        if (tool===t.id) {ctx.font=`600 ${Math.max(13,c.width/42)}px Segoe UI`;ctx.lineWidth=3;ctx.strokeText(String(i+1),x+8,y-8);ctx.fillText(String(i+1),x+8,y-8);}
      });
    });
    if (stroke.length>1) {
      ctx.strokeStyle=TOOLS.find(t=>t.id===tool).color;ctx.lineWidth=Math.max(2,c.width/240);
      ctx.beginPath();stroke.forEach(([x,y],i)=>i?ctx.lineTo(x*(c.width-1),y*(c.height-1)):ctx.moveTo(x*(c.width-1),y*(c.height-1)));ctx.stroke();
    }
    if (cursor && magnify && zoomCanvas.current) {
      const z=zoomCanvas.current.getContext('2d');z.clearRect(0,0,140,140);
      z.drawImage(c,cursor[0]*(c.width-1)-28,cursor[1]*(c.height-1)-28,56,56,0,0,140,140);
      z.strokeStyle='#fff';z.lineWidth=1;z.beginPath();z.moveTo(62,70);z.lineTo(78,70);z.moveTo(70,62);z.lineTo(70,78);z.stroke();
    }
  }, [annotation, tool, traces, visible, ready, stroke, cursor, magnify]);

  const coord=e=>{
    const r=canvas.current.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];
  };
  function down(e) {
    if (disabled||!ready||gesture.current) return;
    if (e.pointerType==='mouse'&&e.button!==0) return;
    e.preventDefault();e.currentTarget.setPointerCapture(e.pointerId);
    const p=coord(e), points=pointsFor(annotation,tool),r=canvas.current.getBoundingClientRect();
    const index=points.findIndex(q=>Math.hypot((p[0]-q[0])*r.width,(p[1]-q[1])*r.height)<13);
    gesture.current={pointer:e.pointerId,index,start:structuredClone(annotation),path:[p],points:structuredClone(points)};
    if(index>=0) return;
    if(tool==='width'||mode==='points') {
      if(points.length<(tool==='width'?2:6)){
        gesture.current.index=points.length;gesture.current.points.push(p);
        change(withPoints(annotation,tool,gesture.current.points),false);
      }
    } else setStroke([p]);
  }
  function move(e) {
    const p=coord(e);setCursor(p);
    const g=gesture.current;if(!g||g.pointer!==e.pointerId||disabled)return;
    if(g.index>=0){g.points[g.index]=p;change(withPoints(g.start,tool,g.points),false);}
    else if(tool!=='width'&&mode==='draw') {
      const last=g.path.at(-1);
      if(Math.hypot((p[0]-last[0])*entry.width,(p[1]-last[1])*entry.height)>2){g.path.push(p);setStroke([...g.path]);}
    }
  }
  function up(e) {
    const g=gesture.current;if(!g||g.pointer!==e.pointerId)return;gesture.current=null;
    if(g.index>=0) change(withPoints(g.start,tool,g.points),true,g.start);
    else if(tool!=='width'&&mode==='draw') {
      const path=[...g.path,coord(e)],points=sampleStroke(path,entry.width,entry.height);
      if(points)change(withPoints(g.start,tool,points),true,g.start);
      else notice('Kéo từ đầu đến cuối đường để vẽ; chọn Đặt điểm nếu muốn bấm từng điểm.');
    }
    setStroke([]);
  }
  return <div className="photo-surface" style={{width:`min(100%, ${entry.width/entry.height*photoHeight}px)`}}>
    <canvas id="canvas" ref={canvas} width={entry.width} height={entry.height}
      aria-label={`Ảnh lòng bàn tay, công cụ ${TOOLS.find(t=>t.id===tool).name}`} role="img"
      onPointerDown={down} onPointerMove={move} onPointerUp={up}
      onPointerCancel={e=>{const g=gesture.current;if(g&&g.pointer===e.pointerId){change(g.start,false);gesture.current=null;setStroke([]);}}}
      onPointerLeave={()=>{if(!gesture.current)setCursor(null)}} />
    {cursor&&magnify&&ready&&<canvas ref={zoomCanvas} className="magnifier" width={140} height={140} aria-hidden="true"/>}
    {!ready&&<div className="photo-loading">Đang tải ảnh…</div>}
  </div>;
}
