import React, { useRef, useEffect } from 'react';
import { useMove } from 'react-aria/useMove';
import { ArrowLeft, ArrowRight, GripHorizontal } from 'lucide-react';
import { Button } from '@mantine/core';
import { swipeSide } from './domain.js';

export default function SwipeTray({ disabled, choose, distance, setDistance }) {
  const position=useRef(0),vertical=useRef(0),cancelled=useRef(false);
  useEffect(()=>{
    const cancel=()=>{cancelled.current=true;setDistance(0)};
    window.addEventListener('pointercancel',cancel,true);
    return()=>window.removeEventListener('pointercancel',cancel,true);
  },[setDistance]);
  const {moveProps}=useMove({
    onMoveStart(){cancelled.current=false;position.current=0;vertical.current=0;},
    onMove(e){if(disabled||e.pointerType==='keyboard')return;vertical.current+=e.deltaY;position.current=Math.max(-260,Math.min(260,position.current+e.deltaX));setDistance(position.current);},
    onMoveEnd(){const hand=Math.abs(position.current)>Math.abs(vertical.current)*1.4?swipeSide(position.current):null;position.current=0;if(!disabled&&!cancelled.current&&hand)choose(hand);else setDistance(0);},
  });
  const side=distance<0?'Tay trái':'Tay phải';
  return <div className="swipe-actions">
    <Button id="chooseLeft" variant="outline" className="hand-button left-hand" leftSection={<ArrowLeft size={19}/>} disabled={disabled} onClick={()=>choose('left')}>Tay trái</Button>
    <div {...moveProps} id="swipeHandle" className="swipe-handle" tabIndex={disabled?-1:0} role="group"
      aria-label="Quẹt để lưu bên tay. Phím mũi tên trái là tay trái, mũi tên phải là tay phải."
      aria-disabled={disabled} onPointerCancelCapture={()=>{cancelled.current=true;setDistance(0)}}
      onKeyDown={e=>{if(disabled)return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();choose(e.key==='ArrowLeft'?'left':'right');}else moveProps.onKeyDown?.(e)}}>
      <GripHorizontal size={22}/><span>{Math.abs(distance)>28?`${side} · ${Math.abs(distance)>=96?'Thả để lưu':'Kéo thêm để lưu'}`:'Quẹt ở đây để lưu'}</span>
      <small>← trái &nbsp; / &nbsp; phải →</small>
    </div>
    <Button id="chooseRight" className="hand-button right-hand" rightSection={<ArrowRight size={19}/>} disabled={disabled} onClick={()=>choose('right')}>Tay phải</Button>
  </div>;
}
