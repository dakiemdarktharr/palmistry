import React, { useState } from 'react';
import { Accordion, Alert, Button, Drawer, FileInput, Group, NumberInput, Select, Stack, Textarea, TextInput } from '@mantine/core';
import { Download, FolderPlus, Play, Upload } from 'lucide-react';
import { request } from './domain.js';

export default function ToolsDrawer({opened,close,state,project,annotation,locked,change,refresh,persistDraft,reloadData,run,notice,created}) {
  const [source,setSource]=useState(null),[name,setName]=useState('palm_'+new Date().toISOString().replace(/[^0-9]/g,'').slice(0,14));
  const [metadata,setMetadata]=useState(null),[epochs,setEpochs]=useState(60),[batch,setBatch]=useState(8);
  const [limit,setLimit]=useState(100),[reports,setReports]=useState([]),[report,setReport]=useState(null),[offset,setOffset]=useState(0);
  const [image,setImage]=useState(null),[prediction,setPrediction]=useState(null);
  const jobRunning=state.job.status==='running';
  const action=kind=>run(async()=>{
    await persistDraft();await request('/api/keypoints/action',{project,kind,epochs:Number(epochs),batch:Number(batch),limit:Number(limit)});
    await refresh();notice('Đã bắt đầu tác vụ. Tiến độ nằm trong Công cụ.');
  });
  async function formRequest(url,form) {
    const response=await fetch(url,{method:'POST',headers:{'X-Palm-CSRF':window.PALM_BOOT.csrf},body:form});
    const result=await response.json();if(!response.ok||result.ok===false)throw Error(result.error||'Tác vụ thất bại.');return result;
  }
  return <Drawer opened={opened} onClose={close} title="Công cụ & dữ liệu" position="right" size="lg"><Stack gap="lg">
    {state.job.status!=='idle'&&<Alert color={state.job.status==='failed'?'red':'violet'} title={state.job.status==='running'?'Tác vụ đang chạy':state.job.status==='completed'?'Tác vụ hoàn thành':state.job.status}><pre className="job-log">{[state.job.error,...(state.job.log||[]).slice(-8)].filter(Boolean).join('\n')}</pre></Alert>}
    <Accordion variant="separated" defaultValue="data">
      <Accordion.Item value="data"><Accordion.Control>Tạo bộ 100 ảnh</Accordion.Control><Accordion.Panel><Stack gap="sm">
        <p className="drawer-help">Ảnh gốc được giữ nguyên. Bộ mới có 100 ảnh; phần còn lại dành cho bước sau.</p>
        <Select label="Nguồn ảnh" data={state.runs} value={source||state.runs.at(-1)||null} onChange={setSource} placeholder="Chọn queue nguồn"/>
        <TextInput label="Tên bộ ảnh" value={name} onChange={e=>setName(e.currentTarget.value)} maxLength={70}/>
        <Button leftSection={<FolderPlus size={16}/>} disabled={locked||jobRunning||!state.runs.length} onClick={()=>run(async()=>{
          await persistDraft();await request('/api/keypoints/create',{run:source||state.runs.at(-1),name,count:100});created(name);await refresh();notice('Đang tạo bộ 100 ảnh…');
        })}>Tạo bộ ảnh</Button>
        <Button component="a" href="/" variant="subtle">Nhập ZIP/RAR ở màn hình chính</Button>
      </Stack></Accordion.Panel></Accordion.Item>
      <Accordion.Item value="metadata"><Accordion.Control>Thông tin bổ sung</Accordion.Control><Accordion.Panel><Stack gap="sm">
        <p className="drawer-help">Tùy chọn khi vẽ. Khi huấn luyện, ảnh cùng người cần cùng mã để tránh chia lẫn tập kiểm tra.</p>
        {annotation&&<><TextInput label="Mã người" value={annotation.subject_id} disabled={locked} onChange={e=>change({...annotation,subject_id:e.currentTarget.value})}/><TextInput label="Nhóm nguồn / lần chụp" value={annotation.source_group} disabled={locked} onChange={e=>change({...annotation,source_group:e.currentTarget.value})}/><Textarea label="Ghi chú" value={annotation.notes} disabled={locked} onChange={e=>change({...annotation,notes:e.currentTarget.value})}/></>}
        <FileInput label="CSV metadata" description="source_path,subject_id; source_group tùy chọn" accept=".csv" value={metadata} onChange={setMetadata}/>
        <Button variant="light" leftSection={<Upload size={16}/>} disabled={locked||jobRunning||!metadata||!project} onClick={()=>run(async()=>{
          await persistDraft();const form=new FormData();form.append('file',metadata);const result=await formRequest('/api/keypoints/metadata/'+project,form);await reloadData();await refresh();notice(`Đã nhập ${result.metadata_rows} dòng metadata.`);
        })}>Nhập CSV</Button>
      </Stack></Accordion.Panel></Accordion.Item>
      <Accordion.Item value="training"><Accordion.Control>Huấn luyện & xuất nhãn</Accordion.Control><Accordion.Panel><Stack gap="sm">
        <p className="drawer-help">Nhãn thiếu được bỏ qua trong loss. Mỗi tập train/val/test phải có nhóm người hoặc nguồn độc lập.</p>
        <Group grow><NumberInput label="Epoch" value={epochs} onChange={setEpochs} min={1} max={500}/><NumberInput label="Batch" value={batch} onChange={setBatch} min={1} max={32}/></Group>
        <Button leftSection={<Play size={16}/>} disabled={locked||jobRunning||!project} onClick={()=>action('train')}>Huấn luyện mô hình 3 đường</Button>
        <Button variant="light" disabled={locked||jobRunning||!project} onClick={()=>action('export-yolo')}>Xuất YOLO Pose (20 điểm)</Button>
        <Button leftSection={<Download size={16}/>} variant="default" disabled={locked||!project} onClick={()=>run(async()=>{await persistDraft();location.href='/api/keypoints/annotations-export/'+project})}>Tải toàn bộ nhãn JSON</Button>
      </Stack></Accordion.Panel></Accordion.Item>
      <Accordion.Item value="pseudo"><Accordion.Control>Nhãn đề xuất & báo cáo</Accordion.Control><Accordion.Panel><Stack gap="sm">
        <NumberInput label="Số ảnh đề xuất (0 = tất cả)" value={limit} onChange={setLimit} min={0} max={1000000}/>
        <Button variant="light" disabled={locked||jobRunning||!project} onClick={()=>action('pseudo')}>Tạo nhãn đề xuất</Button>
        <Button variant="default" disabled={locked||!project} onClick={()=>run(async()=>{const result=await request('/api/keypoints/report/'+project);setReports(result.runs);})}>Xem báo cáo</Button>
        <Select label="Run đề xuất" data={reports.filter(r=>r.name.startsWith('pseudo_')).map(r=>r.name)} value={report} onChange={setReport}/>
        <NumberInput label="Bắt đầu từ dòng" value={offset} onChange={setOffset} min={0} max={1000000} step={100}/>
        <Button variant="light" disabled={locked||jobRunning||!report} onClick={()=>run(async()=>{await persistDraft();const n=project.slice(0,40)+'_review_'+Date.now();await request('/api/keypoints/review-pseudo',{project,run:report,offset:Number(offset),name:n});created(n);await refresh();})}>Mở 100 đề xuất để duyệt</Button>
        {reports.length>0&&<pre className="job-log">{JSON.stringify(reports,null,2)}</pre>}
      </Stack></Accordion.Panel></Accordion.Item>
      <Accordion.Item value="predict"><Accordion.Control>Thử nhận diện ảnh mới</Accordion.Control><Accordion.Panel><Stack gap="sm">
        <FileInput id="predictImage" label="Ảnh lòng bàn tay" accept="image/*" value={image} onChange={setImage}/>
        <Button disabled={locked||!project||!image} onClick={()=>run(async()=>{const form=new FormData();form.append('image',image);form.append('project',project);const result=await formRequest('/api/keypoints/predict',form);setPrediction(result.prediction);})}>Nhận diện 3 đường</Button>
        {prediction&&<pre className="job-log">{JSON.stringify(prediction,null,2)}</pre>}
      </Stack></Accordion.Panel></Accordion.Item>
    </Accordion>
  </Stack></Drawer>;
}
