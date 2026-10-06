"""Primary Heart/Head/Life/Fate keypoint CLI. Also available at /keypoints."""
import sys
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,"reconfigure"):stream.reconfigure(encoding="utf-8",errors="replace")
import argparse
import json
from pathlib import Path
from palm_keypoints.data import create_from_review,create_from_pseudo,export_yolo,write_json,limit_review_queue
from palm_keypoints.workflow import train,pseudo,infer
from palm_keypoints.model import PoseCNN
from prototype_common import read_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    c=subs.add_parser('from-review');c.add_argument('--run',required=True);c.add_argument('--out',required=True);c.add_argument('--count',type=int,default=100)
    c=subs.add_parser('review-pseudo');c.add_argument('--report',required=True);c.add_argument('--out',required=True);c.add_argument('--count',type=int,default=100);c.add_argument('--offset',type=int,default=0)
    c=subs.add_parser('limit-review');c.add_argument('--project',required=True);c.add_argument('--count',type=int,default=100)
    c=subs.add_parser('train');c.add_argument('--project',required=True);c.add_argument('--out',required=True);c.add_argument('--epochs',type=int,default=60);c.add_argument('--batch-size',type=int,default=8)
    c=subs.add_parser('export-yolo');c.add_argument('--project',required=True);c.add_argument('--out',required=True)
    c=subs.add_parser('pseudo');c.add_argument('--project',required=True);c.add_argument('--out',required=True);c.add_argument('--checkpoint',required=True);c.add_argument('--sha256',required=True);c.add_argument('--limit',type=int,default=0)
    c=subs.add_parser('predict');c.add_argument('--checkpoint',required=True);c.add_argument('--sha256',required=True);c.add_argument('--image',required=True);c.add_argument('--out',required=True);c.add_argument('--mirrored',choices=['unknown','yes','no'],default='unknown')
    a=p.parse_args()
    if a.command=='from-review':r=create_from_review(a.run,a.out,a.count)
    elif a.command=='review-pseudo':r=create_from_pseudo(a.report,a.out,a.count,a.offset)
    elif a.command=='limit-review':r=limit_review_queue(a.project,a.count)
    elif a.command=='train':r=train(a.project,a.out,a.epochs,a.batch_size)
    elif a.command=='pseudo':r=pseudo(a.project,a.checkpoint,a.sha256,a.out,a.limit)
    elif a.command=='export-yolo':r=export_yolo(a.project,a.out)
    else:r=infer(PoseCNN.load(a.checkpoint,a.sha256),read_rgb(a.image),a.mirrored)
    if a.command=='predict':write_json(a.out,r)
    elif a.command!='limit-review':write_json(Path(a.out)/'summary.json',r)
    print(json.dumps(r,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
