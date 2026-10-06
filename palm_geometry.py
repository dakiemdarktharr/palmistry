"""Conservative, image-space palm geometry; no personality inference.

Assumes a single open palm facing the camera, fingers upward. Handedness needs
known mirroring. All scores are heuristic, not calibrated probabilities.
"""
import heapq
import math

import cv2
import numpy as np

LINE_IDS = {2: 'life_line', 3: 'head_line', 4: 'heart_line', 6: 'simian_line'}


def palm_reference(hand_mask):
    binary = (hand_mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    if not components:
        return {'valid': False, 'reason': 'no_hand_region'}
    largest = components[0]
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < 64 or area > binary.size * .95:
        return {'valid': False, 'reason': 'invalid_hand_region'}
    if len(components) > 1 and stats[components[1], cv2.CC_STAT_AREA] > area * .25:
        return {'valid': False, 'reason': 'multiple_regions'}
    region = (labels == largest).astype(np.uint8)
    # Zero padding makes boundary distance independent of cropping/padding.
    distance = cv2.distanceTransform(np.pad(region, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    _, radius, _, center = cv2.minMaxLoc(distance)
    if radius < 4:
        return {'valid': False, 'reason': 'hand_region_too_narrow'}
    x,y,w,h,area = [int(v) for v in stats[largest]]
    return {'valid': True, 'method': 'largest_inscribed_palm_diameter',
            'diameter_px': float(radius * 2), 'center_xy': list(center),
            'bbox_xywh': [x,y,w,h], 'area_px': area,
            'limitation': '2D palm-width proxy; requires a valid hand silhouette and a frontal palm'}


def infer_handedness(hand_mask, mirrored=None):
    reference = palm_reference(hand_mask)
    result = {'side': 'unknown', 'thumb_side_image': 'unknown', 'heuristic_score': 0.0,
              'mirrored': mirrored, 'method': 'upright_open_palm_silhouette',
              'reason': reference.get('reason', 'ambiguous_thumb')}
    if not reference['valid']:
        return result
    x,y,w,h = reference['bbox_xywh']; cx,cy = reference['center_xy']
    radius = reference['diameter_px']/2
    binary = hand_mask > 0
    # Four separated fingers in the upper silhouette establish the upright pose.
    finger_runs = 0
    for fy in range(y + int(h*.16), y + int(h*.38), max(1,int(h*.03))):
        profile = binary[fy,x:x+w].astype(np.int8)
        starts = np.flatnonzero(np.diff(np.pad(profile,1)) == 1)
        ends = np.flatnonzero(np.diff(np.pad(profile,1)) == -1)
        finger_runs = max(finger_runs, sum((ends-starts) >= max(2,w*.035)))
    if finger_runs < 3 or h < w*.95:
        result['reason'] = 'pose_unknown_show_open_palm_fingers_up'
        return result
    lo,hi = max(y,int(cy-radius*.8)), min(y+h,int(cy+radius*.35))
    ys,xs = np.where(binary[lo:hi])
    if not len(xs):
        return result
    left, right = (cx-float(xs.min()))/radius, (float(xs.max())-cx)/radius
    margin = abs(left-right)
    if max(left,right) < 1.35 or margin < .30:
        return result
    thumb = 'left' if left > right else 'right'
    result.update(thumb_side_image=thumb, heuristic_score=float(min(.95, margin)), reason='mirror_metadata_required')
    if mirrored is None:
        return result
    side = 'right' if thumb == 'left' else 'left'
    if mirrored:
        side = 'left' if side == 'right' else 'right'
    result.update(side=side, reason='geometric_estimate_not_validated_on_real_hands')
    return result


def principal_trace(skeleton):
    """Longest endpoint-to-endpoint shortest path of the largest component.

    Weighted 8-neighbour graph; diagonal corner shortcuts are excluded. For
    branching components this is a principal-path estimate, not summed branches.
    Cycles and dense/ambiguous skeletons are rejected.
    """
    binary=(skeleton>0).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    ids=sorted(range(1,n),key=lambda i:stats[i,cv2.CC_STAT_AREA],reverse=True)
    if not ids:
        return {'detected':False,'reason':'no_trace'}
    points=[tuple(p) for p in np.argwhere(labels==ids[0])]
    if len(points)<5 or len(points)>25000:
        return {'detected':False,'reason':'trace_size_out_of_range'}
    index={p:i for i,p in enumerate(points)}; graph=[[] for _ in points]
    for i,(y,x) in enumerate(points):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dy or dx): continue
                j=index.get((y+dy,x+dx))
                if j is None: continue
                if dy and dx and ((y,x+dx) in index or (y+dy,x) in index): continue
                graph[i].append((j,math.hypot(dx,dy)))
    ends=[i for i,edges in enumerate(graph) if len(edges)==1]
    if len(ends)<2 or len(ends)>32:
        return {'detected':False,'reason':'ambiguous_branching_or_closed_trace'}
    def distances(start):
        dist={start:0.0}; parent={}; q=[(0.0,start)]
        while q:
            d,i=heapq.heappop(q)
            if d!=dist[i]: continue
            for j,weight in graph[i]:
                value=d+weight
                if value<dist.get(j,float('inf')):
                    dist[j]=value;parent[j]=i;heapq.heappush(q,(value,j))
        return dist,parent
    best=(-1,None,None,None)
    for start in ends:
        dist,parent=distances(start)
        finish=max(ends,key=lambda i:dist.get(i,-1))
        if dist[finish]>best[0]:best=(dist[finish],start,finish,parent)
    length,start,finish,parent=best; path=[finish]
    while path[-1]!=start:path.append(parent[path[-1]])
    path.reverse()
    coverage=len(path)/len(points)
    total=int(binary.sum())
    if coverage<.60 or len(points)/max(1,total)<.65:
        return {'detected':False,'reason':'fragmented_or_branched_trace', 'principal_path_coverage':coverage}
    return {'detected':True,'length_px':float(length),'trace_xy':[[int(points[i][1]),int(points[i][0])] for i in path],
            'component_count':len(ids),'principal_path_coverage':coverage,'length_method':'weighted_geodesic_principal_path'}


def measure_line(semantic, class_id, skeletonize):
    reference=palm_reference(semantic>0)
    if not reference['valid']:
        return {'detected':False,'reason':reference['reason']}
    trace=principal_trace(skeletonize((semantic==class_id).astype(np.uint8)*255))
    if not trace['detected']:return trace
    ratio=trace['length_px']/reference['diameter_px']
    if ratio<.08:
        return {'detected':False,'reason':'trace_too_short'}
    points=np.asarray(trace["trace_xy"],dtype=np.float64)
    trace.update(length_norm=ratio, length_to_palm_ratio=ratio, normalization=reference,
                 center_x=float(points[:,0].mean()/semantic.shape[1]),
                 center_y=float(points[:,1].mean()/semantic.shape[0]))
    return trace


def resolve_transverse_lines(lines):
    """A reported simian line must not duplicate independently reported transverse lines."""
    if lines.get('simian_line',{}).get('detected') and any(lines.get(k,{}).get('detected') for k in ('head_line','heart_line')):
        for key in ('head_line','heart_line','simian_line'):
            lines[key]={'detected':False,'reason':'ambiguous_transverse_pattern'}
    return lines
