"""One ordered open B-spline per line; never skeletons, branches or mask contours."""
import warnings
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad, IntegrationWarning
from . import N_POINTS


def _cross(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def self_intersects(points):
    p=np.asarray(points,float)
    for i in range(len(p)-3):
        a,b=p[i],p[i+1];c,d=p[i+2:-1],p[i+3:]
        ab=b-a;cd=d-c
        v1=_cross(ab,c-a);v2=_cross(ab,d-a)
        v3=_cross(cd,a-c);v4=_cross(cd,b-c)
        if np.any((v1*v2 < -1e-10)&(v3*v4 < -1e-10)):return True
    return False


def trace_line(keypoints, image_size, palm_width_px, confidence=None, min_confidence=.90,
               max_curvature=35.0, smoothing=0.0):
    """Landmarks are normalized image coordinates; numerical length is in image px.

    Missing points are rejected. A polyline may be returned solely for manual preview.
    Curvature is made scale independent by multiplying by palm-width pixels.
    """
    result={'valid':False,'reasons':[],'method':'cubic_bspline','confidence_semantics':'uncalibrated_ranking'}
    try:p=np.asarray(keypoints,dtype=float);w,h=map(int,image_size)
    except (ValueError,TypeError):return dict(result,reasons=['invalid_keypoints'])
    if w<2 or h<2 or p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():
        return dict(result,reasons=['invalid_keypoints'])
    if len(p)!=N_POINTS:return dict(result,reasons=['missing_keypoints'],preview_polyline=p.tolist())
    if np.any((p<0)|(p>1)):return dict(result,reasons=['out_of_bounds'])
    if not np.isfinite(palm_width_px) or palm_width_px<4:
        return dict(result,reasons=['invalid_palm_reference'])
    conf=np.ones(N_POINTS) if confidence is None else np.asarray(confidence,float)
    if conf.shape!=(N_POINTS,) or not np.isfinite(conf).all() or np.any((conf<0)|(conf>1)):
        return dict(result,reasons=['invalid_confidence'])
    if float(conf.min())<min_confidence:result['reasons'].append('low_keypoint_confidence')
    pixels=p*np.array([w-1,h-1]);delta=np.diff(pixels,axis=0);steps=np.linalg.norm(delta,axis=1)
    if float(steps.min())<max(.25,palm_width_px*.002):
        return dict(result,reasons=result['reasons']+['duplicate_or_collapsed_keypoints'],preview_polyline=p.tolist())
    angles=np.arccos(np.clip(np.sum(delta[:-1]*delta[1:],axis=1)/(steps[:-1]*steps[1:]),-1,1))
    if np.max(angles)>np.deg2rad(135):result['reasons'].append('keypoint_order_or_sharp_turn')
    if self_intersects(pixels):result['reasons'].append('keypoint_self_intersection')
    if result['reasons'] and any(r!='low_keypoint_confidence' for r in result['reasons']):
        result['preview_polyline']=p.tolist();return result
    try:
        (tck,u),residual,ier,message=splprep(pixels.T,s=float(smoothing),k=3,per=False,full_output=True)
        if ier>0:raise ValueError(message)
        sample_u=np.linspace(0,1,257);curve=np.asarray(splev(sample_u,tck)).T
        d1=np.asarray(splev(sample_u,tck,der=1)).T;d2=np.asarray(splev(sample_u,tck,der=2)).T
        speed=np.linalg.norm(d1,axis=1)
        curvature=np.abs(_cross(d1,d2))/np.maximum(speed**3,1e-12)*palm_width_px
        with warnings.catch_warnings():
            warnings.simplefilter('error',IntegrationWarning)
            length,error=quad(lambda v:float(np.linalg.norm(splev(v,tck,der=1))),0,1,
                              points=np.unique(tck[0][(tck[0]>0)&(tck[0]<1)]),epsabs=1e-6,epsrel=1e-7,limit=200)
    except (ValueError,TypeError,IntegrationWarning,FloatingPointError) as exc:
        return dict(result,reasons=result['reasons']+['spline_fit_failed'],detail=str(exc),preview_polyline=p.tolist())
    if not np.isfinite(curve).all() or not np.isfinite(length) or not np.isfinite(curvature).all():
        result['reasons'].append('nonfinite_curve')
    if np.any(curve < -.01) or np.any(curve > np.array([w-1,h-1])+.01):result['reasons'].append('spline_overshoot')
    if float(speed.min())<1e-5 or float(curvature.max())>max_curvature:result['reasons'].append('excessive_curvature_or_cusp')
    if self_intersects(curve):result['reasons'].append('curve_self_intersection')
    if length/palm_width_px < .08 or length/palm_width_px>3.5:result['reasons'].append('implausible_length')
    if error>max(1e-4,length*1e-5):result['reasons'].append('arc_length_not_converged')
    result.update(valid=not result['reasons'],curve_xy=curve.tolist(),length_px=float(length),
                  length_to_palm_ratio=float(length/palm_width_px),integration_error_px=float(error),
                  max_normalized_curvature=float(curvature.max()),min_confidence=float(conf.min()),
                  interpolation_residual=float(residual),smoothing=float(smoothing),
                  connected_paths=1,closed=False,branch_count=0)
    return result


def width_from_points(points,image_size):
    p=np.asarray(points,float)
    if p.shape!=(2,2) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Cần 2 điểm chuẩn bề rộng lòng bàn tay trong ảnh.')
    width=float(np.linalg.norm((p[1]-p[0])*(np.asarray(image_size)-1)))
    if width<4:raise ValueError('Hai điểm chuẩn lòng bàn tay quá gần nhau.')
    return width
