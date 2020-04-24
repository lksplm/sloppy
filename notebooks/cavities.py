import numpy as np
import k3d
import matplotlib.pyplot as plt
from sloppy.optic import *
from sloppy.raytracing import RaySystem
from sloppy.utils import *

def SixMirror(dx=27.77, dy=8.0, dz=16.685, d=4.750, dzF=1.5825, Rfast=25.0):

    p5 = np.array([0,0,0])
    p6 = np.array([dx, dy, dzF])
    p1 = np.array([0, dy, dzF])
    p2 = np.array([dx, 2*dy, 0])
    p3 = np.array([d, dy+d, dz])
    p4 = np.array([dx-d, dy-d, dz])
    ps = np.stack([p1,p2,p3,p4,p5,p6], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = 0.5*(geom['yin']+geom['yout'])
    
    hi = 12.7
    qi=7.75
    #reference plane is expected between first and last element!
    elements = [CurvedMirror(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=qi, R=Rfast, curv='CC', thet=angles[0]),\
                Mirror(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=qi),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=qi),\
                CurvedMirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=qi, R=Rfast, curv='CC', thet=angles[5])]
    
    return elements

def GravEM(l=51.0, theta=20., axialL00 = 51.0, R=100.0):
    theta = np.deg2rad(theta)
    l00 = axialL00/np.sqrt(1+2*np.tan(theta/2)**2)
    d1 = l00*np.tan(theta/2)
    d2 = d1
    p1 = np.array([-d1, 0, -l/2])
    p2 = np.array([0, -d2, l/2])
    p3 = np.array([d1, 0, -l/2])
    p4 = np.array([0, d2, l/2])
    ps = np.stack([p1,p2,p3,p4], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = 0.5*(geom['yin']+geom['yout'])
    
    hi = 12.7
    qi=7.75
    #reference plane is expected between first and last element!
    elements = [CurvedMirror(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=qi, R=R, curv='CC', thet=angles[0]),\
                CurvedMirror(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=qi, R=R, curv='CC', thet=angles[0]),\
                CurvedMirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=qi, R=R, curv='CC', thet=angles[0]),\
                CurvedMirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=qi, R=R, curv='CC', thet=angles[0])]
    
    return elements

def Lens_cav(arm1=50., arm2=55., base=19., angle=150., lens_dist=11.075, lens_diam=6.35, lens_thick=4., Rlens=5.0):
    angle = np.deg2rad(angle)
    p0 = np.array([lens_dist/2.-lens_thick/2.,0,0])
    p1 = np.array([lens_dist/2.+lens_thick/2.,0,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_thick/2.,0,0])
    p7 = np.array([-lens_dist/2.+lens_thick/2.,0,0])
    ps = np.stack([p0,p1,p2,p3,p4,p5,p6,p7], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = 0.5*(geom['yin']+geom['yout'])
    
    #fix normal vectosfor transmission
    ns[0,:] = np.array([-1.,0,0])
    ns[1,:] = np.array([-1.,0,0])
    ns[6,:] = np.array([-1.,0,0])
    ns[7,:] = np.array([-1.,0,0])
    #fix axis transmission
    ax_x[[0,1,6,7],:] = np.array([0,-1.,0])
    ax_y[[0,1,6,7],:] = np.array([0,0,1.0])
    #fix basis trafo in transmission
    Rtr[[0,1,6,7]] = np.eye(4)
    
    hi = 12.7
    qi=7.75
    ng = 1.41
    elements = [Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=Rlens, curv='CC', n1=ng),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                CurvedGlass(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng)]
    return elements

def OriginalTwister(betal, R=25., Rlarge=-75., thet=10., asym = 1.25):
    thet = np.deg2rad(thet)
    l = betal*R
    d1 = l*np.tan(thet/2)
    d2 = d1
    p1 = np.array([-d1, 0, -l/2])
    p2 = np.array([0, -d2, l/2])/asym
    p3 = np.array([d1, 0, -l/2])/asym
    p4 = np.array([0, d2, l/2])
    
    ax = 0.5*(p2+p3) - 0.5*(p1+p4)
    p2 += 0.25*ax
    p3 += 0.25*ax
    
    ps = np.stack([p1,p2,p3,p4], axis=0)
    roc = np.array([R, Rlarge, Rlarge, R])
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = 0.5*(geom['yin']+geom['yout'])
    
    hi = 12.7
    qi=7.75
    #reference plane is expected between first and last element!
    elements = [CurvedMirror(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=qi, R=R, curv='CC', thet=angles[0]),\
                CurvedMirror(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=qi, R=Rlarge, curv='CX', thet=angles[1]),\
                CurvedMirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=qi, R=Rlarge, curv='CX', thet=angles[2]),\
                CurvedMirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=qi, R=R, curv='CC', thet=angles[3])]
          
    return elements