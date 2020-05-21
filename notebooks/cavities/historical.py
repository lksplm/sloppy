import numpy as np
from sloppy.optic import *
from sloppy.utils import *

coeffc2 = lambda x: 1./(2*x)
coeffc4 = lambda x: 1./(8*x**3)
coeffc6 = lambda x: 1./(16*x**5)
coeffc8 = lambda x: 5./(128*x**7)


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
                CurvedMirror(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=qi, R=R, curv='CC', thet=angles[1]),\
                CurvedMirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=qi, R=R, curv='CC', thet=angles[2]),\
                CurvedMirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=qi, R=R, curv='CC', thet=angles[3])]
    
    return elements


def SyntheticLandauLevels(betal=18.16, R=25., Rlarge=50., thet=32., asym = 1., shift=0.):
    thet = np.deg2rad(thet)
    l = betal
    d1 = l*np.tan(thet/2)
    d2 = d1
    p1 = np.array([-d1, 0, -l/2])
    p2 = np.array([0, -d2, l/2])/asym
    p3 = np.array([d1, 0, -l/2])/asym
    p4 = np.array([0, d2, l/2])
    
    ax = 0.5*(p2+p3) - 0.5*(p1+p4)
    p2 += shift*ax
    p3 += shift*ax
    
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
                CurvedMirror(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=qi, R=Rlarge, curv='CC', thet=angles[1]),\
                CurvedMirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=qi, R=Rlarge, curv='CC', thet=angles[2]),\
                CurvedMirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=qi, R=R, curv='CC', thet=angles[3])]
          
    return elements

def OriginalTwister(betal=31.67, R=25., Rlarge=-75., thet=20., asym = 1.25):
    thet = np.deg2rad(thet)
    l = betal#*R
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

