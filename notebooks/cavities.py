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
                CurvedMirror(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=qi, R=R, curv='CC', thet=angles[1]),\
                CurvedMirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=qi, R=R, curv='CC', thet=angles[2]),\
                CurvedMirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=qi, R=R, curv='CC', thet=angles[3])]
    
    return elements

def LensCav_old(arm1=50., arm2=55., base=19., angle=150., lens_dist=21.52449, lens_diam=6.35, lens_thick=4., Rlens=5.0): #still needed manual fixing...
    angle = np.deg2rad(angle)
    eps = 1e-4 #hack
    p0 = np.array([lens_dist/2.-lens_thick/2.,eps,0])
    p1 = np.array([lens_dist/2.+lens_thick/2.,-eps,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_thick/2.,-eps,0])
    p7 = np.array([-lens_dist/2.+lens_thick/2.,eps,0])
    ps = np.stack([p0,p1,p2,p3,p4,p5,p6,p7], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = 0.5*(geom['yin']+geom['yout'])
    
    #fix normal vectosfor transmission
    ns[[0,1,6,7],:] = np.array([-1.,0,0])
    #fix axis transmission
    ax_x[[0,1,6,7],:] = np.array([0,-1.,0])
    ax_y[[0,1,6,7],:] = np.array([0,0,1.0])
    #fix basis trafo in transmission
    Rtr[[0,1,6,7]] = np.eye(4)
    
    hi = 12.7
    qi=7.75
    ng = 1.4537#1.41
     #negative sign of firstt cuved surface for abcd matrix
    elements = [Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                CurvedGlass(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng)]
    return elements

def LensCav(arm1=50., arm2=55., base=19., angle=150., lens_dist=21.52449, lens_diam=6.35, lens_thick=4., Rlens=5.0):
    angle = np.deg2rad(angle)
    eps = 0. #1e-4 #hack
    p0 = np.array([lens_dist/2.-lens_thick/2.,eps,0])
    p1 = np.array([lens_dist/2.+lens_thick/2.,-eps,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_thick/2.,-eps,0])
    p7 = np.array([-lens_dist/2.+lens_thick/2.,eps,0])
    ps = np.stack([p0,p1,p2,p3,p4,p5,p6,p7], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
        
    hi = 12.7
    qi=7.75
    ng = 1.4537
     #negative sign of firstt cuved surface for abcd matrix
    elements = [Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                CurvedGlass(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng)]
    return elements

def LensCavQuartic(arm1=50., arm2=55., base=19., angle=150., lens_dist=22.15, lens_diam=6.35, lens_thick=4., Rlens=5.0, c4=0., quart_thick=1.):
    angle = np.deg2rad(angle)
    eps = 0. #1e-4 #hack
    pa = np.array([lens_dist/2.-lens_thick/2.-1.-quart_thick, 0, 0])
    pb = np.array([lens_dist/2.-lens_thick/2.-1.,-eps,0])
    p0 = np.array([lens_dist/2.-lens_thick/2.,eps,0])
    p1 = np.array([lens_dist/2.+lens_thick/2.,-eps,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_thick/2.,-eps,0])
    p7 = np.array([-lens_dist/2.+lens_thick/2.,eps,0])
    pc = np.array([-lens_dist/2.+lens_thick/2.+1.,eps,0])
    pd = np.array([-lens_dist/2.+lens_thick/2.+1.+quart_thick,eps,0])
    ps = np.stack([pa,pb,p0,p1,p2,p3,p4,p5,p6,p7,pc,pd], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
        
    hi = 12.7
    qi=7.75
    ng = 1.4537
    coef = np.zeros(5)
    coef[4] = c4
    elements = [FreeFormInterface(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng, coef=coef),\
                Glass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, n1=ng),\
                Glass(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                Mirror(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=hi),\
                Mirror(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=hi),\
                CurvedGlass(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=lens_diam, n1=ng),\
                Glass(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, n1=ng, coef=-coef)]
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

def LongLensCavMatt(betal=120.8269, rExtraIntra=5.2, angle=16., lens_diam=6.35, lens_thick=4., Rlens=25.0):
    """
    Long lens cavity with lens curved surfaces facing outwards!
    """
    angle = np.deg2rad(angle)
    eps = 0. #1e-5 #hack
    
    L = betal*(1 + rExtraIntra)/4
    l = L/np.sqrt(1. + 2*np.tan(angle/2.)**2)
    
    d1 = l*np.tan(angle/2.)
    
    p0 = np.array([-d1, 0., -l/2.])
    p1 = np.array([0., -d1, l/2.])
    p2 = np.array([d1, 0., -l/2.])
    p3 = np.array([0., d1, l/2.])
    
    aa = 0.5*(betal*rExtraIntra - 3*L)
    v12N = norm(p1 - p0)
    pL1 = p0 + aa*v12N
    pL2 = p0 + (aa+betal)*v12N
    
    pL1G = pL1 + lens_thick/2.*v12N
    pL1C = pL1 - lens_thick/2.*v12N
    
    pL2G = pL2 - lens_thick/2.*v12N
    pL2C = pL2 + lens_thick/2.*v12N
    
    ps = np.stack([pL2G, pL2C, p1, p2, p3, p0, pL1C, pL1G], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
    
    hi = 12.7
    qi=7.75
    ng = 1.4537

    #negative sign of firstt cuved surface for abcd matrix
    elements = [CurvedGlass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, n1=ng),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                Glass(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng)]
    
    return elements#, geom

def LongLensCav(betal=120.8269, rExtraIntra=5.2, angle=16., lens_diam=6.35, lens_thick=4., Rlens=25.0):
    angle = np.deg2rad(angle)
    
    L = betal*(1 + rExtraIntra)/4
    l = L/np.sqrt(1. + 2*np.tan(angle/2.)**2)
    
    d1 = l*np.tan(angle/2.)
    
    p0 = np.array([-d1, 0., -l/2.])
    p1 = np.array([0., -d1, l/2.])
    p2 = np.array([d1, 0., -l/2.])
    p3 = np.array([0., d1, l/2.])
    
    aa = 0.5*(betal*rExtraIntra - 3*L)
    v12N = norm(p1 - p0)
    pL1 = p0 + aa*v12N
    pL2 = p0 + (aa+betal)*v12N
    
    pL1G = pL1 + lens_thick/2.*v12N
    pL1C = pL1 - lens_thick/2.*v12N
    
    pL2G = pL2 - lens_thick/2.*v12N
    pL2C = pL2 + lens_thick/2.*v12N
    
    #ps = np.stack([pL1G, pL1C, p0, p1, p2, p3, pL2C, pL2G], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
    
    hi = 12.7
    qi=7.75
    ng = 1.4537
    #negative sign of firstt cuved surface for abcd matrix
    elements = [Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CX', n1=ng),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                CurvedGlass(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, R=Rlens, curv='CC', n2=ng),\
                Glass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng)]

    return elements#, geom




# To-do: combine these lens cavity geometries with/without quartic plates into one function more sensibly?
def Cav1L0Q(lens_dist=22.3, thet_fullOpen_deg=20, Rlens=5, lens_thick=4,  rExtraIntra=4.82, pathCorr=0.025, lens_diam=6.35, quart_thick=2.):
    """
    Duplicating cavity geometry from Mathematica framework. One lens per side of the focus.
    """
    # Not committed to this parameterization, just for ease of (at least initial) comparison
    
    # Tetrahedron
    theta = np.deg2rad(thet_fullOpen_deg)
    L = lens_dist*(1. + rExtraIntra)/4.
    l = L/np.sqrt(1. + 2*np.tan(theta/2.)**2)
    d1 = l*np.tan(theta/2.)
    d2 = d1
    
    # Mirror positions in the tetrahedron
    pM1 = np.array([-d1, 0, -l/2.])
    pM2 = np.array([0, -d2, l/2.])
    pM3 = np.array([d1, 0, -l/2.])
    pM4 = np.array([0, d2, l/2.])
    
    # Calculate lens locations
    aa = 0.5*( lens_dist*rExtraIntra - 3*L)
    v14N = norm(pM4-pM1)
    pL1 = pM1 + aa*v14N
    pL2 = pL1 + lens_dist*v14N
    pL1_flat = pL1 + lens_thick/2.*v14N
    pL1_curv = pL1 - lens_thick/2.*v14N
    pL2_flat = pL2 - lens_thick/2.*v14N
    pL2_curv = pL2 + lens_thick/2.*v14N
    
    # 'Manually' adjust optic positions so the thick lens bodies aren't hitting the mirrors (after defining lens positions!!)
    pM1 = pM1 - pathCorr*(pM4-pM1)
    pM4 = pM4 + pathCorr*(pM4-pM1)
    
    # Generate cavity geometry
    ps = np.stack([pL1_flat, pL1_curv, pM1, pM2, pM3, pM4, pL2_curv, pL2_flat], axis=0)   
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
 
    hi = 12.7
    qi=7.75
    ng = 1.4537
    elements = [\
                Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                CurvedGlass(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng)\
                    ]
    
    
    return elements


def Cav1L1Q(lens_dist=22.3, thet_fullOpen_deg=20, Rlens=5, lens_thick=4, rExtraIntra=4.82, pathCorr=0.025, lens_diam=6.35, c4=0., quart_thick=2.):
    """
    Duplicating from Mathematica framework. One lens per side of the focus, one quartic plate at upper waist.
    """
    
    # Tetrahedron  
    theta = np.deg2rad(thet_fullOpen_deg)
    L = lens_dist*(1. + rExtraIntra)/4.
    l = L/np.sqrt(1. + 2*np.tan(theta/2.)**2)
    d1 = l*np.tan(theta/2.)
    d2 = d1
    
    # Mirror positions in the tetrahedron
    pM1 = np.array([-d1, 0, -l/2.])
    pM2 = np.array([0, -d2, l/2.])
    pM3 = np.array([d1, 0, -l/2.])
    pM4 = np.array([0, d2, l/2.])
    
    # Calculate lens locations
    aa = 0.5*( lens_dist*rExtraIntra - 3*L)
    v14N = norm(pM4-pM1)
    pL1 = pM1 + aa*v14N
    pL2 = pL1 + lens_dist*v14N
    pL1_flat = pL1 + lens_thick/2.*v14N
    pL1_curv = pL1 - lens_thick/2.*v14N
    pL2_flat = pL2 - lens_thick/2.*v14N
    pL2_curv = pL2 + lens_thick/2.*v14N
    
    # 'Manually' adjust optic positions so the thick lens bodies aren't hitting the mirrors (after defining lens positions!!)
    pM1 = pM1 - pathCorr*(pM4-pM1)
    pM4 = pM4 + pathCorr*(pM4-pM1)
    
    # Quartic plate position
#     pQrtc_crv = (pM2 + pM3)/2. - quart_thick/2.*norm(pM3 - pM2)
#     pQrtc_flt = (pM2 + pM3)/2. + quart_thick/2.*norm(pM3 - pM2)
    pQrtc_crv = pM2 + 0.5*(pM3 - pM2)
    pQrtc_flt = pQrtc_crv + quart_thick * norm(pM3 - pM2)
    
    # Generate cavity geometry
    ps = np.stack([pL1_flat, pL1_curv, pM1, pM2, pQrtc_crv, pQrtc_flt, pM3, pM4, pL2_curv, pL2_flat], axis=0)   
#     ps = np.stack([pL1_flat, pL1_curv, pM1, pM2, pM3, pM4, pL2_curv, pL2_flat], axis=0)   
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
 
    hi = 12.7
    qi=7.75
    ng = 1.4537
    coef = np.zeros(5)
    coef[4] = c4
    elements = [\
                Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
#                 Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
#                 Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
#                 CurvedGlass(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
#                 Glass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng)\
                #FreeFormInterface(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=lens_diam, n2=ng, coef=coef),\
                Glass(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=lens_diam, n2=ng),\
                Glass(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=lens_diam, n1=ng),\
                Mirror(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=hi),\
                Mirror(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=hi),\
                CurvedGlass(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=lens_diam, n1=ng)\
                    ]
    
    return elements



def Cav1L2Q(
    lens_dist=22.3, thet_fullOpen_deg=20., Rlens=5., lens_thick=4., rExtraIntra=4.82, pathCorr=0.025, lens_diam=6.35, c4=0.000650, quart_thick=2., lensQrtcSep=1.
    #lens_dist=44.56+0*45.35, thet_fullOpen_deg=20., Rlens=10., lens_thick=3., rExtraIntra=6.50, pathCorr=0.00, lens_diam=6.35, c4=0.000075, quart_thick=2., lensQrtcSep=1e-3
    ):
    """
    Duplicating from Mathematica framework. One lens per side of the focus, two quartic plates (near the lenses)
    """

    # Tetrahedron
    theta = np.deg2rad(thet_fullOpen_deg)
    L = lens_dist*(1. + rExtraIntra)/4.
    l = L/np.sqrt(1. + 2*np.tan(theta/2.)**2)
    d1 = l*np.tan(theta/2.)
    d2 = d1
    
    # Mirror positions in the tetrahedron
    pM1 = np.array([-d1, 0, -l/2.])
    pM2 = np.array([0, -d2, l/2.])
    pM3 = np.array([d1, 0, -l/2.])
    pM4 = np.array([0, d2, l/2.])
    
    # Calculate lens locations
    aa = 0.5*( lens_dist*rExtraIntra - 3*L)
    v14N = norm(pM4-pM1)
    pL1 = pM1 + aa*v14N
    pL2 = pL1 + lens_dist*v14N
    pL1_flat = pL1 + lens_thick/2.*v14N
    pL1_curv = pL1 - lens_thick/2.*v14N
    pL2_flat = pL2 - lens_thick/2.*v14N
    pL2_curv = pL2 + lens_thick/2.*v14N
    
    # 'Manually' adjust optic positions so the thick lens bodies aren't hitting the mirrors (after defining lens positions!!)
    pM1 = pM1 - pathCorr*(pM4-pM1)
    pM4 = pM4 + pathCorr*(pM4-pM1)

    # Quartic plate positions
    pQrtc_crv1 = pL1_curv + lensQrtcSep * norm(pM1 - pL1_curv)
    pQrtc_flt1 = pQrtc_crv1 + quart_thick * norm(pM1 - pL1_curv)
    pQrtc_crv2 = pL2_curv + lensQrtcSep * norm(pM4 - pL2_curv)
    pQrtc_flt2 = pQrtc_crv2 + quart_thick * norm(pM4 - pL2_curv)
    
    # Generate cavity geometry
    ps = np.stack([pL1_flat, pL1_curv, pQrtc_crv1, pQrtc_flt1, pM1, pM2, pM3, pM4, pQrtc_flt2, pQrtc_crv2, pL2_curv, pL2_flat], axis=0)   
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
 
    hi = 12.7
    qi=7.75
    ng = 1.4537
    coef = np.zeros(5)
    coef[4] = c4
    elements = [\
                Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                FreeFormInterface(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, n2=ng, coef=coef),\
                Glass(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, n1=ng),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                Mirror(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=hi),\
                Mirror(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=hi),\
                Glass(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=lens_diam, n1=ng, coef=-coef),\
                CurvedGlass(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, n1=ng)\
                    ]
    
    return elements

def Cav1A2Q(lens_dist=22.3, thet_fullOpen_deg=20., Rlens=5., lens_thick=4.,  rExtraIntra=4.82, pathCorr=0.025, lens_diam=6.35, c4=0., quart_thick=2., lensQrtcSep=1.):
    """
    One aspheric lens per side of the focus (quadratic and quartic spherical terms only), two quartic plates (near the lenses)
    """
    
    # Tetrahedron    
    theta = np.deg2rad(thet_fullOpen_deg)
    L = lens_dist*(1. + rExtraIntra)/4.
    l = L/np.sqrt(1. + 2*np.tan(theta/2.)**2)
    d1 = l*np.tan(theta/2.)
    d2 = d1
    
    # Mirror positions in the tetrahedron
    pM1 = np.array([-d1, 0, -l/2.])
    pM2 = np.array([0, -d2, l/2.])
    pM3 = np.array([d1, 0, -l/2.])
    pM4 = np.array([0, d2, l/2.])
    
    # Calculate lens locations
    aa = 0.5*( lens_dist*rExtraIntra - 3*L)
    v14N = norm(pM4-pM1)
    pL1 = pM1 + aa*v14N
    pL2 = pL1 + lens_dist*v14N
    pL1_flat = pL1 + lens_thick/2.*v14N
    pL1_curv = pL1 - lens_thick/2.*v14N
    pL2_flat = pL2 - lens_thick/2.*v14N
    pL2_curv = pL2 + lens_thick/2.*v14N
    # Aspheric lens coefficients (to isolate quartic term)
    AsphCoefs = np.zeros(5)
    AsphCoefs[2] = 1/(2*Rlens)
    AsphCoefs[4] = 1/(8*Rlens**3)
    
    
    
    # 'Manually' adjust optic positions so the thick lens bodies aren't hitting the mirrors (after defining lens positions!!)
    pM1 = pM1 - pathCorr*(pM4-pM1)
    pM4 = pM4 + pathCorr*(pM4-pM1)

    # Quartic plate positions
    pQrtc_crv1 = pL1_curv + lensQrtcSep * norm(pM1 - pL1_curv)
    pQrtc_flt1 = pQrtc_crv1 + quart_thick * norm(pM1 - pL1_curv)
    pQrtc_crv2 = pL2_curv + lensQrtcSep * norm(pM4 - pL2_curv)
    pQrtc_flt2 = pQrtc_crv2 + quart_thick * norm(pM4 - pL2_curv)
    # Quartic plate coefficients
    QrtcCoefs = np.zeros(5)
    QrtcCoefs[4] = c4
    
    
    # Generate cavity geometry
    ps = np.stack([pL1_flat, pL1_curv, pQrtc_crv1, pQrtc_flt1, pM1, pM2, pM3, pM4, pQrtc_flt2, pQrtc_crv2, pL2_curv, pL2_flat], axis=0)   
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
 
    hi = 12.7
    qi=7.75
    ng = 1.4537

    elements = [\
                Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, n1=ng, coef=AsphCoefs),\
                FreeFormInterface(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, n2=ng, coef=QrtcCoefs),\
                Glass(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, n1=ng),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                Mirror(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=hi),\
                Mirror(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=hi),\
                Glass(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=lens_diam, n1=ng, coef=-QrtcCoefs),\
                FreeFormInterface(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, n2=ng, coef=-AsphCoefs),\
                Glass(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, n1=ng)\
                    ]    
    return elements


def Cav2L1Q(lens_dist=21.4, thet_fullOpen_deg=20., Rlens=10., lens_thick=3., lens_sep=0.15, rExtraIntra=3.64, pathCorr=0.2, c4=0.000580, quart_thick=2., lens_diam=6.35):
    """
    Duplicating from Mathematica framework. One lens per side of the focus, one quartic plate at upper waist.
    """
    
    # Tetrahedron
    theta = np.deg2rad(thet_fullOpen_deg)
    L = lens_dist*(1. + rExtraIntra)/4.
    l = L/np.sqrt(1. + 2*np.tan(theta/2.)**2)
    d1 = l*np.tan(theta/2.)
    d2 = d1
    
    # Mirror positions in the tetrahedron
    pM1 = np.array([-d1, 0, -l/2.])
    pM2 = np.array([0, -d2, l/2.])
    pM3 = np.array([d1, 0, -l/2.])
    pM4 = np.array([0, d2, l/2.])
    
    
    # Calculate lens locations
    aa = 0.5*( lens_dist*rExtraIntra - 3*L)
    v14N = norm(pM4-pM1)
    pL1a = pM1 + aa*v14N
    pL1b = pL1a - (lens_thick + lens_sep)*v14N
    pL1a_flat = pL1a + lens_thick/2.*v14N
    pL1a_curv = pL1a - lens_thick/2.*v14N
    pL1b_flat = pL1b + lens_thick/2.*v14N
    pL1b_curv = pL1b - lens_thick/2.*v14N
    
    pL2a = pL1a + lens_dist*v14N
    pL2b = pL2a + (lens_thick + lens_sep)*v14N
    pL2a_flat = pL2a - lens_thick/2.*v14N
    pL2a_curv = pL2a + lens_thick/2.*v14N
    pL2b_flat = pL2b - lens_thick/2.*v14N
    pL2b_curv = pL2b + lens_thick/2.*v14N
    

    # 'Manually' adjust optic positions so the thick lens bodies aren't hitting the mirrors (after defining lens positions!!)
    pM1 = pM1 - pathCorr*(pM4-pM1)
    pM4 = pM4 + pathCorr*(pM4-pM1)
    
    # Quartic plate position
    pQrtc_crv = (pM2 + pM3)/2. - quart_thick/2.*norm(pM3 - pM2)
    pQrtc_flt = (pM2 + pM3)/2. + quart_thick/2.*norm(pM3 - pM2)
    
    
    # Generate cavity geometry
    ps = np.stack([pL1a_flat, pL1a_curv, pL1b_flat, pL1b_curv, pM1, pM2, pQrtc_crv, pQrtc_flt, pM3, pM4, pL2b_curv, pL2b_flat, pL2a_curv, pL2a_flat], axis=0)   
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
 
    hi = 12.7
    qi=7.75
    ng = 1.4537
    coef = np.zeros(5)
    coef[4] = c4
    elements = [\
                Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Glass(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                FreeFormInterface(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, n2=ng, coef=coef),\
                Glass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng),\
                Mirror(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=hi),\
                Mirror(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=hi),\
                CurvedGlass(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, n1=ng),\
                CurvedGlass(p=ps[12], n=ns[12], ax=ax_x[12], ay=ax_y[12], Rbasis=Rtr[12], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[13], n=ns[13], ax=ax_x[13], ay=ax_y[13], Rbasis=Rtr[13], diameter=lens_diam, n1=ng)\
                    ]
    
    return elements

def Cav2L2Q(lens_dist=21.4, thet_fullOpen_deg=20., Rlens=10., lens_thick=3., lens_sep=0.15, rExtraIntra=3.64, pathCorr=0.2, c4=0.000140, quart_thick=2., lensQrtcSep=0.001, lens_diam=6.35):
    """
    Duplicating from Mathematica framework. One lens per side of the focus, two quartic plates (near the lenses)
    """

    # Tetrahedron
    theta = np.deg2rad(thet_fullOpen_deg)
    L = lens_dist*(1. + rExtraIntra)/4.
    l = L/np.sqrt(1. + 2*np.tan(theta/2.)**2)
    d1 = l*np.tan(theta/2.)
    d2 = d1
    
    # Mirror positions in the tetrahedron
    pM1 = np.array([-d1, 0, -l/2.])
    pM2 = np.array([0, -d2, l/2.])
    pM3 = np.array([d1, 0, -l/2.])
    pM4 = np.array([0, d2, l/2.])
    
    # Calculate lens locations
    aa = 0.5*( lens_dist*rExtraIntra - 3*L)
    v14N = norm(pM4-pM1)
    pL1a = pM1 + aa*v14N
    pL1b = pL1a - (lens_thick + lens_sep)*v14N
    pL1a_flat = pL1a + lens_thick/2.*v14N
    pL1a_curv = pL1a - lens_thick/2.*v14N
    pL1b_flat = pL1b + lens_thick/2.*v14N
    pL1b_curv = pL1b - lens_thick/2.*v14N
    
    pL2a = pL1a + lens_dist*v14N
    pL2b = pL2a + (lens_thick + lens_sep)*v14N
    pL2a_flat = pL2a - lens_thick/2.*v14N
    pL2a_curv = pL2a + lens_thick/2.*v14N
    pL2b_flat = pL2b - lens_thick/2.*v14N
    pL2b_curv = pL2b + lens_thick/2.*v14N
    
    # 'Manually' adjust optic positions so the thick lens bodies aren't hitting the mirrors (after defining lens positions!!)
    pM1 = pM1 - pathCorr*(pM4-pM1)
    pM4 = pM4 + pathCorr*(pM4-pM1)

    # Quartic plate positions
    pQrtc_crv1 = pL1b_curv + lensQrtcSep * norm(pM1 - pL1b_curv)
    pQrtc_flt1 = pQrtc_crv1 + quart_thick * norm(pM1 - pL1b_curv)
    pQrtc_crv2 = pL2b_curv + lensQrtcSep * norm(pM4 - pL2b_curv)
    pQrtc_flt2 = pQrtc_crv2 + quart_thick * norm(pM4 - pL2b_curv)
    
    # Generate cavity geometry
    ps = np.stack([pL1a_flat, pL1a_curv, pL1b_flat, pL1b_curv, pQrtc_crv1, pQrtc_flt1, pM1, pM2, pM3, pM4, pQrtc_flt2, pQrtc_crv2, pL2b_curv, pL2b_flat, pL2a_curv, pL2a_flat], axis=0)   
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
 
    hi = 12.7
    qi=7.75
    ng = 1.4537
    coef = np.zeros(5)
    coef[4] = c4
    elements = [\
                Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Glass(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                FreeFormInterface(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=lens_diam, n2=ng, coef=coef),\
                Glass(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=lens_diam, n1=ng),\
                Mirror(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=hi),\
                Mirror(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=hi),\
                Mirror(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=hi),\
                Mirror(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=hi),\
                Glass(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, n1=ng, coef=-coef),\
                CurvedGlass(p=ps[12], n=ns[12], ax=ax_x[12], ay=ax_y[12], Rbasis=Rtr[12], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[13], n=ns[13], ax=ax_x[13], ay=ax_y[13], Rbasis=Rtr[13], diameter=lens_diam, n1=ng),\
                CurvedGlass(p=ps[14], n=ns[14], ax=ax_x[14], ay=ax_y[14], Rbasis=Rtr[14], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[15], n=ns[15], ax=ax_x[15], ay=ax_y[15], Rbasis=Rtr[15], diameter=lens_diam, n1=ng)\
                    ]
    
    return elements