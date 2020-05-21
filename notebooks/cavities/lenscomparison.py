import numpy as np
from sloppy.optic import *
from sloppy.utils import *

coeffc2 = lambda x: 1./(2*x)
coeffc4 = lambda x: 1./(8*x**3)
coeffc6 = lambda x: 1./(16*x**5)
coeffc8 = lambda x: 5./(128*x**7)

def Cav2L2QInside(arm1=42., arm2=21., base=17., angle=110., lens_dist=25.1, Rlens=5.0, quart_thick=2., quart_dist=1e-3, c4=3e-4, c6=0., c8=0.):
    angle = np.deg2rad(angle)
    lens_diam=6.35
    lens_thick=4.
    
    pa = np.array([lens_dist/2.-lens_thick/2.-quart_dist-quart_thick, 0, 0])
    pb = np.array([lens_dist/2.-lens_thick/2.-quart_dist,0,0])
    p0 = np.array([lens_dist/2.-lens_thick/2.,0,0])
    p1 = np.array([lens_dist/2.+lens_thick/2.,0,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_thick/2.,0,0])
    p7 = np.array([-lens_dist/2.+lens_thick/2.,0,0])
    pc = np.array([-lens_dist/2.+lens_thick/2.+quart_dist,0,0])
    pd = np.array([-lens_dist/2.+lens_thick/2.+quart_dist+quart_thick,0,0])
    ps = np.stack([pa,pb,p0,p1,p2,p3,p4,p5,p6,p7,pc,pd], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
        
    hi = 12.7
    qi= 7.75
    ng = 1.4537
    coef = np.zeros(9)
    coef[4] = c4
    coef[6] = c6
    coef[8] = c8
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

def Cav2L2QOutside(arm1=42., arm2=21., base=17., angle=110., lens_dist=24.55, Rlens=5.0, quart_thick=2., quart_dist=1e-6, c4=0.4e-2, c6=0., c8=0.):
    angle = np.deg2rad(angle)
    lens_diam=6.35
    lens_thick=4.
    
    p0 = np.array([lens_dist/2.-lens_thick/2.,0,0])
    p1 = np.array([lens_dist/2.+lens_thick/2.,0,0])
    pa = np.array([lens_dist/2.+lens_thick/2.+quart_dist, 0, 0])
    pb = np.array([lens_dist/2.+lens_thick/2.+quart_dist+quart_thick,0,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    pc = np.array([-lens_dist/2.-lens_thick/2.-quart_dist-quart_thick,0,0])
    pd = np.array([-lens_dist/2.-lens_thick/2.-quart_dist,0,0])
    p6 = np.array([-lens_dist/2.-lens_thick/2.,0,0])
    p7 = np.array([-lens_dist/2.+lens_thick/2.,0,0])

    ps = np.stack([p0,p1,pa,pb,p2,p3,p4,p5,pc,pd,p6,p7], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
        
    hi = 12.7
    qi= 7.75
    ng = 1.4537
    coef = np.zeros(9)
    coef[4] = c4
    coef[6] = c6
    coef[8] = c8
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

def Cav4L1Q(arm1=42., arm2=21., base=17., angle=110., lens_dist=25.1, Rlens=5.0, lens_spacing=1e-3, quart_thick=2., c4=0.4e-2, c6=0., c8=0.):
    angle = np.deg2rad(angle)
    lens_diam=6.35
    lens_thick=4.
    
    pa = np.array([lens_dist/2.-lens_spacing/2.-lens_thick, 0, 0])
    pb = np.array([lens_dist/2.-lens_spacing/2.,0,0])
    p0 = np.array([lens_dist/2.+lens_spacing/2.,0,0])
    p1 = np.array([lens_dist/2.+lens_spacing/2+lens_thick,0,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_spacing/2-lens_thick,0,0])
    p7 = np.array([-lens_dist/2.-lens_spacing/2,0,0])
    pc = np.array([-lens_dist/2.+lens_spacing/2,0,0])
    pd = np.array([-lens_dist/2.+lens_spacing/2+lens_thick,0,0])
    v34 = norm(p4-p3)
    pq = 0.5*(p3+p4) #Quartic plate in upper waist
    pq1 = pq - v34*quart_thick
    pq2 = pq + v34*quart_thick
    ps = np.stack([pa,pb,p0,p1,p2,p3,pq1,pq2,p4,p5,p6,p7,pc,pd], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
        
    hi = 12.7
    qi= 7.75
    ng = 1.4537
    coef = np.zeros(9)
    coef[4] = c4
    coef[6] = c6
    coef[8] = c8
    elements = [\
                Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Glass(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                FreeFormInterface(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, n2=ng, coef=0.5*coef),\
                FreeFormInterface(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng, coef=-0.5*coef),\
                Mirror(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=hi),\
                Mirror(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=hi),\
                CurvedGlass(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, n1=ng),\
                CurvedGlass(p=ps[12], n=ns[12], ax=ax_x[12], ay=ax_y[12], Rbasis=Rtr[12], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[13], n=ns[13], ax=ax_x[13], ay=ax_y[13], Rbasis=Rtr[13], diameter=lens_diam, n1=ng)\
               ]
    return elements

def Cav4L1Qflip(arm1=42., arm2=21., base=17., angle=110., lens_dist=25.1, Rlens=5.0, lens_spacing=1e-3, quart_thick=2., c4=0.4e-2, c6=0., c8=0.):
    angle = np.deg2rad(angle)
    lens_diam=6.35
    lens_thick=4.
    
    pa = np.array([lens_dist/2.-lens_spacing/2.-lens_thick, 0, 0])
    pb = np.array([lens_dist/2.-lens_spacing/2.,0,0])
    p0 = np.array([lens_dist/2.+lens_spacing/2.,0,0])
    p1 = np.array([lens_dist/2.+lens_spacing/2+lens_thick,0,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_spacing/2-lens_thick,0,0])
    p7 = np.array([-lens_dist/2.-lens_spacing/2,0,0])
    pc = np.array([-lens_dist/2.+lens_spacing/2,0,0])
    pd = np.array([-lens_dist/2.+lens_spacing/2+lens_thick,0,0])
    v34 = norm(p4-p3)
    pq = 0.5*(p3+p4) #Quartic plate in upper waist
    pq1 = pq - v34*quart_thick
    pq2 = pq + v34*quart_thick
    ps = np.stack([pa,pb,p0,p1,p2,p3,pq1,pq2,p4,p5,p6,p7,pc,pd], axis=0)
    
    geom = geometry(ps)
    ns = geom['refl']
    ps = geom['mir']
    angles = geom['angles']
    Rtr = geom['R']
    ax_x = geom['xin']
    ax_y = np.cross(ns, ax_x)
        
    hi = 12.7
    qi= 7.75
    ng = 1.4537
    coef = np.zeros(9)
    coef[4] = c4
    coef[6] = c6
    coef[8] = c8
    elements = [\
                Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                CurvedGlass(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, n1=ng),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                FreeFormInterface(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, n2=ng, coef=0.5*coef),\
                FreeFormInterface(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng, coef=-0.5*coef),\
                Mirror(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=hi),\
                Mirror(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=hi),\
                Glass(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, n2=ng),\
                CurvedGlass(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, R=-Rlens, curv='CC', n1=ng),\
                CurvedGlass(p=ps[12], n=ns[12], ax=ax_x[12], ay=ax_y[12], Rbasis=Rtr[12], diameter=lens_diam, R=Rlens, curv='CX', n2=ng),\
                Glass(p=ps[13], n=ns[13], ax=ax_x[13], ay=ax_y[13], Rbasis=Rtr[13], diameter=lens_diam, n1=ng)\
               ]
    return elements

def Cav2A(arm1=42., arm2=21., base=17., angle=110., lens_dist=24.47, Rlens=5.0, c4=coeffc4(5.0), c6=coeffc6(5.0), c8=coeffc8(5.0)):
    angle = np.deg2rad(angle)
    lens_diam=6.35
    lens_thick=4.
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
    ax_y = np.cross(ns, ax_x)
        
    hi = 12.7
    qi=7.75
    ng = 1.4537
    
    coef = np.zeros(9)
    coef[2] = coeffc2(Rlens)
    coef[4] = c4
    coef[6] = c6
    coef[8] = c8
     #negative sign of firstt cuved surface for abcd matrix
    elements = [Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, n1=ng, coef=coef),\
                Mirror(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=hi),\
                Mirror(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=hi),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                FreeFormInterface(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=lens_diam, n2=ng, coef=-coef),\
                Glass(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=lens_diam, n1=ng)]
    return elements


def Cav4A(arm1=42., arm2=21., base=17., angle=110., lens_dist=24.47, Rlens=5.0, lens_spacing=1e-3, c4=coeffc4(5.0), c6=coeffc6(5.0), c8=coeffc8(5.0)):
    angle = np.deg2rad(angle)
    lens_diam=6.35
    lens_thick=4.
    pa = np.array([lens_dist/2.-lens_spacing/2.-lens_thick, 0, 0])
    pb = np.array([lens_dist/2.-lens_spacing/2.,0,0])
    p0 = np.array([lens_dist/2.+lens_spacing/2.,0,0])
    p1 = np.array([lens_dist/2.+lens_spacing/2+lens_thick,0,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_spacing/2-lens_thick,0,0])
    p7 = np.array([-lens_dist/2.-lens_spacing/2,0,0])
    pc = np.array([-lens_dist/2.+lens_spacing/2,0,0])
    pd = np.array([-lens_dist/2.+lens_spacing/2+lens_thick,0,0])

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
    
    coef = np.zeros(9)
    coef[2] = coeffc2(Rlens)
    coef[4] = c4
    coef[6] = c6
    coef[8] = c8
     #negative sign of firstt cuved surface for abcd matrix
    elements = [Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, n1=ng, coef=coef),\
                Glass(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, n1=ng, coef=coef),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                Mirror(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=hi),\
                Mirror(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=hi),\
                FreeFormInterface(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=lens_diam, n2=ng, coef=-coef),\
                Glass(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=lens_diam, n1=ng),\
                FreeFormInterface(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, n2=ng, coef=-coef),\
                Glass(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, n1=ng)]
    return elements

def Cav4Aflip(arm1=42., arm2=21., base=17., angle=110., lens_dist=24.47, Rlens=5.0, lens_spacing=1e-3, c4=coeffc4(5.0), c6=coeffc6(5.0), c8=coeffc8(5.0)):
    angle = np.deg2rad(angle)
    lens_diam=6.35
    lens_thick=4.
    pa = np.array([lens_dist/2.-lens_spacing/2.-lens_thick, 0, 0])
    pb = np.array([lens_dist/2.-lens_spacing/2.,0,0])
    p0 = np.array([lens_dist/2.+lens_spacing/2.,0,0])
    p1 = np.array([lens_dist/2.+lens_spacing/2+lens_thick,0,0])
    p2 = np.array([arm1/2.,0,0])
    p3 = np.array([np.cos(angle)*arm2/2.,base,np.sin(angle)*arm2/2.])
    p4 = np.array([-np.cos(angle)*arm2/2.,base,-np.sin(angle)*arm2/2.])
    p5 = np.array([-arm1/2.,0,0])
    p6 = np.array([-lens_dist/2.-lens_spacing/2-lens_thick,0,0])
    p7 = np.array([-lens_dist/2.-lens_spacing/2,0,0])
    pc = np.array([-lens_dist/2.+lens_spacing/2,0,0])
    pd = np.array([-lens_dist/2.+lens_spacing/2+lens_thick,0,0])

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
    
    coef = np.zeros(9)
    coef[2] = coeffc2(Rlens)
    coef[4] = c4
    coef[6] = c6
    coef[8] = c8
     #negative sign of firstt cuved surface for abcd matrix
    elements = [Glass(p=ps[0], n=ns[0], ax=ax_x[0], ay=ax_y[0], Rbasis=Rtr[0], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[1], n=ns[1], ax=ax_x[1], ay=ax_y[1], Rbasis=Rtr[1], diameter=lens_diam, n1=ng, coef=coef),\
                FreeFormInterface(p=ps[2], n=ns[2], ax=ax_x[2], ay=ax_y[2], Rbasis=Rtr[2], diameter=lens_diam, n2=ng, coef=-coef),\
                Glass(p=ps[3], n=ns[3], ax=ax_x[3], ay=ax_y[3], Rbasis=Rtr[3], diameter=lens_diam, n1=ng),\
                Mirror(p=ps[4], n=ns[4], ax=ax_x[4], ay=ax_y[4], Rbasis=Rtr[4], diameter=hi),\
                Mirror(p=ps[5], n=ns[5], ax=ax_x[5], ay=ax_y[5], Rbasis=Rtr[5], diameter=hi),\
                Mirror(p=ps[6], n=ns[6], ax=ax_x[6], ay=ax_y[6], Rbasis=Rtr[6], diameter=hi),\
                Mirror(p=ps[7], n=ns[7], ax=ax_x[7], ay=ax_y[7], Rbasis=Rtr[7], diameter=hi),\
                Glass(p=ps[8], n=ns[8], ax=ax_x[8], ay=ax_y[8], Rbasis=Rtr[8], diameter=lens_diam, n2=ng),\
                FreeFormInterface(p=ps[9], n=ns[9], ax=ax_x[9], ay=ax_y[9], Rbasis=Rtr[9], diameter=lens_diam, n1=ng, coef=coef),\
                FreeFormInterface(p=ps[10], n=ns[10], ax=ax_x[10], ay=ax_y[10], Rbasis=Rtr[10], diameter=lens_diam, n2=ng, coef=-coef),\
                Glass(p=ps[11], n=ns[11], ax=ax_x[11], ay=ax_y[11], Rbasis=Rtr[11], diameter=lens_diam, n1=ng)]
    return elements