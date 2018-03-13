#coding: utf-8
import numpy as np
import math

def gauss(x,y,sigma):
    twoPi = math.pi * 2
    return (1/(twoPi*sigma*sigma))*math.exp(-(x*x+y*y)/float(2*sigma*sigma))

def getGaussian(sigma,size):
    n = size[0]
    m = size[1]
    f=np.array([[gauss(i,j,sigma) for j in range (-(m-1)//2, (m+1)//2)] for i in range(-(n-1)//2, (n+1)//2)])
    f = f / np.sum(f)
    return f

def find(x,y):
    result=[]
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if((x[i]==y[j]).prod()==1):
                result.append(x[i])
    return np.array(result)

def motion_blur(len, ang):
    eps=2.220446049250313e-016
    len=max(1, len)
    half=(len-1)/2
    phi=(ang%180)/180.*math.pi
    cosphi=math.cos(phi)
    sinphi=math.sin(phi)
    xsign=np.sign(cosphi)
    linewdth=1
    sx=int(half*cosphi+linewdth*xsign-len*eps)
    sy=int(half*sinphi+linewdth-len*eps)
    xx=np.arange(0,sx+xsign, xsign)
    yy=np.arange(0, sy+1)
    x,y=np.meshgrid(xx,yy)
    dist2line=(y*cosphi-x*sinphi)
    rad=np.sqrt(x*x+y*y)
    indexs1=np.where(rad>=half)
    indexs2=np.where(np.fabs(dist2line)<=linewdth)
    indexs1=np.array(indexs1)
    indexs1=indexs1.transpose()
    indexs2=np.array(indexs2)
    indexs2=indexs2.transpose()
    indexs=find(indexs1,indexs2)
    x2lastpix=half-np.fabs((x[indexs[:,0], indexs[:,1]]+dist2line[indexs[:,0], indexs[:,1]]*sinphi)/cosphi)
    dist2line[indexs[:,0], indexs[:,1]]=np.sqrt(dist2line[indexs[:,0], indexs[:,1]]**2+x2lastpix**2)
    dist2line=linewdth+eps-abs(dist2line)
    dist2line[np.where(dist2line < 0)] = 0
    h=np.rot90(dist2line,2)
    old_h=np.copy(h)
    h_w=h.shape[0]
    h_h=h.shape[1]
    d_w=dist2line.shape[0]
    d_h=dist2line.shape[1]
    h=np.zeros((h_w+d_w-1, h_h+d_h-1), dtype=np.float64)
    h[0:h_w, 0:h_h]=old_h
    h[h_w-1: h_w+d_w-1, h_h-1: h_h+d_h-1]=dist2line
    h=h/(np.sum(h)+eps*len*len)
    #h(end + (1:end)-1, end + (1:end)-1) = dist2line;
    if cosphi>0:
        h=np.flipud(h)
    return h