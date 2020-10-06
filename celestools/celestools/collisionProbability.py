#!/usr/bin/python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*-coding:Utf-8 -*


"""
  # collision.py
  ##############

  * Authors: Alexis Petit, Gabriel Magnaval, Share My Space
  * Date: 2020/06/12
  * Computation of the collision probability during a conjunction.
"""


#Packages


import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys
import os
import copy
import configparser
import scipy.integrate as integrate
from datetime import datetime,timedelta
from jdcal import gcal2jd,jd2gcal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


#Code


__all__ = [
    'CollisionProbability'
]


class CollisionProbability:
  
  def __init__(self,cdm):
    """ Initialization from a conjonction data message.
    """

    self.read_cdm(cdm)
    

  def read_cdm(self,cdm):
    """ Read conjunction data message.

    Parameters
    ----------

    cdm: dataframe
      Conjunction data message.

    """
       
    x_tca=[]

    x1 = cdm['t_x'].iloc[0]
    y1 = cdm['t_y'].iloc[0]
    z1 = cdm['t_z'].iloc[0]
    vx1 = cdm['t_vx'].iloc[0]
    vy1 = cdm['t_vy'].iloc[0]
    vz1 = cdm['t_vz'].iloc[0]
    state1 = np.array([x1,y1,z1,vx1,vy1,vz1])
    x_tca.append(state1)

    x2 = cdm['c_x'].iloc[0]
    y2 = cdm['c_y'].iloc[0]
    z2 = cdm['c_z'].iloc[0]
    vx2 = cdm['c_vx'].iloc[0]
    vy2 = cdm['c_vy'].iloc[0]
    vz2 = cdm['c_vz'].iloc[0]
    state2 = np.array([x2, y2, z2, vx2, vy2, vz2])
    x_tca.append(state2)

    self.dr = np.array([(x2-x1),(y2-y1),(z2-z1)])
    self.dv = np.array([(vx2-vx1),(vy2-vy1),(vz2-vz1)])
    #print(np.sqrt(self.dr[0]**2+self.dr[1]**2+self.dr[2]**2))

    # rtw contains the two transfer matrix from rtw to xyz
    self.rtw = [np.zeros((3,3)),np.zeros((3,3))]              
    for k in range (0,2):
      self.rtw[k][0] = self.rtw_frame(x_tca[k])[0]   #r
      self.rtw[k][1] = self.rtw_frame(x_tca[k])[1]   #t
      self.rtw[k][2] = self.rtw_frame(x_tca[k])[2]   #w
      self.rtw[k] = np.transpose(self.rtw[k])  

    if 'sig_r1' in cdm.columns:
      sig_r1 = cdm['sig_r1'].iloc[0]
      sig_t1 = cdm['sig_t1'].iloc[0]
      sig_w1 = cdm['sig_w1'].iloc[0]
      sig_r2 = cdm['sig_r2'].iloc[0]
      sig_t2 = cdm['sig_t2'].iloc[0]
      sig_w2 = cdm['sig_w2'].iloc[0]
    else:
      #print('Default covariance')
      sig_r1 = 1000.
      sig_t1 = 3000.
      sig_w1 = 2000.
      sig_r2 = 1000.
      sig_t2 = 3000.
      sig_w2 = 2000.

    # gam_rtw contains the two position covariance matrix in rtw  
    self.gam_rtw=[np.zeros((3,3)),np.zeros((3,3))]     
    self.gam_rtw[0][0,0] = sig_r1**2
    self.gam_rtw[0][1,1] = sig_t1**2
    self.gam_rtw[0][2,2] = sig_w1**2
    self.gam_rtw[1][0, 0] = sig_r2**2
    self.gam_rtw[1][1, 1] = sig_t2**2
    self.gam_rtw[1][2, 2] = sig_w2**2

    #print('GAM')
    #print(self.gam_rtw[0])
    #print(self.gam_rtw[1])          

    area1 = cdm['t_area'].iloc[0]
    radius1=np.sqrt(area1/np.pi)
    mass1 = cdm['t_mass'].iloc[0]
    area2 = cdm['c_area'].iloc[0]
    radius2 = np.sqrt(area2/np.pi)
    mass2 = cdm['c_mass'].iloc[0]
    self.rc = (radius1+radius2)
    #print('RC=',self.rc)    


  def probability(self):
    """ Compute the collision probability with the Foster method.
    """

    # gam_rtw contains the two position covariance matrix in xyz
    gamt = [np.zeros((3,3)),np.zeros((3,3))] 

    for k in range (0,2):
      gamt[k] = np.dot(self.gam_rtw[k],np.transpose(self.rtw[k]))
      gamt[k] = np.dot(self.rtw[k],gamt[k])

    #global covariance matrix  
    self.gamca3 = gamt[0]+gamt[1]                       
    #print('Global cov')
    #print(self.gamca3)
    
    #print(self.dr)
    #print(self.dv)
    
    # Projection onto the target plane (B-plane)
    gamca3tp = self.tp(self.dr,self.dv,self.gamca3)[0]
    self.dr_b = self.tp(self.dr,self.dv,self.gamca3)[1]  
    #print('DR b')
    #print(self.dr_b)  

    # Compute the collision probability
    self.p_col,self.kmax2,self.pmax = self.prob(self.dr_b,gamca3tp,self.rc)
    #print('PC',self.p_col)

    return self.p_col,self.pmax
    

  def tp(self,dr,dv,gamca3):

    """ projection of the state elements onto the target plane.

        Parameters
        ----------

        dr: the relative object position.
        dv: the relative object velocity .
        gamca3: the global covariance matrix.

        Returns
        -------

        gamca3tp: the global covariance matrix projects onti the target plane.
        dr_b: the relative position projects onto the traget plane.
        a,b: semi-major and semi-minorof the confidence ellipsoid.
        ea,eb: direction corresponding to semi-major and semi-minor axes.

        """

    # dr, dv are actually the transposal of dr/dv

    normdr = np.linalg.norm(dr)
    xb = dr / normdr
    w = np.cross(dr, dv)
    normw = np.linalg.norm(w)

    # (xb,yb) basis of the B_plane
    yb = w / normw

    # Transfer matrix from xyz to (xb,yb)
    xbyb = np.zeros((3, 2))
    xbyb[0:3, 0] = xb
    xbyb[0:3, 1] = yb
    gamca3tp = np.zeros((2, 2))
    gamca3tp = np.dot(gamca3, xbyb)

    # Projection of the common covariance matrix
    gamca3tp = np.dot(np.transpose(xbyb), gamca3tp)

    # Projection of dr
    dr_b = np.dot(dr, xbyb)

    """the following is only if you want to compute the semi-major and semi-minor axes of the confidence ellipsoid projected onto the target plane"""

    wr, evec = np.linalg.eig(
      gamca3tp)  # eigenvectors are uncorrelated as they form a basis in which the covariance is a diagonal matrix. Thus, eigenvalues are the semi-minor/semi-major axes
    if wr[0] < wr[1]:  # the bigger one is the semi-major axe
      a = np.sqrt(wr[1])
      b = np.sqrt(wr[0])
      anorm = np.linalg.norm(evec[1])
      ea = evec[1] / anorm
      bnorm = anorm = np.linalg.norm(evec[0])
      eb = evec[0] / bnorm
    else:
      a = np.sqrt(wr[0])
      b = np.sqrt(wr[1])
      anorm = np.linalg.norm(evec[0])
      ea = evec[0] / anorm
      bnorm = anorm = np.linalg.norm(evec[1])
      eb = evec[1] / bnorm

    return gamca3tp, dr_b, a, b, ea, eb






  def rtw_frame(self,X):
    """ Transformation NTW frame.

    Parameters
    ----------

    X: float array
      State vector of the satellite.

    Returns
    -------

    r: the radial direction.
    t: the transversal direction.
    w: the out-of-plane direction.

    """

    x=np.size(X)
    if x!=6 :
        return ("error not a state vector, cannot compute rtw_frame")
    r=np.array(X[0:3])
    v=np.array(X[3:6])
    normr=np.linalg.norm(r)
    r=r/normr
    w=np.cross(r,v)
    normw=np.linalg.norm(w)
    w=w/normw
    t=np.cross(w,r)     

    return r,t,w


  def prob(self,dr_b,gamca3tp,rc):
    """ Integration of the probability density function.

    Parameters
    ----------
    dr_b: projection of dr(=state2-state1) in the B-plane.
    gamca3tp: B-plane covariance matrix.
    rc: radius of the collision cross-section

    Returns
    -------

    p: probability of collision.

    """
    
    fint=0
    gamma=gamca3tp
    cca3tp=np.linalg.inv(gamma)
    lowlim=-rc
    uplimx=rc
    p= integrate.dblquad(self.fintxy,-rc,rc,lambda y:-np.sqrt(rc*rc-y*y),lambda y:np.sqrt(rc*rc-y*y),args=(dr_b,cca3tp,1),epsabs=1.49e-08,epsrel=1.49e-08)[0]
    
    kmax2,pmax=self.dilutionAnalytical(dr_b,cca3tp,rc)
    #print(kmax2, pmax)


    return p,kmax2,pmax



  def dilutionAnalytical(self,dr_b, cca3tp, rc):
    """ Analytical methods to compute the maximum of probability and the dilution

    Parameters
    ----------
    dr_b: projection of dr(=state2-state1) in the B-plane.
    cca3tp: inverse matrix of the B-plane covariance matrix (gamca3tp).
    rc: radius of the collision cross-section

    Returns
    -------
    kmax2= scaling factor squared where the maximum probability is reached
    pmax: maximum of probability.

    """
    K = np.dot(dr_b,cca3tp)
    kmax2 = K[0] * dr_b[0] + K[1] * dr_b[1]
    kmax2=kmax2/2
    det = np.linalg.det(cca3tp)
    d = np.sqrt(det)
    pmax = d * rc * rc / (2 * kmax2 * np.exp(1))
    return kmax2, pmax

  def dilutionNumerical(self,dr_b, cca3tp, rc):
    """ Numerical methods to compute the maximum of probability and the dilution

    Parameters
    ----------
    dr_b: projection of dr(=state2-state1) in the B-plane.
    cca3tp: inverse matrix of the B-plane covariance matrix (gamca3tp).
    rc: radius of the collision cross-section

    Returns
    -------
    kmax2= scaling factor squared where the maximum probability is reached
    pmax: maximum of probability.

    """
    kmax2=scipy.optimize.fmin(lambda k : - self.fintxy(0,0,dr_b,cca3tp,k2=k),0.1,xtol=1e-8,ftol=1e-8,maxiter=10000)[0]
    pmax=integrate.dblquad(self.fintxy,-rc,rc,lambda y:-np.sqrt(rc*rc-y*y),lambda y:np.sqrt(rc*rc-y*y),args=(dr_b,cca3tp,kmax2),epsabs=1.49e-08,epsrel=1.49e-08)[0]
    """self.plot_dilution(kmax2,dr_b,cca3tp,pmax,rc)"""
    return kmax2, pmax

  """def plot_dilution(self,kmax2,dr_b,cca3tp,Pcmax,rc):

    K = np.linspace(0, 10, 30)
    p = [self.probk(k,cca3tp,dr_b,rc) for k in K]
    plt.close()
    plt.plot(K, p)
    plt.plot([kmax2, kmax2], [0, Pcmax], 'r')
    plt.show()"""

  
  def fintxy(self,x,y,dr_b,cca3tp,k2):
    """ 
    two-dimensional (B-plane) probability density function.

    Parameters
    ----------

    x,y: two variables of the function.
    dr_b: projection of dr(=state2-state1) in the B-plane.
    cca3tp: inverse matrix of the B-plane coviance matrix (gamca3tp).
    k2: scaling factor squared

    Returns
    -------

    fintxy: probability density function.

    """

    fintxy=0
    dr=np.array([0.,0.])
    dr[0]=x-dr_b[0]
    dr[1]=y-dr_b[1]
    if k2==0:
      return 0
    cca=cca3tp/k2
    A=np.dot(dr,cca)
    fintxy=A[0]*dr[0]+A[1]*dr[1]
    fintxy=fintxy/2
    fintxy=np.exp(-fintxy)
    det = np.linalg.det(cca)
    d = np.sqrt(det)/(2*np.pi)
    fintxy=fintxy*d

    return fintxy


  def probk(self,k,cca3tp,dr_b,rc):
    probk = \
            integrate.dblquad(self.fintxy, -rc, rc, lambda y: -np.sqrt(rc * rc - y * y), lambda y: np.sqrt(rc * rc - y * y),
                   args=(dr_b, cca3tp,k), epsabs=1.49e-08, epsrel=1.49e-08)[0]
    return probk

  
      
  def plot_projection(self):
    """ Plot covariance ellipsoide onto the B-plane.
    """


    self.res_dir = 'event_{0}'.format(self.event_id)
    if os.path.isdir(self.res_dir):
      os.system('rm -r ' + self.res_dir)
    os.system('mkdir ' + self.res_dir)

    a, b, ea, eb = self.tp(self.dr, self.dv, self.gamca3)[2:7]
    angle = math.acos(ea[0] / np.sqrt(ea[0] ** 2 + ea[1] ** 2)) * 180. / np.pi

    if self.distance > 1000.:
      dist_max = 10E3
    else:
      dist_max = 1E3

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim([-dist_max, dist_max])
    ax1.set_ylim([-dist_max, dist_max])
    plt.plot(self.distance, 0, 'x', c='k', alpha=0.2, markersize=20)
    e1 = patches.Ellipse((0, 0), a, b,
                         angle=angle, linewidth=2, fill=False, zorder=2, color='k')
    e2 = patches.Ellipse((0, 0), a * 2, b * 2,
                         angle=angle, linewidth=2, fill=False, zorder=2, color='k')
    p1 = patches.Circle((self.distance, 0), self.rc)
    ax1.add_patch(e1)
    ax1.add_patch(e2)
    ax1.add_patch(p1)
    ax1.set_xlabel('x [m]', fontsize=14)
    ax1.set_ylabel('y [m]', fontsize=14)
    plt.savefig('{0}/projection_{1}'.format(self.res_dir, self.event_id))
    plt.close()




































    
    
