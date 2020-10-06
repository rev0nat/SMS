#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*-coding:Utf-8 -*

"""
  # coordinates.py
  
  * Authors: Alexis Petit, PhD Student, Namur University
  * Date: 2017/03/09
  * It allows to perform coordinate transformations.
"""


#Packages


import math
import numpy as np
from numpy import cos,sin,sqrt,dot,cross,pi
from numpy.linalg import norm
from math import acos,asin,atan,atan2,floor,fmod,isinf,isnan
from scipy.interpolate import interp1d
import sys
import pandas as pd
import copy
import os
import sys


#Internal packages


from celestools.constants import earth_eq_radius
from celestools.constants import earth_J2
from celestools.constants import earth_mu
from celestools.constants import earth_omega
from celestools.constants import e_earth
from celestools.time_scale import date_2_jd
from celestools.time_scale import jd_2_jd_cnes
from celestools.time_scale import greenwich_mean_sideral_time


#Code


MAX_ITERATIONS = 100


__all__ = [
    'cart_2_kepl',
    'kepl_2_cart',
    'df_kepl_2_cart',
    'df_cart_2_kepl',  
    'cart_2_sph',
    'sph_2_cart', 
    'kepl_2_equinoctial',
    'mean_anomaly_2_ecc_anomaly',
    'true_anomaly_2_ecc_anomaly',
    'ecc_anomaly_2_mean_anomaly',
    'true_anomaly_2_mean_anomaly',
    'ecc_anomaly_2_true_anomaly',
    'mean_anomaly_2_true_anomaly',
    'flight_path_angle',
    'dms_2_rad',
    'hms_2_rad',
    'degree_2_hms',
    'degree_2_dms',
    'geocentric_2_topocentric',
    'topocentric_2_geocentric',
    'latlon_2_ecef',
    'ecef_2_latlon',
    'rotation',
    'convert_ephemeris_2_snapshot',
]


def cart_2_kepl_old(cart,mu,unit='rad'):
  """ Return the orbital elements from cartesian coordinates. It is a old version before vectorization.

  Parameters
  ----------

  cart: float array
    Cartesian coordinates (r[m],v[m/s]).
  mu: float
    Gravitational constant time the mass of the central body.
  unit: string
    Unit of the angles ('rad' or 'deg').

  Returns
  -------

  kepl: float array
    Orbital elements (a,e,i,raan,omega,ma).

  """

  if type(cart) is list:
    cart = np.array(cart)
  if len(cart.shape)==1:
    cart = np.expand_dims(cart,axis=0)

  x = cart[:,0]
  y = cart[:,1]
  z = cart[:,2]
  xp = cart[:,3]
  yp = cart[:,4]
  zp = cart[:,5]
   
  r   = np.sqrt(x**2+y**2+z**2)
  usa = 2./r-(xp**2+yp**2+zp**2)/mu
  sma  = 1./usa
  pn  = usa*np.sqrt(mu*usa)

  c1   = x*yp-y*xp
  c2   = y*zp-z*yp
  c3   = x*zp-z*xp
  w4   = c2**2+c3**2
  ccar = w4+c1**2
  rcar = np.sqrt(ccar)

  uh = 1./rcar
  if (w4>ccar*1.E-12):
    chi  = c1*uh
    sinhi = np.sqrt(1.-chi**2)
    inc    = acos(chi)
    raan  = math.atan2(c2,c3)
    raan  = (raan+2.*np.pi)%(2.*np.pi)      
  else:
    inc   = 0.
    sinhi = 0.
    if (c1<0.):
      inc = np.pi
    raan = 0.

  pp = ccar/mu
  tgbeta = (x*xp+y*yp+z*zp)*uh
  pusa = pp*usa
  ecc = np.sqrt(1.-pusa)

  #On limite la valeur de e a 10e-6

  iie = int(ecc*1000000.)
  if (iie==0):
    ecc = 0.
      
  esinu = tgbeta*np.sqrt(pusa)
  ecosu = 1.-r*usa
  u = math.atan2(esinu,ecosu)
  anom = u-esinu
  anom = (anom+2.*np.pi)%(2.*np.pi)

  w1  = pp-r
  cin = pp*tgbeta
  v   = math.atan2(cin,w1)

  w3      = x*np.cos(raan)+y*np.sin(raan)
  qos     = w3*sinhi
  vpomega = math.atan2(z,qos)
    
  if (iie==0):
    omega  = 0.
    anom = vpomega
  else:
    omega = vpomega-v
    omega = (omega+2.*pi)%(2.*np.pi)
  
  kepl = np.zeros((cart.shape[0], 6))
  kepl[:, 0] = sma
  kepl[:, 1] = ecc
  if unit=='rad':
    kepl[:, 2] = inc
    kepl[:, 3] = (raan+2.*np.pi)%(2.*np.pi)
    kepl[:, 4] = (omega+2.*np.pi)%(2.*np.pi)
    kepl[:, 5] = (anom+2.*np.pi)%(2.*np.pi)
  elif unit=='deg':
    kepl[:, 2] = inc*180/np.pi
    kepl[:, 3] = ((raan+2.*np.pi)%(2.*np.pi))*180/np.pi
    kepl[:, 4] = ((omega+2.*np.pi)%(2.*np.pi))*180/np.pi
    kepl[:, 5] = ((anom+2.*np.pi)%(2.*np.pi))*180/np.pi
  else:
    print('Bad angle unit')
    sys.exit()

  if cart.shape[0] == 1:
    kepl = np.squeeze(kepl)
      
  return kepl
   

def cart_2_kepl(cart,mu,unit='rad'):
  """ Return keplerian elements from cartesian coordinates.

  Parameters
  ----------

  cart: float array
    Cartesian coordinates (r[m],v[m/s]).
  mu: float
    Gravitational constant time the mass of the central body.
  unit: string
    Unit of the angles ('rad' or 'deg').

  Returns
  -------

  kepl: float array
    Orbital elements (a,e,i,raan,omega,ma).

  """

  if type(cart) is list:
    cart = np.array(cart)
  if len(cart.shape) == 1:
    cart = np.expand_dims(cart, axis=0)

  x = cart[:, 0]
  y = cart[:, 1]
  z = cart[:, 2]
  xp = cart[:, 3]
  yp = cart[:, 4]
  zp = cart[:, 5]

  r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
  usa = 2. / r - (xp ** 2 + yp ** 2 + zp ** 2) / mu
  sma = 1. / usa
  pn = usa * np.sqrt(mu * usa)

  c1 = x * yp - y * xp
  c2 = y * zp - z * yp
  c3 = x * zp - z * xp
  w4 = c2 ** 2 + c3 ** 2
  ccar = w4 + c1 ** 2
  rcar = np.sqrt(ccar)

  uh = 1. / rcar

  def func(a, b, c, u, cc, ccc):
    if (a > b * 1.E-18):
      chi = c * u
      sinhi = np.sqrt(1. - chi ** 2)
      inc = acos(chi)
      raan = math.atan2(cc, ccc)
      raan = (raan + 2. * np.pi) % (2. * np.pi)
    else:
      chi = 0.
      inc = 0.
      sinhi = 0.
      if (c1 < 0.):
        inc = np.pi
      raan = 0.
    return chi, sinhi, inc, raan

  vec_func = np.vectorize(func)

  chi, sinhi, inc, raan = vec_func(w4, ccar, c1, uh, c2, c3)

  pp = ccar / mu
  tgbeta = (x * xp + y * yp + z * zp) * uh
  pusa = pp * usa
  ecc = np.sqrt(1. - pusa)

  # On limite la valeur de e a 10e-6

  iie = (ecc * 1000000.).astype(int)

  def func(a, e):
    if a == 0:
      e = 0
    return e
  vec_func = np.vectorize(func)
  ecc = vec_func(iie, ecc)
  esinu = tgbeta * np.sqrt(pusa)
  ecosu = 1. - r * usa
  u = np.arctan2(esinu, ecosu)
  anom = u - esinu
  anom = (anom + 2. * np.pi) % (2. * np.pi)

  w1 = pp - r
  cin = pp * tgbeta
  v = np.arctan2(cin, w1)

  w3 = x * np.cos(raan) + y * np.sin(raan)
  qos = w3 * sinhi
  vpomega = np.arctan2(z, qos)

  def func(i, a, vp, v):
    if i == 0:
      omega = 0
      a = vp
      return a, omega
    else:
      omega = ((vp - v)+ 2. * pi) % (2. * np.pi)
      return a, omega

  vec_func = np.vectorize(func)
  anom, omega = vec_func(iie, anom, vpomega, v)

  kepl = np.zeros((cart.shape[0], 6))
  kepl[:, 0] = sma
  kepl[:, 1] = ecc
  if unit == 'rad':
    kepl[:, 2] = inc
    kepl[:, 3] = (raan + 2. * np.pi) % (2. * np.pi)
    kepl[:, 4] = (omega + 2. * np.pi) % (2. * np.pi)
    kepl[:, 5] = (anom + 2. * np.pi) % (2. * np.pi)
  elif unit == 'deg':
    kepl[:, 2] = inc * 180 / np.pi
    kepl[:, 3] = ((raan + 2. * np.pi) % (2. * np.pi)) * 180 / np.pi
    kepl[:, 4] = ((omega + 2. * np.pi) % (2. * np.pi)) * 180 / np.pi
    kepl[:, 5] = ((anom + 2. * np.pi) % (2. * np.pi)) * 180 / np.pi
  else:
    print('Bad angle unit')
    sys.exit()

  if cart.shape[0] == 1:
    kepl = np.squeeze(kepl)

  return kepl


def kepl_2_cart_old(kepl,mu,unit='rad'):
  """ Return cartesian coordinates from orbital elements.

  Parameters
  ----------

  kepl: float array
    Keplerian elements (a,e,i,raan,omega,ma).
  mu: float
    Gravitational constant time the mass of the central body.
  unit: string
    Unit of the angles.

  Returns
  -------
 
  cart: float array
    Cartesian coordinates (x[m],v[m/s])  

  """

  if type(kepl) is list:
    kepl = np.array(kepl)
  if len(kepl.shape) == 1:
    kepl = np.expand_dims(kepl, axis=0)

  sma  = kepl[:,0]
  ecc  = kepl[:,1]
  if unit=='rad':
    inc   = kepl[:,2]
    raan  = kepl[:,3]
    omega = kepl[:,4]
    anom  = (2*np.pi+kepl[:, 5])%(2*np.pi)
  elif unit=='deg':
    inc   = kepl[:,2]*np.pi/180.
    raan  = kepl[:,3]*np.pi/180.
    omega = kepl[:,4]*np.pi/180.
    anom  = (2*np.pi+kepl[:, 5]*np.pi/180.)%(2*np.pi)
  else:
    print('Bad angle unit')
    sys.exit()

  if np.any(ecc>=1) or np.any(ecc<0):
    print('Error: conversion keplerian coordinates to cartesians coordinates')
    print('Error: not elliptical case : e = ',ecc)
    sys.exit()
  else:
    ecc2 = ecc*ecc
    snraan = np.sin(raan)
    csraan = np.cos(raan)
    snomega = np.sin(omega)
    csomega = np.cos(omega)
    cshi = np.cos(inc)
    snhi = np.sqrt(1.0-cshi*cshi)
    prod = cshi*snomega
    
    px = sma*(-snraan*prod+csraan*csomega)
    py = sma*( csraan*prod+snraan*csomega)
    pz = sma*snhi*snomega
    
    b = sma*np.sqrt(1.0-ecc2)
    prod = cshi*csomega
    
    qx = b*(-snraan*prod-csraan*snomega)
    qy = b*( csraan*prod-snraan*snomega)
    qz = b*  snhi*csomega
    
    u = mean_anomaly_2_ecc_anomaly(ecc,anom)
   
    cosu   = np.cos(u)
    ucosme = cosu-ecc
    sinu   = np.sin(u)
    
    x = ucosme*px+sinu*qx
    y = ucosme*py+sinu*qy
    z = ucosme*pz+sinu*qz
    
    pn = np.sqrt(mu/sma)/sma
    c1 = pn/(1.0-ecc*cosu)
    
    xp = c1*(-px*sinu+qx*cosu)
    yp = c1*(-py*sinu+qy*cosu)
    zp = c1*(-pz*sinu+qz*cosu)

    cart = np.zeros((kepl.shape[0],6))
    cart[:,0] = x
    cart[:,1] = y
    cart[:,2] = z
    cart[:,3] = xp
    cart[:,4] = yp
    cart[:,5] = zp

    if kepl.shape[0] == 1:
      cart = np.squeeze(cart)
      
  return cart


def kepl_2_cart(kepl,mu,unit='rad'):
  """ Return cartesian coordinates from orbital elements.

  Parameters
  ----------

  kepl: float array
    Orbital elements (a,e,i,raan,omega,ma).
  mu: float
    Gravitational constant time the mass of the central body.
  unit: string
    Unit of the angles ('rad' or 'deg').

  Returns
  -------

  cart: float array
    Cartesian coordinates (x[m],v[m/s])

  """

  if type(kepl) is list:
    kepl = np.array(kepl)
  if len(kepl.shape)==1:
    kepl = np.expand_dims(kepl,axis=0)

  sma = kepl[:, 0]
  ecc = kepl[:, 1]
  if unit == 'rad':
    inc = kepl[:, 2]
    raan = kepl[:, 3]
    omega = kepl[:, 4]
    anom = (2 * np.pi + kepl[:, 5]) % (2 * np.pi)
  elif unit == 'deg':
    inc = kepl[:, 2] * np.pi / 180.
    raan = kepl[:, 3] * np.pi / 180.
    omega = kepl[:, 4] * np.pi / 180.
    anom = (2 * np.pi + kepl[:, 5] * np.pi / 180.) % (2 * np.pi)
  else:
    print('Bad angle unit')
    sys.exit()

  if np.any(ecc >= 1) or np.any(ecc < 0):
    print('Error: conversion keplerian coordinates to cartesians coordinates')
    print('Error: not elliptical case : e = ', ecc)
    sys.exit()
  else:
    ecc2 = ecc * ecc
    snraan = np.sin(raan)
    csraan = np.cos(raan)
    snomega = np.sin(omega)
    csomega = np.cos(omega)
    cshi = np.cos(inc)
    snhi = np.sqrt(1.0 - cshi * cshi)
    prod = cshi * snomega

    px = sma * (-snraan * prod + csraan * csomega)
    py = sma * (csraan * prod + snraan * csomega)
    pz = sma * snhi * snomega

    b = sma * np.sqrt(1.0 - ecc2)
    prod = cshi * csomega

    qx = b * (-snraan * prod - csraan * snomega)
    qy = b * (csraan * prod - snraan * snomega)
    qz = b * snhi * csomega

    u = mean_anomaly_2_ecc_anomaly(ecc, anom)

    cosu = np.cos(u)
    ucosme = cosu - ecc
    sinu = np.sin(u)

    x = ucosme * px + sinu * qx
    y = ucosme * py + sinu * qy
    z = ucosme * pz + sinu * qz

    pn = np.sqrt(mu / sma) / sma
    c1 = pn / (1.0 - ecc * cosu)

    xp = c1 * (-px * sinu + qx * cosu)
    yp = c1 * (-py * sinu + qy * cosu)
    zp = c1 * (-pz * sinu + qz * cosu)

    cart = np.zeros((kepl.shape[0], 6))
    cart[:, 0] = x
    cart[:, 1] = y
    cart[:, 2] = z
    cart[:, 3] = xp
    cart[:, 4] = yp
    cart[:, 5] = zp

    if kepl.shape[0] == 1:
       cart = np.squeeze(cart)

  return cart


def df_kepl_2_cart(ic_kepl,unit='rad'):
  """ Return a dataframe of cartesian coordinates from a dataframe of orbital elements.

  Parameters
  ----------

  ic_kepl: dataframe
    Orbital elements (a,e,i,raan,omega,ma).
  unit: string
    Unit of the angles ('rad' or 'deg').

  Returns
  -------
 
  ic_cart: dataframe
    Cartesian coordinates (x[m],v[m/s]).

  """

  headers = ['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]']

  if unit=='deg':
    kepl = ic_kepl[['a[m]','e','i[deg]','raan[deg]','omega[deg]','ma[deg]']]
  elif unit=='rad':
    kepl = ic_kepl[['a[m]','e','i[rad]','raan[rad]','omega[rad]','ma[rad]']]
  ic_cart = kepl_2_cart(kepl.to_numpy(),earth_mu,unit)  

  if len(ic_cart.shape)==1:
    ic_cart = pd.DataFrame([ic_cart],columns=headers)  
  else:
    ic_cart = pd.DataFrame(ic_cart,columns=headers)
  
  ic_cart = pd.DataFrame(ic_cart,columns=headers)
  if 'date' in ic_kepl.columns:
    ic_cart['date'] = ic_kepl['date']
  else:
    ic_cart.index = ic_cart.index
    
  
  return ic_cart


def df_cart_2_kepl(ic_cart,unit='rad'):
  """ Return a dataframe of orbital elements from a dataframe of cartesian coordinates.

  Parameters
  ----------

  ic_cart: dataframe
    Cartesian coordinates (x[m],v[m/s]).
  unit: str
    Unit of the angles ('rad' or 'deg').

  Returns
  -------
 
  ic_kepl: dataframe
    Orbital elements (a,e,i,raan,omega,ma).

  """

  if unit=='deg':
    headers = ['a[m]','e','i[deg]','raan[deg]','omega[deg]','ma[deg]']
  elif unit=='rad':
    headers = ['a[m]','e','i[rad]','raan[rad]','omega[rad]','ma[rad]']

  cart = ic_cart[['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]']]
  ic_kepl = cart_2_kepl(cart.to_numpy(),earth_mu,unit)

  if len(ic_kepl.shape)==1:
    ic_kepl = pd.DataFrame([ic_kepl],columns=headers)  
  else:
    ic_kepl = pd.DataFrame(ic_kepl,columns=headers)

  if 'date' in ic_cart.columns:
    ic_kepl['date'] = ic_cart['date']
  else:
    ic_kepl.index = ic_cart.index

  return ic_kepl


def df_kepl_2_cart_old(ic_kepl,unit='rad'):
  """ Return a dataframe of cartesian coordinates from a dataframe of orbital elements.

  Parameters
  ----------

  ic_kepl: dataframe
    Orbital elements.
  unit: string
    Unit of the angles ('rad' or 'deg').


  Returns
  -------
 
  ic_cart: dataframe
    Cartesian coordinates.

  """

  headers_kepl = ['a[m]','e','i[deg]','raan[deg]','omega[deg]','ma[deg]']
  
  ic_cart = []
  headers_cart = ['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]']
  for i in range(0,len(ic_kepl),1):
    kepl = ic_kepl.iloc[i][headers_kepl].tolist()
    kepl[2] = kepl[2]*np.pi/180.  
    kepl[3] = kepl[3]*np.pi/180.
    kepl[4] = kepl[4]*np.pi/180.
    kepl[5] = kepl[5]*np.pi/180.
    ic_cart.append(kepl_2_cart(kepl,earth_mu,unit='deg'))
    
  ic_cart = pd.DataFrame(ic_cart,columns=headers_cart)
  if 'date' in ic_kepl.columns:
    ic_cart['date'] = ic_kepl['date']
  else:
    ic_cart.index = ic_cart.index
    
  return ic_cart


def df_cart_2_kepl_old(ic_cart,unit='rad'):
  """ Return a dataframe of orbital elements from cartesian coordinates.

  Parameters
  ----------

  ic_cart: dataframe
    Cartesian coordinates.
  unit: string
    Unit of the angles ('rad' or 'deg').

  Returns
  -------
 
  ic_kepl: dataframe
    Orbital elements.

  """

  headers_cart = ['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]']

  ic_kepl = []
  headers_kepl = ['a[m]','e','i[deg]','raan[deg]','omega[deg]','ma[deg]']
  for i in range(0,len(ic_cart),1):
    cart = ic_cart.iloc[i][headers_cart].tolist()
    kepl = cart_2_kepl(cart,earth_mu,unit='deg')
    kepl[2] = kepl[2]*180./np.pi  
    kepl[3] = kepl[3]*180./np.pi
    kepl[4] = kepl[4]*180./np.pi
    kepl[5] = kepl[5]*180./np.pi
    ic_kepl.append(kepl)    

  ic_kepl = pd.DataFrame(ic_kepl,columns=headers_kepl)
  if 'date' in ic_cart.columns:
    ic_kepl['date'] = ic_cart['date']
  else:
    ic_kepl.index = ic_cart.index
  

  return ic_kepl


def cart_2_sph(cart):
  """ Return spherical coordinates from cartesian coordinates.

  Parameters
  ----------

  cart: float array
    Cartesian coordinates (x[m],v[m/s]).

  Returns
  -------

  kepl: float array
    Spherical coordinates (alt,long,lat,dalt,dlong,dlat).
 
  """

  sph = np.zeros(6)
  r = cart[0]**2.0+cart[1]**2.0
  #check if we have a degenerate case
  if r>0 :     
    sph[0] = np.sqrt(cart[0]**2.0+cart[1]**2.0+cart[2]**2.0)
    sph[1] = math.atan2(cart[1],cart[0])%(2*np.pi)
    sph[2] = math.asin(cart[2]/sph[0])
  else:
    sph[1] = 0
    if cart[2]==0:
        sph[2]=0
    else:
      if cart[2]>0:
        sph[2]=np.pi/2.
      else:
        sph[2]=-np.pi/2.
    r = np.abs(cart[2])     
  sph[3] = (cart[0]*cart[3]+cart[1]*cart[4]+cart[2]*cart[5])/sph[0]
  sph[4] = (cart[0]*cart[4]-cart[3]*cart[1])/(cart[0]**2.0+cart[1]**2.0) 
  sph[5] = (cart[5]*(cart[0]**2.0+cart[1]**2.0)-cart[2]*(cart[0]*cart[3]+cart[1]*cart[4])) / (sqrt(cart[0]**2.0+cart[1]**2.0)*(cart[0]**2.0+cart[1]**2.0+cart[2]**2.0))

  return sph


def sph_2_cart(sph):
  """ Return cartesian coordinates from spherical coordinates.
    
  Parameters
  ----------

  sph: float array
    Spherical coordinates (alt,long,lat,dalt,dlong,dlat)

  Returns
  -------

  cart: float array
    Cartesian coordinates (x[m],v[m/s])
   
  """

  cart = np.zeros(6)
  cart[0] = sph[0]*cos(sph[1])*sin(sph[2]) 
  cart[1] = sph[0]*sin(sph[1])*sin(sph[2])
  cart[2] = sph[0]*cos(sph[2])             
  cart[3] = sph[3]*(cos(sph[2])*cos(sph[1]))-sph[0]*( sph[5]*sin(sph[2])*cos(sph[1])+sph[4]*cos(sph[2])*sin(sph[1]))
  cart[4] = sph[3]*(cos(sph[2])*sin(sph[1]))+sph[0]*(-sph[5]*sin(sph[2])*sin(sph[1])+sph[4]*cos(sph[2])*cos(sph[1]))
  cart[5] = sph[3]*sin(sph[2])+sph[0]*sph[5]*cos(sph[2])
  
  return cart


def kepl_2_equinoctial(kepl,mu):
  """ Return equinoctial coordinates from orbital elements.
  Ref: Vallado, 2012, p. 108
    
  Parameters
  ----------

  kepl: float array
    Keplerian elements (a,e,i,raan,omega,ma).
  mu: float
    Gravitational constant time the Earth mass.    

  Returns
  -------

  equinoc: float array
    Equinoctial coordinates (a,k,h,l,p,q) 
 
  """   

  #check if retrograde
  if i>90:
    f = -1
  else:
    f = 1
        
  equinoc = np.zeros(6)  
  equinoc[0] = kepl[0]                                   
  equinoc[1] = kepl[1]*cos(kepl[3]+f*kepl[4])
  equinoc[2] = kepl[1]*cos(kepl[3]+f*kepl[4])
  equinoc[3] = kepl[5]+kepl[4]+kepl[3]
  equinoc[4] = sin(kepl[2])*sin(kepl[3])/(1+cos(kepl[2])**f)
  equinoc[5] = sin(kepl[2])*cos(kepl[3])/(1+cos(kepl[2])**f)

  return equinoc


def cart_2_flight_elements(cart):
    """ Return flight elements from cartesian coordinates.

    Parameters
    ----------

    cart: float array
      Cartesian coordinates
 
    Returns
    -------

    flight_elem: float array
      Flight elements

    """

    flight_elem = np.zeros(6)

    #position and velocity magnitude
    r = np.sqrt(cart[0]**2+cart[1]**2+cart[2]**2)
    v = np.sqrt(cart[3]**2+cart[4]**2+cart[5]**2)

    #right ascension (alpha)
    alpha = np.acos(cart[0]/np.sqrt(cart[0]**2+cart[1]**2))
    
    #declination (delta)    
    delta = np.asin(cart[2]/r)

    #matrice to transform a vector in the SEZ system
    m = np.zeros(3,3)
    m[0,0] = np.cos(alpha)*np.cos(delta)
    m[0,1] = np.sin(alpha)*np.cos(delta)
    m[0,2] = -np.sin(delta)
    m[1,0] = -np.sin(alpha)
    m[1,1] = np.cos(alpha)
    m[1,2] = 0.0    
    m[1,0] = np.sin(delta)*np.cos(alpha)
    m[1,1] = np.sin(alpha)*np.sin(delta)
    m[1,2] = np.cos(delta)
    v_sez = m*cart[3:5]

    #flight-path angle from the local horizontal
    phi = np.acos(v_sez[2]/v)
    
    #azimut (beta)
    beta = np.acos(-v_sez[0]/np.sqrt(v_sez[0]**2+v_sez[1]**2))
    
    flight_elem[0] = r 
    flight_elem[1] = v
    flight_elem[2] = alpha   
    flight_elem[3] = delta
    flight_elem[4] = phi
    flight_elem[5] = beta   

    return flight_elem


def mean_anomaly_2_ecc_anomaly(e,ma,precision=1e-14,unit='rad'):
  """ Return eccentric anomaly from mean anomaly.
    
  Parameters
  ----------

  ecc: float
    Excentricity
  ma: float
    Mean anomaly

  Returns
  -------

  ea: float
    Eccentric anomaly

  """

  if unit=='deg':
    ma = ma*np.pi/180.
  elif unit=='rad':
    pass    
  else:
    print('Error: bad unit')
    sys.exit()

  if np.any(e<0):
    print('Error with the eccentricity: e < 0')
  elif np.any(e>1):
    print('Error with the eccentricity: e > 1')
  else:

    U = 0.
    e2 = e*e
    ANOM0 = (2*np.pi+ma)%(2*np.pi)
      
    ee2 = e2*0.5
    ee3 = ee2*e*0.25
    u = ANOM0+(e-ee3)*np.sin(ANOM0)+ee2*np.sin(2.*ANOM0)+3.*ee3*np.sin(3.*ANOM0)
    tol = 5.e-13
    correc = 1.
    niter = 0

    while (np.abs(correc)>tol):
      CORREC = (ANOM0-(u-e*np.sin(u)))/(1.-(e*np.cos(u)))
      u = u + CORREC
      niter = niter + 1
      if (niter>=100): 
        #print "Nombre maximal d'etapes de resolution de l'equation de kepler atteint : 100"
        break
      
    #We solve Kepler's equation (Vallado, 2014, p.65)
     
    #if (((ma<0) and (ma>-np.pi)) or(ma>np.pi)):
    #  ea0 = ma - ecc
    #else:
    #  ea0 = ma + ecc
      
    #ea=1E6
    #while(np.abs(ea-ea0)>precision):
    #  ea0 = ea 
    #  ea = ea0 + (ma-ea0+ecc*sin(ea0))/(1-ecc*cos(ea0))

  if unit=='deg':
    u = u*180./np.pi  
    
  return u


def true_anomaly_2_ecc_anomaly(ecc,nu,unit='rad'):
  """ Return eccentric anomaly from true anomaly.

  Parameters
  ----------

  ecc: float
    Eccentricity
  nu: float
    True anomaly
  unit: string
    Unit of the angle.

  Returns
  -------

  ea: float
    Eccentric anomaly
  
  """

  if unit=='deg':
    nu = nu*np.pi/180.
  elif unit=='rad':
    pass    
  else:
    print('Error: bad unit')
    sys.exit()

  ea = atan2(sqrt(1-ecc**2)*sin(nu),ecc+cos(nu))
  ea = fmod(ea,2*np.pi)

  if unit=='deg':
    ea = ea*180./np.pi  
    
  return ea


def ecc_anomaly_2_mean_anomaly(ecc,ea,unit='rad'):
  """ Return mean anomaly from eccentric anomaly.
    
  Parameters
  ----------

  ecc: float
    Eccentricity
  ea: float
    Eccentric anomaly    

  Returns
  -------

  ma: float
    Mean anomaly
 
  """

  if unit=='deg':
    ea = ea*np.pi/180.
  elif unit=='rad':
    pass    
  else:
    print('Error: bad unit')
    sys.exit()
    
  ma = ea-ecc*sin(ea)
  ma = fmod(ma,2*np.pi)
  if ma<0:
    ma += 2*np.pi

  if unit=='deg':
    ma = ma*180./np.pi
    
  return ma


def true_anomaly_2_mean_anomaly(ecc,nu,unit='rad'):
  """ Return mean anomaly from true anomaly.

  INPUTS
  ------

  ecc: float
    Eccentricity
  nu: float
    True anomaly

  RETURN
  ------

  ea: float
    Eccentric anomaly

  """

  if unit=='deg':
    nu = nu*np.pi/180.
  elif unit=='rad':
    pass    
  else:
    print('Error: bad unit')
    sys.exit()
    
  ea = true_anomaly_2_ecc_anomaly(ecc,nu,unit=unit)
  ma = ecc_anomaly_2_mean_anomaly(ecc,ea,unit=unit)
  ma = fmod(ma,2*np.pi)

  if unit=='deg':
    ma = ma*180./np.pi
    
  return ma


def ecc_anomaly_2_true_anomaly(ecc,ea,unit='rad'):
  """ Return true anomaly from eccentric anomaly.
    
  INPUTS
  ------

  ecc: float
    Eccentricity
  ea: float
    Eccentric anomaly

  RETURN
  ------

  nu: float
    True anomaly

  """

  if unit=='deg':
    ea = ea*np.pi/180.
  elif unit=='rad':
    pass
  else:
    print('Error: bad unit')
    sys.exit()
    
  nu = 2*atan2(sqrt(1+ecc)*sin(ea/2.),sqrt(1-ecc)*cos(ea/2.))

  if unit=='deg':
    nu = nu*180./np.pi
  
  return nu


def mean_anomaly_2_true_anomaly(ecc,ma,tolerance=1e-14,unit='rad'):
  """ Return true anomaly from mean anomaly.
    
  INPUTS
  ------

  ecc: float
    Excentricity
  ma: float
    Mean anomaly

  RETURN
  ------

  nu: float
    True anomaly

  """

  if unit=='deg':
    ma = ma*np.pi/180.
  elif unit=='rad':
    pass
  else:
    print('Error: bad unit')
    sys.exit()
  
  ea = mean_anomaly_2_ecc_anomaly(ecc,ma,tolerance)
  nu = ecc_anomaly_2_true_anomaly(ecc,ea)

  if unit=='deg':
    nu = nu*180./np.pi
  
  return nu


def flight_path_angle(ecc,ea,unit='rad'):
  """ Return flight path angle.

  Paramters
  ---------

  ecc: float
    Eccentricity.
  ea: float
    Eccentric anomaly.
  unit: string
    Unit of the angle.
 
  Returns
  ------

  fpa: float
    Flight path angle.

  """

  if unit=='rad':
    pass
  elif unit=='deg':
    ea = ea*np.pi/180.
  else:
    print('Bad type of unit')

  if ecc==0:

    fpa = 0.
    
  elif ecc>0 and ecc<1.:

    cosfpa = np.sqrt((1-ecc**2)/(1-ecc**2*np.cos(ea)**2))
    sinfpa = ecc*np.sin(ea)/np.sqrt(1-ecc**2*np.cos(ea)**2) 
    fpa = atan2(sinfpa,cosfpa)

  elif ecc==1:

    nu = ecc_anomaly_2_true_anomaly(ecc,ea)  
    fpa = nu/2.

  elif ecc>1:

    print('Path flight: work in progress...')
    sys.exit()
    cosfpa = np.sqrt((ecc**2-1)/(ecc**2*np.cosh(ea)**2-1))
    sinfpa = -ecc*np.sinh(ea)/np.sqrt(ecc**2*np.cos(ea)**2-1) 
    fpa = atan2(sinfpa,cosfpa)
      
  else:

    print('Error: ecc < 0')

  if unit=='degree':
    fpa = fpa*180./np.pi
    
  return fpa


def dms_2_rad(dms):
  """ Return angle in radians from angle in degrees, minutes, and seconds.

  Parameters
  ----------

  dms: float array (dim 3)
    Degrees, minutes, and seconds

  Returns
  -------

  angle: float
    Angle in radians.

  """

  if dms[0]<0:
    angle = (np.abs(dms[0])+dms[1]/60.+dms[2]/3600.)
    angle = -angle
  else:
    angle = (np.abs(dms[0])+dms[1]/60.+dms[2]/3600.)

  angle = angle*np.pi/180.
    
  return angle


def hms_2_rad(hms):
  """ Return angle in radians from an angle in hours, minutes, and seconds.

  Parameters
  ----------

  hms: float array (dim 3)
    Hours, minutes, and seconds

  Returns
  -------

  angle: float
    Angle in radians.

  """

  angle = 15*(hms[0]+hms[1]/60.+hms[2]/3600.)
  
  return angle


def degree_2_hms(angle):
  """ Return angle in hours, minutes, and seconds from angle in degress.
  
  Parameters
  ----------

  angle: float
    Angle in degress.

  Returns
  -------

  hh: float
    Hours.
  mm: float:
    Minutes.
  ss: float
    Seconds.

  """ 

  hh = int(angle*24/360.)
  mm = int((angle-hh*360./24.)/(360./24./60.))
  ss = (angle-hh*360./24.-mm*360./24./60.)/(360./24./60./60.)
  
  return hh,mm,ss


def degree_2_dms(angle):
  """ Return angle in degrees, minutes, and seconds from angle in degress.
  
  Paramters
  ---------

  angle: float
    Angle in degress.

  Returns
  -------

  dd: float
    Degress.
  mm: float:
    Minutes.
  ss: float
    Seconds.

  """
  
  dd = int(angle)
  mm = int((angle-dd)*60.)
  ss = (angle-dd-mm/60.)*3600.
  
  return dd,mm,ss


def geocentric_2_topocentric(theta_lst,lat_gd,r_geo,unit='rad'):
  """ Return the cartesian position in the geocentric frame from cartesian position in the topocentric frame defined by geodetic coordinates (Ref: Vallado, p.162).

  INPUTS
  ------

  theta_lst: float
    Local sideral time [deg]
  lat_gd: float
    Geodedic latitude [deg].
  r_geo: array of float
    Position in the geocentric frame.

  RETURN
  ------

  r_topo: float array
    Position in the topocentric frame.

  """

  if unit=='deg':  
    theta_lst = theta_lst*np.pi/180.
    lat_gd = lat_gd*np.pi/180.
  elif unit=='rad':
    pass
  else:
    print('Error: bad unit')
    sys.exit()    
    
  rot = rotation(np.pi/2.-lat_gd,2).dot(rotation(theta_lst,3))
  r_topo = rot.dot(r_geo)

  return r_topo


def topocentric_2_geocentric(theta_lst,lat_gd,r_topo,unit='rad'):
  """ Return the cartesian position in the topocentric frame defined by geodetic coordinates from cartesian position in the topocentric frame (Ref: Vallado, p.162).

  Paramters
  ---------

  theta_lst: float
    Local sideral time.
  lat_gd: float
    Geodedic latitude.
  r_topo: array of float
    Position in the topocentric frame.

  Returns
  -------

  r_geo: float array
    Cartesian position in the geocentric frame.

  """

  if unit=='deg':
    theta_lst = theta_lst*np.pi/180.
    lat_gd = lat_gd*np.pi/180.
  elif unit=='rad':
    pass
  else:
    print('Error: bad unit')
    sys.exit()
    
  rot = rotation(-theta_lst,3).dot(rotation(-(np.pi/2.-lat_gd),2))
  r_geo = rot.dot(r_topo)

  return r_geo


def latlon_geocentric_2_ecef(phi_gc,lamb,h_ell,unit='rad'):
  """ Give the cartesian coordinate of a station from the 
  geocentric longitude and latitude (Vallado, p.144).

  Parameters
  ----------

  phi_gc: float
    Decimal geocentric latitude  [deg].
  lamb: float
    Decimal longitude [deg].
  h_ell: float
    Ellipsoidal height [m].

  Returns
  -------

  r: array of float
    Cartesian coordinates in meters.

  """

  if angle=='deg':
    phi_gc = phi_gc*np.pi/180.
    lamb = lamb*np.pi/180. 
  elif angle=='rad':
    pass
  else:
    print('Error: bad unit')
    sys.exit()

  r = np.zeros(3)
  r[0] = np.cos(phi_gc)*np.cos(lamb)
  r[1] = np.cos(phi_gc)*np.sin(lamb)
  r[2] = np.sin(phi_gc)
  
  return r


def latlon_2_ecef(phi_gd,lamb,h_ell,unit='rad'):
  """ Return cartesian coordinate of a station from the 
  geodedic longitude, latitude and altitude.
  Ref: Vallado, p.144.

  Parameters
  ----------

  phi_gd: float
    Decimal geodetic latitude.
  lamb: float
    Decimal longitude.
  h_ell: float
    Ellipsoidal height [m].
  unit: string
    Unit of the angles ('rad' or 'deg').

  Returns
  -------

  r: array of float
    Cartesian coordinates in [m] and [m/s].

  """

  if unit=='deg':
    phi_gd = phi_gd*np.pi/180.
    lamb = lamb*np.pi/180. 
  elif unit=='rad':
    pass
  else:
    print('Bad angle unit')
    sys.exit()
      
  #earth_eq_radius = 6378.137E3#63E3
  #e_earth = np.sqrt(0.006694385)

  c_earth = earth_eq_radius/np.sqrt(1-e_earth**2*np.sin(phi_gd)**2)
  s_earth = c_earth*(1-e_earth**2)
  #print(c_earth)
  #print(s_earth)

  r = np.zeros(6)
  r[0] = (c_earth+h_ell)*np.cos(phi_gd)*np.cos(lamb)
  r[1] = (c_earth+h_ell)*np.cos(phi_gd)*np.sin(lamb)
  r[2] = (s_earth+h_ell)*np.sin(phi_gd)
  
  return r


def ecef_2_latlon(r_ecef):
  """ Return longitude, latitude and altitude from cartesian coordinates in the ECEF frame. 
  Ref: Algorithm 12, p.172 in Vallado 2012.
    
  Parameters
  ----------

  r_ecef: float array
    Cartesian coordinates in the ECEF frame [m].

  Returns
  -------

  lon: float
    Longitude [rad]
  lat: float
    Geodetic latitude [rad]

  """

  e_earth = np.sqrt(0.006694385)
    
  r = np.sqrt(r_ecef[0]**2+r_ecef[1]**2+r_ecef[2]**2)
  r_delta = np.sqrt(r_ecef[0]**2+r_ecef[1]**2)
  
  sin_alpha = r_ecef[1]/r_delta
  cos_alpha = r_ecef[0]/r_delta
  tan_alpha = sin_alpha/cos_alpha
  alpha     = atan(tan_alpha)
  alpha     = atan2(r_ecef[1],r_ecef[0])

  delta = asin(r_ecef[2]/r)
  #print 'Alpha:',alpha*180./np.pi
  #print 'Delta:',delta*180./np.pi

  phi_gd = delta
  counter = 0
  phi_gd_old = 1E10
  c_earth = earth_eq_radius/np.sqrt(1-e_earth**2*np.sin(phi_gd)**2)

  while np.abs(phi_gd-phi_gd_old)>5.E-10:
    phi_gd_old = phi_gd
    tanphi_gd = (r_ecef[2]+c_earth*e_earth**2*np.sin(phi_gd))/r_delta
    phi_gd    = atan(tanphi_gd)
    counter += 1
    if counter>10:
      break

  h_ell = r_delta/np.cos(phi_gd)-c_earth
  if phi_gd*180./np.pi>89:
    s_earth = earth_eq_radius*(1-e_earth**2)/np.sqrt(1-e_earth**2*np.sin(phi_gd)**2) 
    h_ell = r_ecef[2]/np.sin(phi_gd)-s_earth
  
  lamb   = alpha    #Geodetic longitude
  phi_gd = delta    #Geodetic latitude

  return lamb,phi_gd,h_ell


def rotation(alpha,axe,unit='rad'):
  """ Return a matrix rotation.
  Ref: Vallado p.162
 
  Parameters
  ----------
 
  alpha: float 
    Angle of rotation.
  axe: integer
    Axe of rotation.
  unit: string
    Unit of the angles ('rad' or 'deg').

  Returns
  -------

  rot: matrix of float [3x3]
    Matrix rotation.

  """

  if unit=='deg':
    alpha = alpha*np.pi/180.
  elif unit=='rad':
    pass
  else:
    print('Error: bad unit')
    sys.exit()
  
  rot = np.zeros((3,3))   
  if axe==1:
    rot[0,0] = 1
    rot[1,1] = np.cos(alpha)
    rot[1,2] = np.sin(alpha)
    rot[2,1] = -np.sin(alpha)
    rot[2,2] = np.cos(alpha)
  elif axe==2:
    rot[0,0] = np.cos(alpha)
    rot[0,2] = -np.sin(alpha)
    rot[1,1] = 1
    rot[2,0] = np.sin(alpha)
    rot[2,2] = np.cos(alpha)
  elif axe==3:
    rot[0,0] = np.cos(alpha)
    rot[0,1] = np.sin(alpha)
    rot[1,0] = -np.sin(alpha)
    rot[1,1] = np.cos(alpha)
    rot[2,2] = 1
  else:
    print('Error: bad choice of the rotation axis.')
    sys.exit()

  return rot 

    
def convert_ephemeris_2_snapshot(list_ephemeris):
  """ Return a list of snapshots from a list of ephemerides.
  
  Parameters
  ----------

  list_ephemeris: list of dataframes
    List of ephemeris.
  
  Returns
  -------

  list_snapshots: list of dataframes
    List of snapshots.

  """

  list_snapshots = []

  nb = 0
  for item in list_ephemeris:
    if len(item)>nb:
      nb = len(item)

  for i in range(0,nb,1):
    list_coord = []
    for ephemeris in list_ephemeris:
      if i<len(ephemeris):
        list_coord.append(ephemeris[i:i+1])
        date = ephemeris[i:i+1]['Date'].iloc[0]
    snapshot = pd.concat(list_coord,ignore_index=True)
    list_snapshots.append([date,snapshot])                          
    
  return list_snapshots



