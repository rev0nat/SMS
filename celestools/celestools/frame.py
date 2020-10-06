#!/usr/bin/python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*-coding:Utf-8 -*

"""
  # frame.py
  
  * Author: Delphine Ly, Share My Space
  * Date: 2020/04/07
  * It allows to perform coordinate transformations.
"""


#Packages


import astropy.units as u
import numpy as np
import pandas as pd
import sys
import os

from datetime import datetime
from math import pi

from astropy.constants import GM_earth
from astropy.coordinates import (CartesianRepresentation, CartesianDifferential)
from astropy.coordinates import SkyCoord, ITRS, GCRS
from scipy.spatial.transform import Rotation as R
from pyproj import Transformer
from tqdm import tqdm

import orekit
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir()
if os.path.exists('orekit-data.zip')==False:
  orekit.pyhelpers.download_orekit_data_curdir()

from org.orekit.utils import Constants
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.utils import PVCoordinates, AbsolutePVCoordinates, TimeStampedPVCoordinates, IERSConventions
from org.orekit.frames import FramesFactory
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.orbits import CartesianOrbit, KeplerianOrbit, PositionAngle, OrbitType, Orbit, CircularOrbit

teme_f = FramesFactory.getTEME()
utc = TimeScalesFactory.getUTC()
itrf_f = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
gcrf_f = FramesFactory.getGCRF()
eme2000_f = FramesFactory.getEME2000()
muearth = Constants.WGS84_EARTH_MU


#Code


__all__ = [
    'itrf_latlon',
    'convert_frame_cart_orekit',
    'gcrf_itrf_astropy',
    'itrf_gcrf_astropy',
    'xyz_to_rsw',
    'xyz_to_ntw',
    'ntw_frame',
    'rsw_frame',
    'kepler_polarnodal',
    'polarnodal_cartesian',
    'kepler_gcrf_orekit'
]


def itrf_latlon(itrf):
        
  # 5332 is ITRF2008's EPSG
  # 4326 is WGS84's EPSG
  # itrf is a DataFrame with XYZ coordinates in ITRF2008
  # returns a DataFrame with corresponding latitudes, longitudes and altitudes

  x = itrf['x[m]']
  x= x.tolist()
  y = itrf['y[m]']
  y=y.tolist()
  z = itrf['z[m]']
  z=z.tolist()
  coords= list(zip(x, y, z))

  conv = Transformer.from_proj(5332,4326,always_xy=True)
  data = conv.itransform(coords)
  latlon = pd.DataFrame(data, columns= ['lon[deg]', 'lat[deg]', 'alt[m]'])
  latlon.index = itrf.index

  return latlon


def convert_cart_single_orekit(to_conv,ref_to_conv,target_ref):
  """ Coordinate frame transformation with Orekit modules.
 
  INPUTS
  ------

  to_conv: dataframe
    Cartesian coordinates.
  ref_to_conv: string
    Initial reference frame ('itrf','gcrf','teme', 'eme2000')
  target_ref: string
    Final reference frame ('itrf','gcrf','teme', 'eme2000')
  
  RETURN
  ------

  dframe: dataframe
    Cartesian coordinates.

  """

  date = to_conv.index[0]
  X = float(to_conv['x[m]'][0])
  Y = float(to_conv['y[m]'][0])
  Z = float(to_conv['z[m]'][0])
  Vx = float(to_conv['vx[m/s]'][0])
  Vy = float(to_conv['vy[m/s]'][0])
  Vz = float(to_conv['vz[m/s]'][0])
  ok_date = AbsoluteDate(date.year, date.month, date.day, date.hour, date.minute,
                         date.second + float(date.microsecond) / 1000000., utc)
  PV_coordinates = PVCoordinates(Vector3D(X, Y, Z), Vector3D(Vx, Vy, Vz))
  start_state = TimeStampedPVCoordinates(ok_date, PV_coordinates)
  if ref_to_conv == 'itrf':
    frame_to_conv = itrf_f
  elif ref_to_conv == 'gcrf':
    frame_to_conv = gcrf_f
  elif ref_to_conv == 'teme':
    frame_to_conv = teme_f
  elif ref_to_conv == 'eme2000':
    frame_to_conv = eme2000_f
  else:
    print(f'Unknown reference frame: {ref_to_conv}')
  if target_ref == 'itrf':
    target_frame = itrf_f
  elif target_ref == 'gcrf':
    target_frame = gcrf_f
  elif target_ref == 'teme':
    target_frame = teme_f
  elif target_ref == 'eme2000':
    target_frame = eme2000_f
  else:
    print(f'Unknown reference frame: {target_ref}')

  state_to_conv = AbsolutePVCoordinates(frame_to_conv,start_state)
  target_state = AbsolutePVCoordinates(target_frame,state_to_conv.getPVCoordinates(target_frame))
  pos = target_state.position
  vel = target_state.velocity
  X = pos.getX()
  Y = pos.getY()
  Z = pos.getZ()
  Vx = vel.getX()
  Vy = vel.getY()
  Vz = vel.getZ()
  dframe = pd.DataFrame({'x[m]': [X], 'y[m]': [Y], 'z[m]': [Z], 'vx[m/s]': [Vx], 'vy[m/s]': [Vy], 'vz[m/s]': [Vz]},index=to_conv.index)

  return dframe


def convert_frame_cart_orekit(to_conv_df,ref_to_conv,target_ref):
  """ Return cartesian coordinates after frame transformation with Orekit modules.
 
  Parameters
  ----------

  to_conv: dataframe
    Cartesian coordinates in the initial frame.
  ref_to_conv: string
    Initial reference frame ('itrf','gcrf','teme', 'eme2000')
  target_ref: string
    Final reference frame ('itrf','gcrf','teme', 'eme2000')
  
  Returns
  -------

  conv_df: dataframe
    Cartesian coordinates in the final frame.

  """

  save_index = to_conv_df.index

  if 'date' in to_conv_df.columns:
    to_conv_df.index = to_conv_df['date'] 
  
  conv_df = pd.DataFrame()
  for i in range(len(to_conv_df)):
    conv_df = pd.concat([conv_df,convert_cart_single_orekit(to_conv_df[i:i+1],ref_to_conv,target_ref)])

  conv_df.index = save_index
  if 'date' in to_conv_df.columns:
    to_conv_df.index = save_index
    conv_df['date'] = to_conv_df['date'] 
  
  return conv_df


def gcrs_itrs_single(gcrs):
  """ Transformation GCRS to ITRS.
  """
  
  utc = TimeScalesFactory.getUTC()
  date = gcrs.index[0]
  X = float(gcrs['x[m]'][0])
  Y = float(gcrs['y[m]'][0])
  Z = float(gcrs['z[m]'][0])
  Vx = float(gcrs['vx[m/s]'][0])
  Vy = float(gcrs['vy[m/s]'][0])
  Vz = float(gcrs['vz[m/s]'][0])
  ok_date = AbsoluteDate(date.year, date.month, date.day, date.hour, date.minute,
                         date.second + float(date.microsecond) / 1000000., utc)
  PV_coordinates = PVCoordinates(Vector3D(X, Y, Z), Vector3D(Vx, Vy, Vz))
  start_state = TimeStampedPVCoordinates(ok_date, PV_coordinates)
  state_gcrf = AbsolutePVCoordinates(gcrf, start_state)
  state_itrf = AbsolutePVCoordinates(itrf, state_gcrf.getPVCoordinates(itrf))
  pos = state_itrf.position
  vel = state_itrf.velocity
  X = pos.getX()
  Y = pos.getY()
  Z = pos.getZ()
  Vx = vel.getX()
  Vy = vel.getY()
  Vz = vel.getZ()
  dframe = pd.DataFrame({'x[m]': [X], 'y[m]': [Y], 'z[m]': [Z], 'vx[m/s]': [Vx], 'vy[m/s]': [Vy], 'vz[m/s]': [Vz]}, index=gcrs.index)
    
  return dframe


def itrs_gcrs_single(itrs):
  """ Transformation ITRS to GCRS.
  """

  utc = TimeScalesFactory.getUTC()
  date = itrs.index[0]
  X = float(itrs['x[m]'][0])
  Y = float(itrs['y[m]'][0])
  Z = float(itrs['z[m]'][0])
  Vx = float(itrs['vx[m/s]'][0])
  Vy = float(itrs['vy[m/s]'][0])
  Vz = float(itrs['vz[m/s]'][0])
  ok_date = AbsoluteDate(date.year, date.month, date.day, date.hour, date.minute,
                         date.second + float(date.microsecond) / 1000000., utc)
  PV_coordinates = PVCoordinates(Vector3D(X, Y, Z), Vector3D(Vx, Vy, Vz))
  start_state = TimeStampedPVCoordinates(ok_date, PV_coordinates)
  state_itrf = AbsolutePVCoordinates(itrf, start_state)
  state_gcrf = AbsolutePVCoordinates(gcrf, state_itrf.getPVCoordinates(gcrf))
  pos = state_gcrf.position
  vel = state_gcrf.velocity
  X = pos.getX()
  Y = pos.getY()
  Z = pos.getZ()
  Vx = vel.getX()
  Vy = vel.getY()
  Vz = vel.getZ()
  dframe = pd.DataFrame({'x[m]': [X], 'y[m]': [Y], 'z[m]': [Z], 'vx[m/s]': [Vx], 'vy[m/s]': [Vy], 'vz[m/s]': [Vz]}, index=itrs.index)

  return dframe


def gcrs_itrs(gcrs_df):
  """ Transformation ITRS to GCRS for a dataframe.
  """

  if 'date' in gcrs_df.columns:
    gcrs_df.index = gcrs_df['date']
  
  itrs_df = pd.DataFrame()
  for i in range(len(gcrs_df)):
      itrs_df = pd.concat([itrs_df, gcrs_itrs_single(gcrs_df[i:i + 1])])
      
  return itrs_df


def itrf_gcrf_astropy(itrf):
  """ Coordinate frame transformation ITRF to GCRF with Astropy modules.    

  INPUTS
  ------    

  itrf: dataframe
    Cartesian coordinates in ITRF frame.    

  RETURN
  ------    
  
  gcrf: dataframe
    Cartesian coordinates in GCRF frame.   
 
  """

  epochs = itrf.index.tolist()
  a = 0
  if 'x[m]' not in itrf.columns:
    itrf = itrf.rename(columns ={"precise_x[m]": "x[m]", "precise_y[m]":"y[m]","precise_z[m]": "z[m]",
                            "precise_vx[m/s]": "vx[m/s]","precise_vy[m/s]": "vy[m/s]",
                            "precise_vz[m/s]": "vz[m/s]"})
    a = 1

  x = itrf['x[m]'].astype(str).astype(float).values
  y = itrf['y[m]'].astype(str).astype(float).values
  z = itrf['z[m]'].astype(str).astype(float).values
  x *= u.m
  y *= u.m
  z *= u.m
  vx = itrf['vx[m/s]'].astype(str).astype(float).values
  vx *= u.m / u.s
  vy = itrf['vy[m/s]'].astype(str).astype(float).values
  vy *= u.m / u.s
  vz = itrf['vz[m/s]'].astype(str).astype(float).values
  vz *= u.m / u.s
  itrfs = SkyCoord(x=x,y=y,z=z,frame='itrs',v_x=vx,v_y=vy,v_z=vz,obstime=epochs, \
                     representation_type=CartesianRepresentation,differential_type=CartesianDifferential)
  gcrs = itrfs.transform_to(GCRS(obstime=epochs,representation_type=CartesianRepresentation, \
                                   differential_type=CartesianDifferential))
  v_vec = [gcrs.velocity.d_x.to(u.m/u.s).value, gcrs.velocity.d_y.to(u.m/u.s).value,
             gcrs.velocity.d_z.to(u.m/u.s).value]
  p_vec = [gcrs.cartesian.x.to(u.m).value,gcrs.cartesian.y.to(u.m).value,gcrs.cartesian.z.to(u.m).value]
  v_vec = np.array(v_vec).transpose()  # dimension (3, number samples)
  p_vec = np.array(p_vec).transpose()  # dimension (3, number samples)

  if a == 0:
    gcrs = pd.DataFrame(np.concatenate([p_vec,v_vec],axis=1),
                            columns=['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]'])
  if a == 1:
    gcrs = pd.DataFrame(np.concatenate([p_vec,v_vec],axis=1),
                            columns=['precise_x[m]','precise_y[m]','precise_z[m]','precise_vx[m/s]',
                                     'precise_vy[m/s]','precise_vz[m/s]'])
  gcrs.index = itrf.index

  return gcrs

    
def gcrf_itrf_astropy(gcrs):
  """ Coordinate frame transformation GCRF to ITRF with Astropy modules.    

  INPUTS
  ------    

  itrf: dataframe
    Cartesian coordinates in GCRF frame.    

  RETURN
  ------    

  itrf: dataframe
    Cartesian coordinates in ITRF frame.    

  """
    
  if 'x[m]' not in gcrs.columns:
    gcrs = gcrs.rename(columns ={"precise_x[m]": "x[m]", "precise_y[m]":"y[m]","precise_z[m]": "z[m]",
                            "precise_vx[m/s]": "vx[m/s]","precise_vy[m/s]": "vy[m/s]",
                            "precise_vz[m/s]": "vz[m/s]"})
  epochs = gcrs.index.tolist()
  x = gcrs['x[m]'].astype(str).astype(float).values
  y = gcrs['y[m]'].astype(str).astype(float).values
  z = gcrs['z[m]'].astype(str).astype(float).values
  x *= u.m
  y *= u.m
  z *= u.m
  vx = gcrs['vx[m/s]'].astype(str).astype(float).values
  vx *= u.m / u.s
  vy = gcrs['vy[m/s]'].astype(str).astype(float).values
  vy *= u.m / u.s
  vz = gcrs['vz[m/s]'].astype(str).astype(float).values
  vz *= u.m / u.s
  gcrss = SkyCoord(x=x, y=y, z=z,frame='gcrs',v_x=vx,v_y=vy,v_z=vz,obstime=epochs, \
                   representation_type=CartesianRepresentation, differential_type=CartesianDifferential)
  itrss = gcrss.transform_to(ITRS(obstime=epochs, representation_type=CartesianRepresentation, \
                                  differential_type=CartesianDifferential))
  vec = [itrss.cartesian.x.to(u.m).value, itrss.cartesian.y.to(u.m).value, itrss.cartesian.z.to(u.m).value,
         itrss.velocity.d_x.to(u.m / u.s).value, itrss.velocity.d_y.to(u.m / u.s).value,
         itrss.velocity.d_z.to(u.m / u.s).value]
  vec = np.array(vec).transpose()
  itrs = pd.DataFrame(vec, columns=['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]'])
  itrs.index = gcrs.index

  return itrs

  
def xyz_to_rsw(vec_xyz_df,reference):
  """ Compute cartesian coordinates in the RSW frame.


  INPUTS
  ------

  vec_xyz_df: dataframe
    Residu between two serie of ephemeris.
  reference: dataframe
    Ephemeris of reference.

  RETURN
  ------

  vec_rsw: dataframe
    Coordinates in RSW frame.
  
  """

  headers = ['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]']

  list_vec_rsw = []  
  for i in range(0,len(reference),1):

    coord_xyz = np.zeros(6)
    coord_xyz[0] = reference.iloc[i]['x[m]']
    coord_xyz[1] = reference.iloc[i]['y[m]']    
    coord_xyz[2] = reference.iloc[i]['z[m]']      
    coord_xyz[3] = reference.iloc[i]['vx[m/s]']
    coord_xyz[4] = reference.iloc[i]['vy[m/s]']    
    coord_xyz[5] = reference.iloc[i]['vz[m/s]']

    vec_xyz = np.zeros(6)
    vec_xyz[0] = vec_xyz_df.iloc[i]['x[m]']
    vec_xyz[1] = vec_xyz_df.iloc[i]['y[m]']
    vec_xyz[2] = vec_xyz_df.iloc[i]['z[m]']
    vec_xyz[3] = vec_xyz_df.iloc[i]['vx[m/s]']
    vec_xyz[4] = vec_xyz_df.iloc[i]['vy[m/s]']
    vec_xyz[5] = vec_xyz_df.iloc[i]['vz[m/s]']

    rsw = rsw_frame(coord_xyz)
    vec_rsw = np.zeros(6)
    vec_rsw[0:3] = np.dot(rsw,vec_xyz[0:3])
    vec_rsw[3:6] = np.dot(rsw,vec_xyz[3:6])
    list_vec_rsw.append([vec_rsw[0],vec_rsw[1],vec_rsw[2],vec_rsw[3],vec_rsw[4],vec_rsw[5]])
      
  headers = ["dr[m]","ds[m]","dw[m]","dvr[m/s]","dvs[m/s]","dvw[m/s]"]
  vec_rsw = pd.DataFrame(list_vec_rsw,columns=headers)
  if 'date' in reference.columns:
    vec_rsw['date'] = reference['date'] 

  vec_rsw.index = reference.index
    
  return vec_rsw


def xyz_to_ntw(vec_xyz_df,reference):
  """ Compute cartesian coordinates in the NTW frame.


  INPUTS
  ------

  vec_xyz_df: dataframe
    Residu between two serie of ephemeris.
  reference: dataframe
    Ephemeris of reference.

  RETURN
  ------

  vec_ntw: dataframe
    Coordinates in NTW frame.
  
  """

  headers = ['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]']

  list_vec_ntw = []  
  for i in range(0,len(reference),1):

    coord_xyz = np.zeros(6)
    coord_xyz[0] = reference.iloc[i]['x[m]']
    coord_xyz[1] = reference.iloc[i]['y[m]']    
    coord_xyz[2] = reference.iloc[i]['z[m]']      
    coord_xyz[3] = reference.iloc[i]['vx[m/s]']
    coord_xyz[4] = reference.iloc[i]['vy[m/s]']    
    coord_xyz[5] = reference.iloc[i]['vz[m/s]']

    vec_xyz = np.zeros(6)
    vec_xyz[0] = vec_xyz_df.iloc[i]['x[m]']
    vec_xyz[1] = vec_xyz_df.iloc[i]['y[m]']
    vec_xyz[2] = vec_xyz_df.iloc[i]['z[m]']
    vec_xyz[3] = vec_xyz_df.iloc[i]['vx[m/s]']
    vec_xyz[4] = vec_xyz_df.iloc[i]['vy[m/s]']
    vec_xyz[5] = vec_xyz_df.iloc[i]['vz[m/s]']

    ntw = ntw_frame(coord_xyz)
    vec_ntw = np.zeros(6)
    vec_ntw[0:3] = np.dot(ntw,vec_xyz[0:3])
    vec_ntw[3:6] = np.dot(ntw,vec_xyz[3:6])
    list_vec_ntw.append([vec_ntw[0],vec_ntw[1],vec_ntw[2],vec_ntw[3],vec_ntw[4],vec_ntw[5]])

  headers = ["dn[m]","dt[m]","dw[m]","dvn[m/s]","dvt[m/s]","dvw[m/s]"]
  vec_ntw = pd.DataFrame(list_vec_ntw,columns=headers)
  if 'date' in reference.columns:
    vec_ntw['date'] = reference['date']

  vec_ntw.index = reference.index  
  
  return vec_ntw


def ntw_frame(cart):
  """ Give the matrix to pass from cartesian to the NTW frame.
      Normal, in-track, cross-track.

  INPUTS
  ------

  cart: float array
    Cartesian coordinates (r[m],v[m/s]).

  RETURN
  ------

  ntw: matrix of floats
    Passage matrix.

  """

  t = cart[3:6]/np.sqrt(np.dot(cart[3:6],cart[3:6]))
  w = np.cross(cart[0:3],cart[3:6])
  w = w[0:3]/np.sqrt(np.dot(w[0:3],w[0:3]))
  n = np.cross(cart[3:6],w[0:3])
  n = n[0:3]/np.sqrt(np.dot(n[0:3],n[0:3]))
  
  ntw = np.zeros((3,3))
  ntw[0,0:3] = n[0:3]
  ntw[1,0:3] = t[0:3] 
  ntw[2,0:3] = w[0:3]
  
  return ntw


def rsw_frame(cart):
  """ Give the passage matrix for the RSW frame.
      Radial, along-track, cross-track.

  coord_xyz: array of float
    Cartesian coordinates.

  RETURN
  ------

  rsw: matrix of floats
    Passage matrix.

  """

  r = cart[0:3]/np.sqrt(np.dot(cart[0:3],cart[0:3]))
  w = np.cross(cart[0:3],cart[3:6],axis=0)
  w = w[0:3]/np.sqrt(np.dot(w[0:3],w[0:3]))
  s = np.cross(r[0:3],w[0:3],axis=0)
  s = s/np.sqrt(np.dot(s[0:3],s[0:3]))  
  
  rsw = np.zeros((3,3))
  rsw[0:3,0] = r[0:3]
  rsw[0:3,1] = s[0:3]
  rsw[0:3,2] = w[0:3]

  return rsw


def ntw_frame_vec(cart):
  """ Give the matrix to pass from cartesian to the NTW frame.
      Normal, in-track, cross-track.

  INPUTS
  ------
  cart: float array
    Cartesian coordinates (r[m],v[m/s]).

  RETURN
  ------

  ntw: matrix of floats
    Passage matrix.

  """
  
  t = cart[3:6]/np.sqrt(np.sum(cart[3:6]**2,axis=0))
  w = np.cross(cart[0:3],cart[3:6],axis=0)
  w = w[0:3]/np.sqrt(np.sum(w[0:3]**2,axis=0))
  n = np.cross(cart[3:6],w[0:3],axis=0)
  n = n[0:3]/np.sqrt(np.sum(n[0:3]**2,axis=0))
  ntw = np.zeros((3,)+n.shape)
  ntw[0,0:3] = n[0:3]
  ntw[1,0:3] = t[0:3]
  ntw[2,0:3] = w[0:3]
  
  return ntw


def add_ntw(df):
  """ Add local NTW coordinates in df, with respect to the accurate coordinates.
  """

  # Compute the rotation matrix.
  precise_coord = ['precise_x[m]'   , 'precise_y[m]'   , 'precise_z[m]'   , \
                   'precise_vx[m/s]', 'precise_vy[m/s]', 'precise_vz[m/s]']
  cart = df[precise_coord]
  cart = np.array(cart)

  if len(cart.shape) > 1:
    cart=np.transpose(cart)
  rot_mat = ntw_frame_vec(cart)
  # Add the relative positions in the NTW frame
  df['dn[m]'] = df['x[m]'] - df['precise_x[m]']
  df['dt[m]'] = df['y[m]'] - df['precise_y[m]']
  df['dw[m]'] = df['z[m]'] - df['precise_z[m]']
  ntw = ['dn[m]', 'dt[m]', 'dw[m]']
  df[ntw] = np.transpose(np.einsum('ijk,jk->ik', rot_mat, np.transpose(df[ntw])))

  # Add the relative velocities in the NTW frame
  df['dvn[m/s]'] = df['vx[m/s]'] - df['precise_vx[m/s]']
  df['dvt[m/s]'] = df['vy[m/s]'] - df['precise_vy[m/s]']
  df['dvw[m/s]'] = df['vz[m/s]'] - df['precise_vz[m/s]']
  ntw = ['dvn[m/s]', 'dvt[m/s]', 'dvw[m/s]']
  df[ntw] = np.transpose(np.einsum('ijk,jk->ik', rot_mat, np.transpose(df[ntw])))
    
  return df


def kepler_polarnodal(kep):
  
  # Units for angles: rad, for distance: km
  # input must be floats, not quantities (no units)
  # input must have columns ['a[m]', 'e', 'i[rad]', 'raan[rad]', 'omega[rad]', 'ta[rad]']
  epsilon = 10e-8
  a, e, f, i, raan, w, m = kep['a[m]'], kep['e'], kep['ta[rad]'], kep['i[rad]'], kep['raan[rad]'], kep['omega[rad]'], kep.index.tolist()
  r = a * (1 - e ** 2) / (1 + e * np.cos(f) + epsilon)  # unit: m
  L = np.sqrt(GM_earth.value * a)  # unit: m2/s
  U = L * np.sqrt(1 - e ** 2 + epsilon)  # unit: m2/s  (norm? of angular momentum)
  H = U * np.cos(i)  # unit: m2/s (projection of angular momentum on z)
  t = w + f  # unit: rad
  t = t.map(lambda x: x if x < 2 * pi else x - 2 * pi)
  R = e * U * np.sin(f) / (a * (1 - e ** 2 + epsilon))  # unit: m/s
  polarnodal = pd.concat([r, t, raan, R, U, H, i], axis=1)
  polarnodal.columns = ['r[m]', 'arglat[rad]', 'raan[rad]', 'vel', 'norm_angmom[m2/s]', 'z_angmom[m2/s]', 'i[rad]']
  polarnodal['date'] = m
  polarnodal.set_index('date', inplace=True)

  return polarnodal


def polarnodal_cartesian(pn):
  
  alpha = pn['arglat[rad]']
  beta = pn['i[rad]']
  gamma = pn['raan[rad]']

  pn['zeros'] = np.zeros(len(pn))
  angles = pd.concat([alpha, beta, gamma], axis=1).values

  r = R.from_euler('zxz', angles, degrees=False)
  pos = pd.concat([pn['r[m]'], pn['zeros'], pn['zeros']], axis=1)
  vel = pd.concat([pn['vel'], (pn['norm_angmom[m2/s]'] / pn['r[m]']), pn['zeros']], axis=1)

  pos_gcrs = r.apply(pos)
  vel_gcrs = r.apply(vel)

  gcrs = pd.DataFrame(np.concatenate([pos_gcrs, vel_gcrs], axis=1), columns=['x[m]', 'y[m]', 'z[m]', 'vx[m/s]', 'vy[m/s]', 'vz[m/s]'])
  gcrs.index = pn.index

  return gcrs


def kepler_gcrs_single(kep):

  date = kep.index[0]
  ok_date = AbsoluteDate(date.year, date.month, date.day, date.hour, date.minute,
                         date.second + float(date.microsecond) / 1000000., utc)
  kep = kep.astype(float)
  a, e, f, i, raan, w, date = kep['a[m]'][0], kep['e'][0], kep['ta[rad]'][0], kep['i[rad]'][0], kep['raan[rad]'][0], kep['omega[rad]'][0], \
                                kep.index[0]
  kep_orbit = KeplerianOrbit(a, e, i, w, raan, f, PositionAngle.TRUE, teme_f, ok_date, muearth)

  state = AbsolutePVCoordinates(gcrf_f, kep_orbit.getPVCoordinates(gcrf_f))
  pos = state.position
  vel = state.velocity
  X = pos.getX()
  Y = pos.getY()
  Z = pos.getZ()
  Vx = vel.getX()
  Vy = vel.getY()
  Vz = vel.getZ()
  dframe = pd.DataFrame({'x[m]': [X], 'y[m]': [Y], 'z[m]': [Z], 'vx[m/s]': [Vx], 'vy[m/s]': [Vy], 'vz[m/s]': [Vz]},index=kep.index)
  return dframe


def kepler_gcrf_orekit(df):
  
  cart_df = pd.DataFrame()
  for i in range(len(df)):
    cart_df = pd.concat([cart_df, kepler_gcrs_single(df[i:i + 1])])

  return cart_df
