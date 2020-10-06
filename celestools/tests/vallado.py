#!/usr/bin/python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*-coding:Utf-8 -*

"""
  vallado.py
  Author: Alexis Petit, Share My Space
  Date 2020/03/25
  It allows to check the algorithms of Vallado.
"""


#Packages


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime,timedelta


#Packages celestialpy


sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))


from celestools import earth_mu,earth_eq_radius
from celestools import cart_2_kepl,kepl_2_cart
from celestools import df_cart_2_kepl,df_kepl_2_cart
from celestools import true_anomaly_2_ecc_anomaly
from celestools import flight_path_angle
from celestools import TLE_data
from celestools import date_2_jd

#chapter 3
from celestools import greenwich_mean_sideral_time
from celestools import convert_frame_cart_orekit


#Code


def test_flight_path_angle():  
  """ Reproduction of the figure 2-18, p.105 in Vallado (2012)
  """

  print('Flight path angle')
  print('p.105 in Vallado (2012)')
  
  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.set_xlabel('True anomaly [deg]',fontsize=14)
  ax1.set_ylabel('Flight path angle [deg]',fontsize=14)
  ax1.set_xlim([-180,180])
  
  list_nu = range(-180,181,1)
  list_ecc = [0.0,0.2,0.5,0.98]
  for ecc in list_ecc:
    list_fpa = []
    for nu in list_nu:
      ea = true_anomaly_2_ecc_anomaly(ecc,nu*np.pi/180.)
      list_fpa.append(np.abs(flight_path_angle(ecc,ea)*180./np.pi))
    plt.plot(list_nu,list_fpa,label='e = '+str(ecc))

  plt.legend()
  plt.savefig('true_anomaly_vs_flight_path_angle')
  plt.close()
  
  print('')
  

def cartesian_keplerian_coordinates():
  """ Transformation between cartesian and keplerian coordinates.
  Example p.114 in Vallado (2012)
  """

  print('Cartesian coordinates - orbital elements')
  print('p.114 in Vallado (2012)')
  
  cart = np.zeros(6)
  cart[0] = 6524.834E3
  cart[1] = 6862.875E3
  cart[2] = 6448.296E3  
  cart[3] = 4901.327
  cart[4] = 5533.756  
  cart[5] = 1976.341

  print('r = ',cart[0:3],' m ')
  print('v = ',cart[3:6],' m/s ')

  kepl = cart_2_kepl(cart,earth_mu,unit='deg')

  print('a     = ',kepl[0],' m ')
  print('e     = ',kepl[1])
  print('i     = ',kepl[2],' degrees ')
  print('RAAN  = ',kepl[3],' degrees ')
  print('Omega = ',kepl[4],' degrees ')
  print('M     = ',kepl[5],' degrees ')

  cart = kepl_2_cart(kepl,earth_mu,unit='deg')
  
  print('r = ',cart[0:3],' m ')
  print('v = ',cart[3:6],' m/s ')
  print('')

  print('Test with a dataframe')
  cart_df = [cart,cart]
  cart_df = [cart]
  headers = ['x[m]','y[m]','z[m]','vx[m/s]','vy[m/s]','vz[m/s]']
  cart_df = pd.DataFrame(cart_df,columns=headers)
  print('Cartesian coordinates')
  print(cart_df)
  kepl_df = df_cart_2_kepl(cart_df,unit='deg')
  print('Keplerian coordinates')
  print(kepl_df)
  cart_df = df_kepl_2_cart(kepl_df,unit='deg')
  print('Cartesian coordinates')
  print(cart_df)
  print('')
  

def tle_reader():
  """ Read a TLE observation.
  """

  print('Test of the TLE reader')
  date = None

  line0 = 'TEST'
  line1 = '1   118U 61015C   20009.67606489 -.00000070 +00000-0 +58253-5 0  9996'
  line2 = '2   118 066.7632 157.1433 0077640 129.1689 231.6327 14.01664051986663'

  #line0 = '0 DELTA 2 R/B(1)'
  #line1 = '1 20362U 89097B   20071.03900496  .07981546  21192-5  38161-3 0  9999'
  #line2 = '2 20362  35.5926  37.1119 0002167 220.5407 139.5348 16.42979716656623'

  #line0 = '0 MICROSAT-R DEB'
  #line1 = '1 44154U 19006AR  20082.58437406  .18764251  24859-5  65438-2 0  9998'
  #line2 = '2 44154  96.5830 264.5300 0041199  27.7648 332.5831 16.22864770 49549'
  
  tlelines = [line0,line1,line2]

  print(line0)
  print(line1)
  print(line2)
  print('')

  tle = TLE_data(tlelines,'name_plus_2_lines',method='sgp4',coord='kepl',frame='teme',date=date,unit='deg')
  print('TLE in keplerian coordinates in TEME (deg)')
  print(tle.observations)
  
  tle = TLE_data(tlelines,'name_plus_2_lines',method='sgp4',coord='kepl',frame='gcrf',date=date,unit='deg')
  print('TLE in keplerian coordinates in GCRF (deg)')
  print(tle.observations)

  tle = TLE_data(tlelines,'name_plus_2_lines',method='sgp4',coord='kepl',frame='gcrf',date=date,unit='rad')
  print('TLE in keplerian coordinates (rad)')
  print(tle.observations)  

  tle = TLE_data(tlelines,'name_plus_2_lines',method='sgp4',coord='cart',frame='gcrf',date=date,unit='deg')
  print('TLE in cartesian coordinates')
  print(tle.observations)
  print('')


############################

# Vallado, chapter 3

############################


def sideral_time():
  """ Give the sideral time. 
  """

  print('')
  print('> Give the sideral time')
  print('> Example 3.6 p.189 in Vallado 2012')

  date = datetime(1992,8,20,12,14,00)
  print('Date: ',date)

  jd = date_2_jd(date)
  gmst,lst = greenwich_mean_sideral_time(jd,-104)

  print('GMST = {0} deg (152.578 788 27)'.format(gmst))
  print('LST  = {0} deg (48.578 788 27)'.format(lst))
  print('')  
  

def iau76_reduction():
  """ Transform cartesian coordinates from ITRF to GCFR.
  """

  print('')
  print('From ITRF to GCRF')
  print('> Example 3-15 p.230 in Vallado 2012')

  cart_itrf = np.zeros(6)
  cart_itrf[0] = -1033.479383E3
  cart_itrf[1] = 7901.2952754E3
  cart_itrf[2] = 6380.3565958E3
  cart_itrf[3] = -3.225636520E3
  cart_itrf[4] = -2.872451450E3
  cart_itrf[5] = 5.531924446E3
  date = datetime(2004,4,6,7,51,28,386006)

  print('Position [m]:   ',cart_itrf[0:3])
  print('Velocity [m/s]: ',cart_itrf[3:6])
  print('Date:           ',date)
  print('JD:             ',date_2_jd(date))
  
  headers = ["date","x[m]","y[m]","z[m]","vx[m/s]","vy[m/s]","vz[m/s]"]
  cart_itrf = [[date,cart_itrf[0],cart_itrf[1],cart_itrf[2],cart_itrf[3],cart_itrf[4],cart_itrf[5]]]
  cart_itrf = pd.DataFrame(cart_itrf,columns=headers)

  #nutation_file = '../data/nut_IAU1980.dat'
  #cart_gcrf = fk5_itrf_2_gcrf(cart_itrf,date,nutation_file,coord='posvit')
  cart_gcrf = convert_frame_cart_orekit(cart_itrf,'itrf','gcrf')
  
  print('')
  print('Vector position GCRF: ')
  print('rx= {0} km ( 5102.508 959)'.format(cart_gcrf['x[m]'][0]/1000.))
  print('ry= {0} km ( 6123.011 403)'.format(cart_gcrf['y[m]'][0]/1000.))
  print('rz= {0} km ( 6378.136 925 )'.format(cart_gcrf['z[m]'][0]/1000.))
  print('vx= {0} km/s ( -4.743 220 16)'.format(cart_gcrf['vx[m/s]'][0]/1000.))
  print('vy= {0} km/s (  0.790 536 50)'.format(cart_gcrf['vy[m/s]'][0]/1000.))
  print('vz= {0} km/s (  5.553 755 73)'.format(cart_gcrf['vz[m/s]'][0]/1000.))

  #cart_itrf = fk5_gcrf_2_itrf(cart_gcrf,date,nutation_file,coord='posvit')
  cart_gcrf = convert_frame_cart_orekit(cart_itrf,'gcrf','itrf')

  print('')
  print('Vector position ITRF: ')
  print('rx= {0} km ( -1033.479 382)'.format(cart_itrf['x[m]'][0]/1000.))
  print('ry= {0} km (  7901.295 275 4)'.format(cart_itrf['y[m]'][0]/1000.))
  print('rz= {0} km (  6380.356 595 8)'.format(cart_itrf['z[m]'][0]/1000.))
  print('vx= {0} km/s ( -3.225 636 520 )'.format(cart_itrf['vx[m/s]'][0]/1000.))
  print('vy= {0} km/s ( -2.872 451 450 )'.format(cart_itrf['vy[m/s]'][0]/1000.))
  print('vz= {0} km/s (  5.531 924 446 )'.format(cart_itrf['vz[m/s]'][0]/1000.))
  print('')
  
  
##########################################################################

##########################################################################

##########################################################################
#
# Main
if __name__ == "__main__":

  test_flight_path_angle()  
  cartesian_keplerian_coordinates()
  tle_reader()

  print('')
  print('--- Test from Vallado ---')
  list_chap = ['chap3']  
  
  if 'chap3' in list_chap:

    print('')
    print('--- Chapter 3 ---')
    sideral_time()                  #Example 3-6  p.189
    iau76_reduction()               #Example 3-15 p.230
