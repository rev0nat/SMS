#!/usr/bin/python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*-coding:Utf-8 -*

"""
  time_conversion.py
  Author: Alexis Petit, PhD Student, Namur University
  Date: 2017 03 01
  It contains methods to convert time scales.
"""


#Package


import numpy as np
from jdcal import gcal2jd,jd2gcal
from datetime import datetime,date,timedelta
import os
import sys


#Code


__all__ = [
    'jd_2_decimal_year',
    'decimal_year_2_date',
    'date_2_decimal_year',
    'date_2_jd',
    'jd_2_date',
    'jd_2_jd_cnes',
    'jd_cnes_2_jd',
    'jd_2_theta',
    'hms_2_rad',
    'rad_2_hms',
    'hms_2_tod',
    'tod_2_hms',
    'utc_2_tai',
    'tai_2_utc',
    'greenwich_mean_sideral_time',
    'conv_time'
]


def jd_2_decimal_year(jd):
    """ Compute the decimal year from the Julian day.

    INPUT
    -----

    jd: float
      The date Julian day

    """

    date = jd_2_date(jd)
    year = date/strftime('%Y')
    dec  = date.strftime('%j')

    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
      year += dec/366. #leap year
    else:
      year += dec/365.      

    frac_day = (date.strftime('%H')*3600.+date.strftime('%M')*60.+date.strftime('%S'))/86400.
    year += frac_day    
    
    return year


def decimal_year_2_date(decimal_year):
    """ Compute the date from the decimal year.

    Input
    ----------   
    decimal_year : scalar
      The decimal year.

    Return
    ------
    date : 
      The date.

    """  

    start  = decimal_year
    year   = int(start)
    rem    = start - year
    base   = datetime(year,1,1)
    date = base + timedelta(seconds=(base.replace(year=base.year+1)-base).total_seconds()*rem)

    return date 
    

def date_2_decimal_year(date):
    """ Compute the decimal year from the date.

    INPUT
    -----

    date: datetime
      The date.

    RETURN
    ------

    decimal_year: float
      The decimal year

    """ 
  
    if ((float(date.strftime('%Y'))%400==0) or (float(date.strftime('%Y'))%4==0 and float(date.strftime('%Y'))%100!=0)):
      decimal_year = float(date.strftime('%Y'))+float(date.strftime('%j'))/366
    else:
      decimal_year = float(date.strftime('%Y'))+float(date.strftime('%j'))/365
    
    return decimal_year


def date_2_jd(date):
  """ Compute the Julian day from the date.

  INPUT
  -----

  date: datetime
    The date.

  RETURN
  ------

  jd: float
    The Julian day

  """
    
  yy = int(date.strftime('%Y'))
  mm = int(date.strftime('%m'))
  dd = int(date.strftime('%d'))
  hh = int(date.strftime('%H'))
  mi = int(date.strftime('%M'))
  ss = float(date.strftime('%S.%f'))
     
  sec = hh*3600.+mi*60.+ss
  day_dec = sec/86400.
    
  jd = np.sum(gcal2jd(yy,mm,dd))+day_dec
  
  return jd


def jd_2_date(julian_day):
    """ Compute the date from the Julian day.

    INPUT
    -----

    julian_day: float
      The date in Julian day.

    RETURN
    ------

    t: datetime
      The date.

    """
    
    date = jd2gcal(2400000.5,julian_day-2400000.5)  
    sec_day = date[3]*24*3600
    year = date[0] 
    month= date[1]
    day  = date[2]
    hour = int(divmod(sec_day,3600)[0])
    minu = int(divmod(sec_day-hour*3600,60)[0])
    seco = int(divmod(sec_day-hour*3600,60)[1])
    frac = int((divmod(sec_day-hour*3600,60)[1]-seco)*1E6)
    t = datetime(year,month,day,hour,minu,seco,frac)

    return t


def jd_2_jd_cnes(jd):
    """ From Julian day compute the Julian day CNES and the number of second.

    INPUT
    -----

    jd: float
      The date in Julian day

    RETURN
    ------

    jd_cnes: integer 
      The Julian day CNES
    sec_cnes: float
      The number of second CNES

    """

    jd_cnes  = int(jd-2433282.5)
    sec_cnes = ((jd-2433282.5)-int(jd-2433282.5))*86400.0

    return jd_cnes,sec_cnes


def jd_cnes_2_jd(jd_cnes,sec_cnes):
    """ From the Julian day CNES and the number of second compute the Julian day.

    INPUT
    -----

    jd_cnes: integer 
      The Julian day CNES
    sec_cnes: float
      The number of second CNES

    RETURN
    ------

    jd: float
      The date in Julian day

    """

    jd = jd_cnes + 2433282.5 + sec_cnes/86400. 

    return jd


def jd_2_theta(jd):
  """ From the decimal Julian day compute Greenwich Mean Sideral Time (GMST).

  INPUT
  -----
 
  jd: float
    The Julian day

  RETURN
  ------

  theta: float
    The GMST

  """

  mjd = jd-2400000.5
  theta = ((280.4606+360.9856473*(mjd-51544.5))*np.pi/180.)%2*np.pi

  return theta


def hms_2_rad(h,minu,s):
  """ Conversion angle (Vallado, p.198).

  INPUT
  -----
 
  h: integer
   Hour
  minu: integer
   Minute
  s: float
   Second

  RETURN
  ------

  tau: float
    Angle in radian

  """

  tau = 15*(h+minu/60.+s/3600.)*np.pi/180
  
  return tau

def rad_2_hms(tau):
  """ Conversion angle (Vallado, p.198).

  INPUT
  -----
 
  tau: float
    Angle in radian

  RETURN
  ------

  h: integer
   Hour
  minu: integer
   Minute
  s: float
   Second

  """
  
  temp = tau*180/(15*np.pi)

  h = int(temp)
  minu = int((temp-h)*60)
  s = (temp-h-minu/60.)*3600.
  
  return h,minu,s


def hms_2_tod(h,minu,s):
  """ Conversion angle for the time of the day (Vallado, p.198).

  INPUT
  -----
 
  h: integer
   Hour
  minu: integer
   Minute
  s: float
   Second

  RETURN
  ------

  tod: float
    Time of the day

  """
  
  tod = 3600*h+60*minu+s

  return tod


def tod_2_hms(tod):
  """ Conversion angle for the time of the day (Vallado, p.198).

  INPUT
  -----
 
  tod: float
    Time of the day

  RETURN
  ------

  h: integer
   Hour
  minu: integer
   Minute
  s: float
   Second

  """

  temp = tod/3600.
  
  h = int(temp)
  minu = int((temp-h)*60)
  s = (temp-h-minu/60.)*3600

  return h,minu,s


def utc_2_tai(utc):
  """ From UTC compute TAI.

  INPUT
  -----

  RETURN
  ------

  """

  tai = utc - utc_tai(utc)/86400.

  return tai


def tai_2_utc(tai):
  """ From TAI compute UTC.

  INPUT
  -----

  tai: float
    TAI

  RETURN
  ------

  UTC: float
    UTC

  """

  utc = tai + utc_tai(tai)/86400.

  return utc


def utc_tai(jd):
  """ Time scale conversion.

  INPUT
  -----
 
  time: datetime
    Time in TAI or UTC and Julian day

  RETURN
  ------

  difsec: float
    Gap in seconds.

  """

  #Derniere modification : 1er janvier 2017 JJcnes 24472 JJ 2457754
  if jd>=2457754.5:
    difsec = -37.
  elif (jd>=2457204.5) and (jd<2457754.5):
    difsec = -36.
  elif (jd>=2456109.5) and (jd<2457204.5):
    difsec = -35.
  elif (jd>=2454832.5) and (jd<2456109.5):
    difsec =-34.
  elif (jd>=2453736.5) and (jd<2454832.5):
    difsec = -33.
  elif (jd>=2451179.5) and (jd<2453736.5):
    difsec = -32.
  elif (jd>=2450630.5) and (jd<2451179.5):
    difsec = -31.
  elif (jd>=2450083.5) and (jd<2450630.5):
    difsec = -30.
  elif (jd>=2449534.5) and (jd<2450083.5):
    difsec = -29.
  elif (jd>=2449169.5) and (jd<2449534.5):
    difsec = -28.
  elif (jd>=2448804.5) and (jd<2449169.5):
    difsec = -27.
  elif (jd>=2448257.5) and (jd<2448804.5):
    difsec =-26.
  elif (jd>=2447892.5) and (jd<2448257.5):
    difsec = -25.
  elif (jd>=2447161.5) and (jd<2447892.5):
    difsec = -24.
  elif (jd>=2446247.5) and (jd<2447161.5):
    difsec = -23.
  elif (jd>=2445516.5) and (jd<2446247.5):
    difsec = -22.
  elif (jd>=2445151.5) and (jd<2445516.5):
    difsec = -21.
  elif (jd>=2444786.5) and (jd<2445151.5):
    difsec = -20.
  elif (jd>=2444239.5) and (jd<2444786.5):
    difsec = -19.
  elif (jd>=2443874.5) and (jd<2444239.5):
    difsec = -18.
  elif (jd>=2443509.5) and (jd<2443874.5):
    difsec = -17.
  elif (jd>=2443144.5) and (jd<2443509.5):
    difsec = -16.
  elif (jd>=2442778.5) and (jd<2443144.5):
    difsec = -15.
  elif (jd>=2442413.5) and (jd<2442778.5):
    difsec = -14.
  elif (jd>=2442048.5) and (jd<2442413.5):
    difsec = -13.
  elif (jd>=2441683.5) and (jd<2442048.5):
    difsec = -12.
  elif (jd>=2441499.5) and (jd<2441683.5):
    difsec = -11.
  elif (jd>=2441317.5) and (jd<2441499.5):
    difsec = -10.
  else:
    difsec = 0.
        
  return difsec


def greenwich_mean_sideral_time(jd,longitude=0.0):
  """ Give the Greenwich Mean Sideral Time (GMST) 
  from the Julian day. Reference: Algorithm 15, p.188 in Vallado 2012. 

  INPUT
  -----

  jd: float
    Julian day.
  longitude: float
    Longitude [deg]

  RETURN
  ------
 
  theta: float
    Greenwich Mean Sideral Time [deg].
    
  """

  t_ut1 = (jd-2451545.0)/36525.
  hour = 3600.
  gmst_0 = 67310.54841+(876600*hour+8640184.812866)*t_ut1+0.093104*t_ut1**2-6.2E-6*t_ut1**3   
  gmst = (gmst_0/240.)%360.

  if longitude==0.0:      
    return gmst
  else:
    lst = gmst + longitude  
    return gmst,lst


def date_2_dat_dut1(date):
  """
  Table p.191

  INPUTS
  ------

  date: datetime

  """

  dat  = 0.
  dut1 = 0.

  if date>datetime(1972,1,1):
    dat  = 10
    dut1 = 0.   
  if date>datetime(1972,7,1):
    dat  = 11
    dut1 = 0.
  if date>datetime(1973,1,1):
    dat  = 12
    dut1 = 0.
  if date>datetime(1974,1,1):
    dat  = 13
    dut1 = 0.
  if date>datetime(1975,1,1):
    dat  = 14
    dut1 = 0.
  if date>datetime(1976,1,1):
    dat  = 15
    dut1 = 0.
  if date>datetime(1977,1,1):
    dat  = 16
    dut1 = 0.
  if date>datetime(1978,1,1):
    dat  = 17
    dut1 = 0.
  if date>datetime(1979,1,1):
    dat  = 18
    dut1 = 0.
  if date>datetime(1980,1,1):
    dat  = 19
    dut1 = 0.
  if date>datetime(1981,7,1):
    dat  = 20
    dut1 = 0.    
  if date>datetime(1982,7,1):
    dat  = 21
    dut1 = 0.
  if date>datetime(1983,7,1):
    dat  = 22
    dut1 = 0.
  if date>datetime(1985,7,1):
    dat  = 23
    dut1 = 0.
  if date>datetime(1988,1,1):
    dat  = 24.
    dut1 = 0.364300    
  if date>datetime(1989,1,1):
    dat  = 24.  
    dut1 = -0.116600
  if date>datetime(1989,6,1):
    dat  = 24.
    dut1 = -0.386640
  if date>datetime(1990,1,1):
    dat  = 25.
    dut1 = 0.328640
  if date>datetime(1991,1,1):
    dat  = 26.
    dut1 = 0.618670
  if date>datetime(1992,1,1):
    dat  = 26.
    dut1 = -0.125300
  if date>datetime(1992,7,1):
    dat  = 27.
    dut1 = 0.4429100
  if date>datetime(1994,1,1):
    dat  = 28.
    dut1 = 0.1992800
  if date>datetime(1994,7,1):
    dat  = 29.
    dut1 = 0.7826700
  if date>datetime(1995,1,1):
    dat  = 29.
    dut1 = 0.3985000
  if date>datetime(1996,1,1):
    dat  = 30.
    dut1 = 0.5552900
  if date>datetime(1997,1,1):
    dat  = 30.
    dut1 = -0.1110260
  if date>datetime(1997,7,1):
    dat  = 31.
    dut1 = 0.5268940
  if date>datetime(1998,1,1):
    dat  = 31.
    dut1 = 0.2181430
  if date>datetime(1999,1,1):
    dat  = 32.
    dut1 = 0.7166370
  if date>datetime(2000,1,1):
    dat  = 32.
    dut1 = 0.3555000
  if date>datetime(2001,1,1):
    dat  = 32.
    dut1 = 0.0932276
  if date>datetime(2002,1,1):
    dat  = 32.
    dut1 = -0.1158223
  if date>datetime(2003,1,1):
    dat  = 32.
    dut1 = -0.2893180
  if date>datetime(2004,1,1):
    dat  = 32.
    dut1 = -0.3895908
  if date>datetime(2005,1,1):
    dat  = 32.
    dut1 = -0.5039024
  if date>datetime(2005,12,1):
    dat  = 32.
    dut1 = -0.6611230
  if date>datetime(2006,1,1):
    dat  = 33.
    dut1 = 0.3388335
  if date>datetime(2009,1,1):
    dat  = 34.
    dut1 = 0.4071728

  return dat,dut1


def conv_time(date):#,dut1,dat):
  """ Time converter.
  Algorithm 16, p. 195, Vallado 2013.

  INPUTS
  ------

  date: datetime
    Date and time UTC
  dut1: float
  dat: float

  RETURN
  ------

  ut1: float
  tai: float
  tt: float
  tdb: float

  """

  dat,dut1 = date_2_dat_dut1(date)
  #print dat,dut1
  #dut1 = -0.463326
  
  date_ut1 = date+timedelta(microseconds=dut1*1E6)
  date_tai = date.replace(microsecond=0)+timedelta(seconds=dat)
  date_gps = date.replace(microsecond=0)+timedelta(seconds=(dat-19.))
  date_tt  = date_tai+timedelta(seconds=32,microseconds=184000)

  jd_tt   = date_2_jd(date_tt)
  t_tt    = (jd_tt-2451545.0)/36525.
  jd_tai  = date_2_jd(date_tai)
  t_tai   = (jd_tai-2451545.0)/36525.

  dt_sec_tdb  = 0.001657*np.sin(628.3076*t_tt+62401)
  dt_msec_tdb = (dt_sec_tdb-int(dt_sec_tdb))*1E6  
  date_tdb = date_tt + timedelta(seconds=int(dt_sec_tdb),microseconds=dt_msec_tdb)

  t0 = 2443144.5

  dt_sec_tcb  = 1.55051976772E-8*(jd_tai-t0)*86400.
  dt_msec_tcb = (dt_sec_tcb-int(dt_sec_tcb))*1E6 
  date_tcb = date_tt + timedelta(seconds=int(dt_sec_tcb),microseconds=dt_msec_tcb)

  lg = 6.9692903E-10
  
  dt_sec_tcg  =  lg/(1-lg)*(jd_tt-t0)*86400.
  dt_msec_tcg = (dt_sec_tcg-int(dt_sec_tcg))*1E6 
  date_tcg = date_tt + timedelta(seconds=int(dt_sec_tcg),microseconds=dt_msec_tcg)
  
  return date_ut1,date_tai,date_gps,date_tdb,date_tcb,date_tcg,date_tt,t_tt
