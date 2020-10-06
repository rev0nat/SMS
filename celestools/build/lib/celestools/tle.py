#!/usr/bin/env python3
# Licensed 3-clause BSD style license - see LICENSE.rst
# -*-coding:Utf-8 -*

"""
  tle.py
  ######

  * Author: Alexis Petit, PhD Student, Namur University
  * Date: 2017/02/23
  * It reads the TLE format.
"""


#Packages


import pandas as pd
import numpy as np
import sys
import copy
from datetime import date,datetime,timedelta
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv


#Internal packages


from celestools.constants import earth_eq_radius,earth_mu
from celestools.coordinates import cart_2_kepl
from celestools.coordinates import df_cart_2_kepl
from celestools.frame import convert_frame_cart_orekit


#Code


__all__ = [
    'TLE_data'
]


class TLE_data():
  """ A class reading the TLE format.
  """

  def __init__(self,data_tle,format_tle='2_lines',method='raw',coord='kepl',frame='teme',date=None,unit='rad',delta=0):
    """ Read TLE format and compute orbital elements 
    and informations about the RSO.

    INPUTS
    ------

    data_tle: list of string
      Contains one set or more TLE lines.
    format_tle: string
      Give the TLE format (2 or 3 lines).
    method: string
      Give the method to compute orbital elements.
    date: datetime
      Date of the coordinates.

    """
    
    if (format_tle=='2_lines') or (format_tle=='2_lines_serie'):
      self.name         = 'None'
      self.tle          = data_tle
      self.id_year      = data_tle[0][9:11]
      self.id_launch    = data_tle[0][11:14]
      self.id_piece     = data_tle[0][14:15]
      self.idsat        = int(data_tle[1][2:7])
      idcospar_part1    = data_tle[0][9:14]
      idcospar_part2    = data_tle[0][14:18].strip()
      self.id_cospar    = idcospar_part1+'-'+idcospar_part2
      #try: 
      self.observations = self.read(format_tle,method,date,coord,frame,unit,delta)
      #except:
      #  print("Error with TLE reader")
      #  self.observations = pd.DataFrame(columns=["date","x[m]","y[m]","z[m]","vx[m/s]","vy[m/s]","vz[m/s]","bstar[m2/kg]","bc[m2/kg]"])
        
    elif (format_tle=='name_plus_2_lines'):
      self.name         = data_tle[0][2:].strip()
      self.tle          = data_tle[1:]
      self.id_year      = data_tle[1][9:11]
      self.id_launch    = data_tle[1][11:14]
      self.id_piece     = data_tle[1][14:15]
      self.idsat        = int(data_tle[2][2:7])
      idcospar_part1    = data_tle[1][9:14]
      idcospar_part2    = data_tle[1][14:18].strip()
      self.id_cospar    = idcospar_part1+'-'+idcospar_part2
      format_tle = '2_lines_serie'
      try: 
        self.observations = self.read(format_tle,method,date,coord,frame,unit,delta)
      except:
        print("Error with TLE reader")
        self.observations = pd.DataFrame(columns=["date","x[m]","y[m]","z[m]","vx[m/s]","vy[m/s]","vz[m/s]","bstar[m2/kg]","bc[m2/kg]"])
      
    else:
      print('Error: bad format')

      
  def read(self,format_tle,method,date,coord,frame,unit,delta):
    """ Read the TLE lines and extract the date, the orbital elements
    and the bstar parameter.

    INPUT
    -----

    format_tle: string
      Format of the TLE data.
    method: string
      Method to transform TLE data.
    date: datetime
      Date of the coordinates.

    RETURN
    ------

    observations: dataframe object
      Contains all observations.

    """

    observations = []

    if format_tle=='2_lines':

        line1 = self.tle[0].strip()
        line2 = self.tle[1].strip()        
        observation = self.convert_one_tle(line1,line2,method,date,delta)
        if observation != None:
          observations.append(observation)
          
    elif format_tle=='2_lines_serie':
      
      for k in range(0,len(self.tle),2):
        line1 = self.tle[k].strip()
        line2 = self.tle[k+1].strip()
        observation = self.convert_one_tle(line1,line2,method,date,delta)
        if observation != None:
          observations.append(observation)

    if method=='sgp4':
      headers = ["date","x[m]","y[m]","z[m]","vx[m/s]","vy[m/s]","vz[m/s]","bstar[m2/kg]","bc[m2/kg]"]    
      observations = pd.DataFrame(observations,columns=headers)
    elif method=='convert':
      headers = ["date","a[m]","e","i[rad]","raan[rad]","omega[rad]","ma[rad]","bstar[m2/kg]","bc[m2/kg]"]        
      observations = pd.DataFrame(observations,columns=headers)
        
    if observations.empty:
      if coord=='kepl':
        if unit=='rad':
          headers = ["date","a[m]","e","i[rad]","raan[rad]","omega[rad]","ma[rad]","bstar[m2/kg]","bc[m2/kg]"]    
        elif unit=='deg':
          headers = ["date","a[m]","e","i[deg]","raan[deg]","omega[deg]","ma[deg]","bstar[m2/kg]","bc[m2/kg]"]    
        observations = pd.DataFrame(observations,columns=headers)
        
      return observations
    
    if frame=='gcrf':
      new_observations = convert_frame_cart_orekit(observations,'teme','gcrf')
    elif frame=='itrf':
      new_observations = convert_frame_cart_orekit(observations,'teme','itrf')
    elif frame=='teme':
      new_observations = copy.deepcopy(observations)
    else:
      print('Error: bad frame')
      sys.exit()
      
    #print(new_observations)
    if coord=='kepl' and method=='convert':
      if unit=='deg':
        new_observations['i[deg]'] = new_observations['i[rad]']*180/np.pi
        new_observations['raan[deg]'] = new_observations['raan[rad]']*180/np.pi
        new_observations['omega[deg]'] = new_observations['omega[rad]']*180/np.pi
        new_observations['ma[deg]'] = new_observations['ma[rad]']*180/np.pi
        new_observations = new_observations.drop(['ma[rad]','omega[rad]','raan[rad]','i[rad]'],axis=1)
    elif coord=='cart' and method=='convert':
      new_observations = df_kepl_2_cart(new_observations,unit='rad')     
    elif coord=='kepl':
      new_observations = df_cart_2_kepl(new_observations,unit=unit)
    elif coord=='cart':
      pass
    else:
      print('Error: bad type of coordinates')
      sys.exit()

    new_observations["bstar[m2/kg]"] = observations["bstar[m2/kg]"].iloc[0]
    new_observations["bc[m2/kg]"] = observations["bc[m2/kg]"].iloc[0]
    observations = copy.deepcopy(new_observations)
    
    if delta != 0:
      observations['delta_epoch[min]'] = [delta]*len(observations)

    return observations
  

  def convert_one_tle(self,line1,line2,method,date,delta):
    """ Convert one pseudo-observation TLE to orbital elements.

    INPUTS
    ------

    line1: string
      First TLE line.
    line2: string 
      Second TLE line.
    method: string
      Method to transform TLE data.
    date_coord: datetime
      Date of the coordinates

    RETURN
    ------

    observation: list
      Contain the date, orbital elements, ballistic coefficients.

    """

    try:

      if method=='convert':
        
        # Compute the date 
        if int(line1[18:20])<50:
          year = 2000 + int(line1[18:20])
        else:
          year = 1900 + int(line1[18:20])
        doy = float(line1[20:32]) - 1
        date = datetime(year,1,1) + timedelta(days=doy)
        
        # Compute the orbital elements
        n = float(line2[52:63])*2*np.pi/86400. #(rad per sec)
        sma = (earth_mu/n**2)**(1./3.)
        ecc = float('0.'+line2[26:33])
        inc = float(line2[8:16])*np.pi/180.
        raan = float(line2[17:25])*np.pi/180.
        arg = float(line2[34:42])*np.pi/180.
        mea = float(line2[43:51])*np.pi/180.
        
        # Compute the BSTAR parameter
        bstar = float('0.'+line1[54:59])*10**float(line1[59:61])#*100
        bc = 12.741621*bstar

        # Add the data in a list
        observation = [date,sma,ecc,inc,raan,arg,mea,bstar,bc]
        
        return observation

      elif method=='sgp4':

        satellite = twoline2rv(line1,line2,wgs72)
        
        if type(date)==type(None):
          date = satellite.epoch
          if delta!=0:
            date += delta
        else:
          date_tle = satellite.epoch
          if date<date_tle:
            print('Error with the date:',date_tle)
            print(line1)
            print(line2)
            return None
          elif (date-date_tle).days>10:
            print('TLE too old (days>10):',date_tle)
            print(line1)
            print(line2)
            return None
            
        yy = int(date.strftime('%Y'))
        mm = int(date.strftime('%m'))
        dd = int(date.strftime('%d'))
        hh = int(date.strftime('%H'))
        mi = int(date.strftime('%M'))
        ss = float(date.strftime('%S.%f'))       

        # Compute the BSTAR parameter
        bstar = float('0.'+line1[54:59])*10**float(line1[59:61])#*100
        bc = 12.741621*bstar

        # Coordinate
        position,velocity = satellite.propagate(yy,mm,dd,hh,mi,ss)
        posvel = list(position+velocity)
        posvel = np.array([float(item)*1000. for item in posvel])
        if np.isnan(posvel[0]):
          print('Error: problem with SGP4')
          print(line1)
          print(line2)
          return None
        #if coord=='kepl':
        #  kepl = cart_2_kepl(posvel,earth_mu,unit=unit)
        #  sma = kepl[0]
        #  ecc = kepl[1]
        #  inc = kepl[2]
        #  raan = kepl[3]
        #  arg = kepl[4]
        #  mea = kepl[5]
        #  observation = [date,sma,ecc,inc,raan,arg,mea,bstar,bc]
        #elif coord=='cart':
        observation = [date,posvel[0],posvel[1],posvel[2],posvel[3],posvel[4],posvel[5],bstar,bc]
        #else:
        #  print('Error: bad type of coordinates')
        #  sys.exit()          
          
        return observation
      
      else:

        print('Error with the method to compute orbital elements from TLE data')
        
    except ValueError:

      print('Error with the TLE data')
      
      return None
  

  def plot_orbital_evolution(self,save=0):
    """ Plot the orbital elements (a,e,i,RAAN,Omega,ME)
    """

    xticks = [x.strftime('%Y\n%m/%d') for x in self.observations["Date"]]
    #print xticks
    point = 'ro'
    
    fig = plt.figure(figsize=(15,12))
    fig.suptitle('ID NORAD : '+str(self.idsat),fontsize=22)
    ax1 = fig.add_subplot(3,2,1)
    ax1.grid(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylabel('a [km]',fontsize=14)
    ax1.plot(self.observations["date"],self.observations["a[m]"]/1000.,point)
    ax1.plot(self.observations["date"],self.observations["a[m]"]/1000.)
    ax1.set_xticks(ax1.get_xticks()[::3])
    ax2 = fig.add_subplot(3,2,2)
    ax2.grid(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylabel('e',fontsize=14)
    ax2.plot(self.observations["date"],self.observations["e"],point)
    ax2.plot(self.observations["date"],self.observations["e"])
    ax2.set_xticks(ax2.get_xticks()[::3])
    ax3 = fig.add_subplot(3,2,3)
    ax3.grid(True)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_ylabel('i [deg]',fontsize=14)
    ax3.plot(self.observations["date"],self.observations["i[deg]"],point)
    ax3.plot(self.observations["date"],self.observations["i[deg]"])
    ax3.set_xticks(ax3.get_xticks()[::3])
    ax4 = fig.add_subplot(3,2,4)
    ax4.grid(True)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_ylabel('$\Omega$ [deg]',fontsize=14)
    ax4.plot(self.observations["date"],self.observations["raan[deg]"],point)
    ax4.plot(self.observations["date"],self.observations["raan[deg]"])
    ax4.set_ylim([0,360])
    ax4.set_xticks(ax4.get_xticks()[::3])
    ax5 = fig.add_subplot(3,2,5)
    ax5.grid(True)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.set_ylabel('$\omega$ [deg]',fontsize=14)
    ax5.set_xlabel('Time',fontsize=14)
    ax5.plot(self.observations["date"],self.observations["omega[deg]"],point)
    ax5.plot(self.observations["date"],self.observations["omega[deg]"])    
    ax5.set_ylim([0,360])
    ax5.set_xticks(ax5.get_xticks()[::3])
    ax6 = fig.add_subplot(3,2,6)
    ax6.grid(True)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.set_ylabel('M [deg]',fontsize=14)
    ax6.set_xlabel('Time',fontsize=14)
    ax6.plot(self.observations["date"],self.observations["ma[deg]"],point)
    ax6.plot(self.observations["date"],self.observations["ma[deg]"])
    ax6.set_ylim([0,360])
    ax6.set_xticks(ax6.get_xticks()[::3])
    if save==0:
      plt.show()
    else:
      plt.savefig(str(self.idsat)+'_oe')
      

  def plot_bstar(self):
    """ Plot the BSTAR parameter
    """
    
    plt.title('ID NORAD : '+str(self.idsat),fontsize=14)
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('BC [m2/kg]',fontsize=14)
    plt.plot(self.observations["date"],self.observations["bstar[m2/kg]"])
    plt.show()
      
    



  

  
 
