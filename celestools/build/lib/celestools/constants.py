#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*-coding:Utf-8 -*

"""
  contants.py
  Author: Alexis Petit, PhD Student, Namur University
  Date: 2016 09 05
  Important constants (values given by Montenbruck, 2005)
"""

# Time

mjd = 51544.5                  #IAU 1976
tt_less_tai = 32.184           #[sec] IAU 1991
gps_less_tai = -19             #[sec]

# Universal

c = 299792458                  #[m/s] IAU 1976
g_constant = 6.673E-20         #[km3/(kg.s2)] (Cohen & Taylor, 1987)

# Earth

earth_k = 0.01720209895        #IAU 1939
earth_mu = 0.3986004415E15     #[km3/s2] JGM-3
earth_J2 = 1.0826E-3           #JGM-3
earth_eq_radius = 6378137      #[m] WGS-84 (NIMA 1997)
earth_f = 1/298.257223563      #WGS-84 (NIMA 1997)
earth_omega = 0.7292115E-4     #Moritz 1980

#e_earth = 0.006694385
e_earth = 0.081819221456       #value p.132

# Sun

sun_mu = 1.32712440018E11     #[km3/s2] DE405 (Standish, 1998)
au = 149.597870691E9          #[m] DE405 (Standish, 1998)
sun_eq_radius = 6.94E8        #[m] (Seidelmann, 1992)
sun_p = 4.560E-6              #[N/m2] IERS 1996 (McCarthy, 1996)

# Moon

moon_mu = 4902.801             #[km3/s2] DE405 (Standish, 1998)
moon_sma = 384.4E6             #[m] (Seidelmann, 1992)
moon_eq_radius = 1.738E6       #[m] (Seidelmann, 1992)

