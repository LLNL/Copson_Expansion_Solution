# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Axom Project Developers. See the top-level COPYRIGHT file for details.
#
# LLNL-CODE-802401
#
# SPDX-License-Identifier: (BSD-3-Clause)
#------------------------------------------------------------------------------
# Copson_Expansion_Solution project
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import (AutoMinorLocator)
import numpy as np
import scipy.optimize
import scipy.integrate
import math
from CopsonFuncs import *

def makePlot(alpha, beta, FinalTime, filename):
   """
   make a plot of the characteristics for a given alpha, beta
   also make a streamlines plot.
   """
   N = 6     # number of characteristics in surface layer
   N2 = 100  # number of points in lines.
   #FinalTime = 1.5
   LowerX = -2.0

   # surface, r=0 characteristic
   rvals = np.linspace(0.0, 1.5, num=N2+1)

   # get free surface position
   bndy_time = evalPhi3p(rvals, alpha, beta, 1.0, 1.0)/(-4.0)
   bndy_xvals = 2*rvals*bndy_time + 0.5*evalPhi2p(rvals, alpha, beta, 1.0, 1.0)

   # extend surface into region 2 with constant slope
   if bndy_time[-1] > FinalTime:
      FinalTime = bndy_time[-1] + 1.0
   slope = 3.
   x2 = bndy_xvals[-1] + slope*(FinalTime - bndy_time[-1])

   bndy_time = np.append(bndy_time, FinalTime)
   bndy_xvals = np.append(bndy_xvals, x2)
   
   # start the plot
   plt.figure(num=1, figsize=(5,3.75)) #(6.0, 4.5)
   plt.figure(num=2, figsize=(5,3.75))

   plt.figure(num=1)
   # put at end so it is on top
   # plt.plot(bndy_time, bndy_xvals, color='g', lw=2.0)

   # initial values of r to start characteristics at
   rvals = np.linspace(0.0, 1.5, num=N+1)
   #
   for r in rvals[1:]:
      # At t=0, u = 0 so r = s
      # values of s to get points for. These extend to the free surface where c = 0, so s = -r. So s varies from r -> -r
      svals = np.linspace(r, -r, num=N2)

      time = 1.5*reg1_t(r, svals, alpha, beta, 1.0, 1.0)

      xvals = reg1_r_characteristic(r, svals, alpha, beta, 1.0, 1.0) + (2.0/3.0)*( 2*r - svals )*time

      width = 0.5
      if r == 1.5:
         width = 2.0
      plt.plot(time, xvals, color='b', lw=width)
      
      #  s characteristics starting in surface layer. Start at same value of r since r=s at t=0
      #  These characteristics go to the boundary with region 2 where r = 1.5
      s = r
      rvals = np.linspace(r, 1.5, num=N2)

      stime = 1.5*reg1_t(rvals, s, alpha, beta, 1.0, 1.0)
      sxvals = reg1_s_characteristic(rvals, s, alpha, beta, 1.0, 1.0) - (2.0/3.0)*( 2*s - rvals )*stime
      
      # extend s characteristic in region 2 with straight line since r is constant in region 2
      rbnd = 1.5
      slope = -4.*s/3. + 1.0
      x2 = sxvals[-1] + slope*(FinalTime - stime[-1])
      
      # s characteristics starting in surface layer
      stime = np.append(stime, FinalTime)
      sxvals = np.append(sxvals,x2)
      LowerX = x2

      # Lower X will get updated until we get to the last s characteristci in the surface layer

      width = 0.5
      if s == 1.5:
         width = 2.0
      if s > 0:
         plt.plot(stime, sxvals, color='r', lw=width)
      elif s == 0:
         plt.plot(stime, sxvals, color='g', lw=1.0)  # This is the surface, t = r = s = 0, x = h = 1.0
      else:
         plt.plot(stime, sxvals, color='r', lw=width)

   # s characteristics starting at surface as reflection of an r characteristic
   rvals = np.linspace(0.0, 1.5, num=N+1)
   for i in range(5): #range(N-1):
      s = -rvals[i]
      rvallist = np.linspace(rvals[i], 1.5, num=N2)

      stime = 1.5*reg1_t(rvallist, s, alpha, beta, 1.0, 1.0)
      sxvals = reg1_s_characteristic(rvallist, s, alpha, beta, 1.0, 1.0) - (2.0/3.0)*( 2*s - rvallist )*stime

      slope = -4.*s/3. + 1.0
      x2 = sxvals[-1] + slope*(FinalTime - stime[-1])
      
      stime = np.append(stime, FinalTime)
      sxvals = np.append(sxvals,x2)

      width = 0.5

      if s > 0:
         plt.plot(stime, sxvals, color='r', lw=width)
      elif s == 0:
         plt.plot(stime, sxvals, color='r', lw=width)
      else:
         plt.plot(stime, sxvals, color='r', lw=width)

   # add s characteristics in stationary gas
   xinit = np.linspace(0, LowerX, num=N)
   for i in range(1,N):
      Tfinal = FinalTime
      Xfinal = xinit[i] - FinalTime
      if Xfinal < LowerX:
         Tfinal = -(LowerX - xinit[i])
         Xfinal = xinit[i] - Tfinal
      xvals = np.array([xinit[i], Xfinal])
      tvals = np.array([0., Tfinal])
      width = 0.5
      plt.plot(tvals, xvals, color='r', lw=width)

   
   # add r characteristics in the second layer, they start in the stationary gas
   # first in unmoving material
   xinit = np.linspace(0, 2*LowerX, num=3*N)
   for i in range(1,3*N - 1):
      # first do the part in the stationary gas, a straight line
      Xfinal = 0.5*xinit[i]
      Tfinal = -0.5*xinit[i]
      xvals = np.array([xinit[i], Xfinal])
      tvals = np.array([0., Tfinal])
      width = 0.5
      plt.plot(tvals, xvals, color='b', lw=width)
      # integrate ODE from (Xfinal, Tfinal) out FinalTime
      times = np.linspace(Tfinal, FinalTime, num=N)
      y0 = np.array([Xfinal])

      xvalues = scipy.integrate.odeint(RHS, y0, times, args=(alpha, beta, 1.0,1.0))

      plt.plot(times, xvalues, color='b', lw=width)

   plt.plot(bndy_time, bndy_xvals, color='g', lw=2.0) # add surface at end so it is on top

   plt.xlabel(r'$t\ (a/h)$')
   plt.ylabel(r'$x/h$')
   plt.xlim((0.,FinalTime))
   plt.ylim((LowerX,bndy_xvals[-1]))
   fractionA = False
   fractionB = False
   if (abs(-3*alpha - int(-3*alpha+0.5)) < 0.001):
      fractionA = True
      alphaDenom = 3
      alphaNum = int(-3*alpha+0.5)
   elif (abs(-9*alpha - int(-9*alpha+0.5)) < 0.001):
      fractionA = True
      alphaDenom = 9
      alphaNum = int(-9*alpha+0.5)
   
   if (abs(-3*beta - int(-3*beta+0.5)) < 0.001):
      fractionB = True
      betaDenom = 3
      betaNum = int(-3*beta+0.5)
   elif (abs(-9*beta - int(-9*beta+0.5)) < 0.001):
      fractionB = True
      betaDenom = 9
      betaNum = int(-9*beta+0.5)

   if fractionA and fractionB:
      if alphaNum == 0:
         plt.title( r'$\alpha = %d$, $\beta = -\frac{%d}{%d}$' % (alphaNum, betaNum, betaDenom))
      else:
         plt.title( r'$\alpha = -\frac{%d}{%d}$, $\beta = -\frac{%d}{%d}$' % (alphaNum, alphaDenom, betaNum, betaDenom))
   elif fractionA:
      if alphaNum == 0:
         plt.title( r'$\alpha = %d$, $\beta = $ %0.3f' % (alphaNum, beta))
      else:
         plt.title( r'$\alpha = -\frac{%d}{%d}$, $\beta = $ %0.3f' % (alphaNum, alphaDenom, beta))
   elif fractionB:
      plt.title( r'$\alpha = $ %0.3f, $\beta = -\frac{%d}{%d}$' % (alpha, betaNum, betaDenom))
   else:
      plt.title( r'$\alpha = $%0.3f, $\beta =$ %0.3f' % (alpha, beta))
   # plt.title( r'Characteristics ($\alpha = $%0.3f, $\beta =$ %0.3f)' % (alpha, beta))
   ax = plt.gca()
   ttl = ax.title
   ttl.set_position([.5, 1.025])
   ax.xaxis.set_minor_locator(AutoMinorLocator())
   ax.yaxis.set_minor_locator(AutoMinorLocator())

   #plt.grid(True)
   plt.tight_layout()
   plt.savefig(filename, format="pdf", dpi=1200)
   
   # Do streamlines starting from 1 -> 0, calculate starting point from r values like characteristics
   plt.figure(2)
   rvals = np.linspace(0.0, 1.5, num=N+1)

   plt.plot(bndy_time, bndy_xvals, color='g', lw=2.0) # add surface as a streamline
   for r in rvals[1:]:
      Xstart = reg1_r_characteristic(r, r, alpha, beta, 1.0, 1.0)
      Stream_times = np.linspace(0.0, FinalTime, num=N2)
      y0 = np.array([Xstart])

      Stream_xvals = scipy.integrate.odeint(velocity, y0, Stream_times, args=(alpha, beta, 1.0,1.0))
      plt.plot(Stream_times, Stream_xvals, lw=width)

   xinit = np.linspace(0, LowerX, num=3*N/2)
   for i in range(1,3*N/2):
      Xstart = xinit[i]
      Stream_times = np.linspace(0.0, FinalTime, num=N2)
      y0 = np.array([Xstart])

      Stream_xvals = scipy.integrate.odeint(velocity, y0, Stream_times, args=(alpha, beta, 1.0,1.0))
      plt.figure(2)
      plt.plot(Stream_times, Stream_xvals, lw=width)
   plt.xlabel(r'$t\ (a/h)$')
   plt.ylabel(r'$x/h$')
   plt.xlim((0.,FinalTime))
   plt.ylim((LowerX,bndy_xvals[-1]))
   if fractionA and fractionB:
      if alphaNum == 0:
         plt.title( r'Streamlines ($\alpha = %d$, $\beta = -\frac{%d}{%d}$)' % (alphaNum, betaNum, betaDenom))
      else:
         plt.title( r'Streamlines ($\alpha = -\frac{%d}{%d}$, $\beta = -\frac{%d}{%d}$)' % (alphaNum, alphaDenom, betaNum, betaDenom))
   elif fractionA:
      if alphaNum == 0:
         plt.title( r'Streamlines ($\alpha = %d$, $\beta = $ %0.3f)' % (alphaNum, beta))
      else:
         plt.title( r'Streamlines ($\alpha = -\frac{%d}{%d}$, $\beta = $ %0.3f)' % (alphaNum, alphaDenom, beta))
   elif fractionB:
      plt.title( r'Streamlines ($\alpha = $ %0.3f, $\beta = -\frac{%d}{%d}$)' % (alpha, betaNum, betaDenom))
   else:
      plt.title( r'Streamlines ($\alpha = $%0.3f, $\beta =$ %0.3f)' % (alpha, beta))
   # plt.title( r'Streamlines ($\alpha = $%0.3f, $\beta =$ %0.3f)' % (alpha, beta))
   ax = plt.gca()
   ttl = ax.title
   ttl.set_position([.5, 1.025])
   ax.xaxis.set_minor_locator(AutoMinorLocator())
   ax.yaxis.set_minor_locator(AutoMinorLocator())
   plt.tight_layout()
   name = filename[0:-4]+"streamline"+filename[-4:]
   plt.savefig(name, format="pdf", dpi=1200)

   plt.show()

alpha =  -1.0/3.0
beta  = -17.0/9.0

makePlot(alpha,beta, 2.0, "Char_test_A0.333_B1.889.pdf")
# quit()
alpha = -0.
beta = -1.7

makePlot(alpha,beta, 3.0, "Char_A0.0_B1.7.pdf")

alpha = -0.0
beta = -2.95

makePlot(alpha,beta, 3.0, "Char_A0.0_B2.95.pdf")

alpha = -0.95
beta = -1.05

makePlot(alpha,beta, 3.0, "Char_A0.95_B1.0.pdf")
quit()
