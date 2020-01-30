# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Axom Project Developers. See the top-level COPYRIGHT file for details.
#
# LLNL-CODE-802401
#
# SPDX-License-Identifier: (BSD-3-Clause)
#------------------------------------------------------------------------------
# Copson_Expansion_Solution project
#------------------------------------------------------------------------------

import numpy as np
from CopsonFuncs import *

alpha = -1.0/3.0
beta = -17./9.
c = 1.0
h = 1.0

r =1.5


rvals = np.linspace(-3., 3., num=21)

epsilons = np.logspace(-8, 0, num=9)

# test factored solutions versus originals
# this shows where the numerics drive the original expressions to bad results
for eps in epsilons:
	s = eps - r
 	poly1 = r - s
 	poly3 = 4.*r**3 - 3.*r*r*s + 2.*r*s*s - s**3
 	poly2 = r*r - (4./3.)*r*s + s*s
	
 	print eps, "t ", (2.0*(evalPhi(r, alpha, beta, c, h) - evalPhi(s, alpha, beta, c, h)) - (eps)*(evalPhip(r, alpha, beta, c, h) - evalPhip(s, alpha, beta, c, h)))/eps**3, \
 	-(1./6.)*evalPhi3p(r, alpha, beta, c, h) + (1./12.)*evalPhi4p(r, alpha, beta, c, h)*eps - (1./40.)*evalPhi4p(r, alpha, beta, c, h)*eps*eps, -(1./6.)*evalPhi3p(r, alpha, beta, c, h), \
 	( (4./3.)*(poly1 - (8./15.)*poly2) - (4./9.)*alpha*(1. - 2.*poly1 + (4./5.)*poly2) + (4./9.)*beta*(poly1 - (4./5.)*poly2) )
 	poly2 = 3.*r*r - 2.*r*s + s*s
 	print eps, "x ", evalPhip(r,alpha,beta,c,h)/(r + s + 1e-20) - (evalPhi(r,alpha,beta,c,h) - evalPhi(s,alpha,beta,c,h))/(r + s + 1e-20)**2,\
 	0.5*evalPhi2p(r,alpha,beta,c,h) - evalPhi3p(r,alpha,beta,c,h)*eps/6.0 +  evalPhi4p(r,alpha,beta,c,h)*eps/24.0, 0.5*evalPhi2p(r,alpha,beta,c,h),\
 	( 1 - (2./3.)*poly2 + (32./135.)*poly3 + (4./9.)*alpha*(r + poly1 - poly2 + (4./15.)*poly3) - (2./9.)*beta*(poly2 - (8./15.)*poly3) )
 	print eps, "t reg2 ", evalPhi3p(s,alpha,beta,c,h)/6.0,  ( (4./3.)*(poly1 - (8./15.)*poly2) - (4./9.)*alpha*(1. - 2.*poly1 + (4./5.)*poly2) + (4./9.)*beta*(poly1 - (4./5.)*poly2) ), \
 	 -(evalPhip(1.5*c,alpha,beta,c,h) - evalPhip(s,alpha,beta,c,h))/(1.5*c + s + 1e-20)**2 + 2.0*(evalPhi(1.5*c,alpha,beta,c,h) - evalPhi(s,alpha,beta,c,h))/(1.5*c + s + 1e-20)**3

	poly1 = -128.*s**2 + 66.*s + 27.
	poly2 = -32.*s**2 - 6.*s + 3.
	poly3 = 128.*s**3 + 36.*s*s - 36.*s + 27

	print eps, "reg 2 x,t ", evalPhip(s,alpha,beta,c,h)/(1.5*c + s + 1e-20) + (evalPhi(1.5*c,alpha,beta,c,h) - evalPhi(s,alpha,beta,c,h))/(1.5*c + s + 1e-20)**2,\
	(1.5*c + s + 1e-20)*( poly1 + 2.*alpha*poly2)/135. - beta*poly3/270.

	poly1 = 1.5 - s
	poly2 = 9.0/4.0 - 2.0*s + s*s
	poly3 = 27.0/2.0 - (27.0/4.0)*s + 3.0*s*s - s**3
	print eps, - (evalPhip(1.5*c,alpha,beta,c,h) - evalPhip(s,alpha,beta,c,h))/(1.5*c + s + 1e-20)**2 + 2.0*(evalPhi(1.5*c,alpha,beta,c,h) - evalPhi(s,alpha,beta,c,h))/(1.5*c + s + 1e-20)**3,\
	( (4./3.)*(poly1 - (8./15.)*poly2) - (4./9.)*alpha*(1. - 2.*poly1 + (4./5.)*poly2) + (4./9.)*beta*(poly1 - (4./5.)*poly2) )

# quit()
# now test solution of r,s from x,t near the free surface
t = 0.5
xvals = np.linspace(1.110, 1.1112, num=101)

time1 = (-2.0*alpha/3.0)*h/c
# time region 2 reaches free surface
time2 = -2.0*h*(1 + alpha/3.0 + beta)/c

# find minimum possible s value from free surface
# if time after when region 2 hits the surface limit is -1.5*c
if t >= time2:
   Smin = -1.5*c
# if time before when r=0 characteristic hits the surface limit is 0.
elif t <= time1:
   Smin = 0.0 
else:
   A = 8.0*(2 + alpha + beta)/9.0
   B = -2.0*(3.0 + 2.0*alpha + beta)/3.0
   C = alpha/3.0 + 0.5*c*t/h
   disc = 0.0
   if abs(A) < 1e-8:
      Smin = C/B*(1 + A*C/B**2)
   else:
      disc = B*B - 4.0*A*C
      if disc < 0.0:
         print "error finding surface value of r"
         Smin = -1.5*c
      else:
         Smin = -(-B - math.sqrt(disc))/(2.0*A)
# find x position of free surface at time t (r1 = -Smin)
x1 = -2.0*Smin*t + 0.5*evalPhi2p(-Smin,alpha,beta,c,h)

# find s value at region1/ region 2 boundary at time t
if t > time2:
   s2 = -1.5*c
else:
   s2 = scipy.optimize.brentq(SXT2, Smin, 1.5*c, args=(t, alpha, beta, c, h)) 
# Calculate x value at region 1/region 2 boundary
x2 = -(4.0/3.0*s2 - c)*t + evalPhip(s2,alpha,beta,c,h)/(1.5*c + s2 + 1e-20) + (evalPhi(1.5*c,alpha,beta,c,h) - evalPhi(s2,alpha,beta,c,h))/(1.5*c + s2 + 1e-20 )**2

# Calculate x value at region 2/region 3 boundary
x3 = -c*t
print "time1, time2 ", time1, time2
print "Smin ", Smin
print " x1, x2, x3 ", x1, x2, x3

for x in xvals:
	print "Final x, t ", x, t, " r, s ", xt2rs(x, t, alpha, beta, c, h)
r,s = (0.08557694500887883, -0.65020885877273915)
x = 1.0876
print RS_fromXTreg1((r, s), x, t, alpha, beta, c, h)
# x, t  1.0876 0.5  r, s  (0.08557694500887883, -0.65020885877273915)
xt2rs(x, t, alpha, beta, c, h)
