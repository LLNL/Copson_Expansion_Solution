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
import scipy.optimize
import math

# c is the sound speed, normally set to 1.0
# h is the thickness of the surface layer, normally set to 1.0

# alpha and beta are the slopes of the x(r) curve

def evalPhi(r, alpha, beta, c, h):
    """
    evaluate phi(r)
    """
    x = np.abs(r/c)  # This will ensure the function is even
    poly1 = 1.0 + x*x*( -2.0/3.0 + x*32.0/135.0)
    poly2 = 1.0 + x*( -1.0 + x*4.0/15.0)
    poly3 = 1.0 - x*8.0/15.0

    alpha49 = (4.0/9.0)*alpha
    beta29  = (2.0/9.0)*beta
    
    # result = h*r*r* ( poly1 + (4.0/9.0)*x*poly2*alpha - (2.0/9.0)*x*x*poly3*beta )
    result = h*r*r* ( 1.0 + x*( alpha49 + x*( -2.0/3.0 - alpha49 - beta29 + x*(4.0/15.0)*( 8.0/9.0 + alpha49 + 2.0*beta29 ) ) ) )
    return result

def evalPhip(r, alpha, beta, c, h):
    """
    evaluate first derivative of phi(r)
    """
    x = np.abs(r/c)  # This will ensure the function is odd
    poly1 = 1.0 + x*x*( -(4.0/3.0) + x*16.0/27.0)
    poly2 = 1.0 + x*( -4.0/3.0 + x*4.0/9.0)
    poly3 = 1.0 - x*2.0/3.0
    
    alpha23 = (2.0/3.0)*alpha
    beta49  = (4.0/9.0)*beta

    # result = 2.0*h*r* ( poly1 + (2.0/3.0)*x*poly2*alpha - (4.0/9.0)*x*x*poly3*beta )
    result = 2.0*h*r* ( 1.0 + x*( alpha23 + x*( -4.0/3.0 - (4.0/3.0)*alpha23 - beta49 + x*(2.0/3.0)*( 8.0/9.0 + (2.0/3.0)*alpha23 + beta49 ) ) ) )
    return result

def evalPhi2p(r, alpha, beta, c, h):
    """
    evaluate second derivative of phi(r)
    """
    x = np.abs(r/c)  # This will ensure the function is even
    poly1 = 1.0 + x*x*( -4.0 + x*64.0/27.0)
    poly2 = 1.0 + x*( -2.0 + x*8.0/9.0)
    poly3 = 1.0 - x*8.0/9.0
    
    alpha43 = (4.0/3.0)*alpha
    beta43  = (4.0/3.0)*beta

    # result = 2.0*h* ( poly1 + (4.0/3.0)*x*poly2*alpha - (4.0/3.0)*x*x*poly3*beta )
    result = 2.0*h* ( 1.0 + x*( alpha43 + x*( -4.0 - 2.0*alpha43 - beta43 + x*(8.0/9.0)*( 8.0/3.0 + alpha43 + beta43 ) ) ) )
    return result

def evalPhi3p(r, alpha, beta, c, h):
    """
    evaluate third derivative of phi(r)
    """
    x = np.abs(r/c)  # This will ensure the function is odd
    poly1 = x*(-2.0 + x*16.0/9.0)
    poly2 = 1.0 + x*( -4.0 + x*8.0/3.0)
    poly3 = 1.0 - x*4.0/3.0
    
    alpha13 = (1.0/3.0)*alpha
    beta23  = (2.0/3.0)*beta

    # result = 8.0*(h/c)* ( poly1 + (1.0/3.0)*poly2*alpha - (2.0/3.0)*x*poly3*beta )
    result = 8.0*(h/c)* ( alpha13 + x*( -2.0 - 4.0*alpha13 - beta23 + x*(4.0/3.0)*( 4.0/3.0 + 2.0*alpha13 + beta23 ) ) )
    return result*r/(np.abs(r) + 1e-80)  # This will ensure the function is odd

def evalPhi4p(r, alpha, beta, c, h):
    """
    evaluate fourth derivative of phi(r)
    """
    x = np.abs(r/c)  # This will ensure the function is even
    poly1 =  1.0 - x*16.0/9.0
    poly2 =  1.0 - x*4.0/3.0
    poly3 =  1.0 - x*8.0/3.0
    
    alpha13 = (2.0/3.0)*alpha
    beta13  = (1.0/3.0)*beta

    # result = 8.0*(h/(c*c))* ( poly1 + (1.0/3.0)*poly2*alpha - (1.0/3.0)*poly3*beta )
    result = -16.0*(h/(c*c))* ( poly1 + alpha13*poly2 + beta13*poly3 )
    return result

def evalPhi5p(r, alpha, beta, c, h):
    """
    evaluate fourth derivative of phi(r)
    """
    x = np.abs(r/c)  # This will ensure the function is odd
    result = (128.0/9.0)*(h/(c*c*c))* ( 2.0 + alpha + beta )
    return result*r/(np.abs(r) + 1e-80)

def TimeFirstStage(r, s, alpha, beta, c, h):
    """
    evaluate time in Region 1 from r and s
    """
    # x = r/c
    t = ( 3*( evalPhi(r, alpha, beta, c, h) - evalPhi(s, alpha, beta, c, h) ) - 1.5*(r + s)*( evalPhip(r, alpha, beta, c, h) - evalPhip(s, alpha, beta, c, h) ) )/((r + s)**3 + 1.e-80)
    return t

def reg2_xtRHS(s, alpha, beta, c, h):
    """
    right hand side of x + ((4/3)s - c)t eqn
    """
    ss = s/c
    if s == -1.5*c:
        value = evalPhi2p(s,alpha,beta,c,h)/2.0
    elif s < 0.:
        # when s < 0 the expression can be factored and you avoid the
        # difference of nearly equal numbers and dividing by a small number
        poly1 = -128.*ss**2 + 66.*ss + 27.
        poly2 = -32.*ss**2 - 6.*ss + 3.
        poly3 = 128.*ss**3 + 36.*ss*ss - 36.*ss + 27
        value = (1.5*c + ss + 1e-20)*( poly1 + 2.*alpha*poly2)/135. - beta*poly3/270.
    else:
        value = evalPhip(s,alpha,beta,c,h)/(1.5*c + s + 1e-20) + (evalPhi(1.5*c,alpha,beta,c,h) - evalPhi(s,alpha,beta,c,h))/(1.5*c + s + 1e-20)**2
    return value

def S_from_XT_reg2(s, x, t, alpha, beta, c, h):
    """
    find s from x,t in region 2
    """
    value = x - c*t + 4./3.*s*t - reg2_xtRHS(s,alpha,beta,c,h)

    return value

def reg1_t(r, s, alpha, beta, c, h):
    """
    evaluate (2/3)t in region 1
    """
    # when s < 0 the expression can be factored and you avoid the
    # difference of nearly equal numbers and dividing by a small number
    rr = r/c
    ss = s/c

    poly1 = rr - ss
    poly2 = rr*rr - (4.0/3.0)*rr*ss + ss*ss
    poly3 = 4.0*rr**3 - 3.0*rr*rr*ss + 2.0*rr*ss*ss - ss**3
    value = np.where(s <= 0., (4./3.)*(poly1 - (8./15.)*poly2) - (4./9.)*alpha*(1. - 2.*poly1 + (4./5.)*poly2) + (4./9.)*beta*(poly1 - (4./5.)*poly2), 
                            (2.0*(evalPhi(r, alpha, beta, c, h) - evalPhi(s, alpha, beta, c, h)) - (r + s)*(evalPhip(r, alpha, beta, c, h) - evalPhip(s, alpha, beta, c, h)))/(r + s + 1e-20)**3)

    return value

def reg1_r_characteristic(r, s, alpha, beta, c, h):
    """
    evaluate x - ((4/3)r - (2/3)s)t in region 1
    """
    # when s < 0 the expression can be factored and you avoid the
    # difference of nearly equal numbers and dividing by a small number
    rr = r/c
    ss = s/c

    poly1 = 2.0*rr - ss
    poly2 = 3.0*rr*rr - 2.0*rr*ss + ss*ss
    poly3 = 4.0*rr**3 - 3.0*rr*rr*ss + 2.0*rr*ss*ss - ss**3
    value = np.where(s <= 0., 1.0 - (2./3.)*poly2 + (32./135.)*poly3 + (4./9.)*alpha*(poly1 - poly2 + (4./15.)*poly3) - (2./9.)*beta*(poly2 - (8./15.)*poly3),
                              evalPhip(r,alpha,beta,c,h)/(r + s + 1e-20) - (evalPhi(r,alpha,beta,c,h) - evalPhi(s,alpha,beta,c,h))/(r + s + 1e-20)**2 )
    return value

def reg1_s_characteristic(r, s, alpha, beta, c, h):
    """
    evaluate x + ((4/3)s - (2/3)r)t in region 1
    """
    # when s < 0 the expression can be factored and you avoid the
    # difference of nearly equal numbers and dividing by a small number
    rr = r/c
    ss = s/c

    poly1 = rr - 2.0*ss
    poly2 = rr*rr - 2.0*rr*ss + 3.0*ss*ss
    poly3 = rr**3 - 2.0*rr*rr*ss + 3.0*rr*ss*ss - 4.0*ss**3
    value = np.where( s <= 0., 1.0 - (2./3.)*poly2 + (32./135.)*poly3 + (4./9.)*alpha*(poly1 - poly2 + (4./15.)*poly3) - (2./9.)*beta*(poly2 - (8./15.)*poly3), 
                               evalPhip(s,alpha,beta,c,h)/(r + s + 1e-20) + (evalPhi(r,alpha,beta,c,h) - evalPhi(s,alpha,beta,c,h))/(r + s + 1e-20)**2 )
    return value

def SXT2(s, t, alpha, beta, c, h):
    """
    find s from t at boundary of region 1 and 2
    """
    rhs = reg1_t(1.5*c, s, alpha, beta, c, h)
    value = -2.0*t/3.0 + rhs
    return value

def RS_fromXTreg1((r, s), x, t, alpha, beta, c, h):
    """
    Evaluate equations whose roots will give r and s from x and t
    """
    value1 = x - (4./3.*r - 2./3.*s)*t - reg1_r_characteristic(r, s, alpha, beta, c, h)
    value2 = 2.0*t/3.0 - reg1_t(r, s, alpha, beta, c, h)
    return [value1, value2]

def FreeSurface_S(t, alpha, beta, c, h):
    """
    find the value of s at the free surface for time t
    This is the negative of the r value
    """
    # time free surface starts moving (r = 0 characteristic reaches surface)
    time1 = (-2.0*alpha/3.0)*h/c
    # time region 2 reaches free surface
    time2 = -2.0*h*(1 + alpha/3.0 + beta)/c
    # find minimum possible s value from free surface value
    # if time after when region 2 hits the surface limit is -1.5*c
    if t >= time2:
        Smin = -1.5*c
    # if time before when r=0 characteristic hits the surface limit is 0.
    elif t <= time1:
        Smin = 0.0 
    else:
        # solve the quadratic from -0.25*phi(iii)(r) = t for -r = s
        A = 8.0*(2 + alpha + beta)/9.0
        B = -2.0*(3.0 + 2.0*alpha + beta)/3.0
        C = alpha/3.0 + 0.5*c*t/h
        disc = 0.0
        if abs(A) < 1e-8:
            # use linear approximation when A is small
            Smin = C/B*(1 + A*C/B**2)
        else:
            disc = B*B - 4.0*A*C
            if disc < 0.0:
                print "error finding surface value of r"
                Smin = -1.5*c
            else:
                Smin = -(-B - math.sqrt(disc))/(2.0*A)
    return Smin

def RHS(x, t, alpha, beta, c, h):
    """
    Used as the right hand side values for integrating the r characteristics in Region 2
    dx/dt = 2c - (2/3)s
    To get s we do a search after finding the bounding values at the free surface
    and the interface between Region 2 and 3
    """
    Smin = FreeSurface_S(t, alpha, beta, c, h)
    #print " alpha, beta, t, a, b, c, disc, Smin, Smax", alpha, "  ", beta, "  ", t, "  ",  A, "  ", B, "  ", C, "  ", disc, "  ", Smin, "  ", -(-B + math.sqrt(disc))/(2.0*A)
    s = scipy.optimize.brentq(S_from_XT_reg2, Smin, 1.5*c, args=(x, t, alpha, beta, c, h)) # for alpha=0 , beta= -3
    #print (" x, t, s ", x[0] , "  ", t, "  ", s)
    return 2.0*c - (2.0*s/3.0)

def velocity(x, t, alpha, beta, c, h):
    """
    find the particle velocity at x and t
    """
    r, s = xt2rs(x, t, alpha, beta, c, h)
    return r - s

def xt2rs(x, t, alpha, beta, c, h):
    """
    find the r and s characteristics that go through the point (x, t)
    """
    Smin = FreeSurface_S(t, alpha, beta, c, h)
 
    # time region 2 reaches free surface
    time2 = -2.0*h*(1 + alpha/3.0 + beta)/c
 
    # find x position of free surface at time t (r1 = -Smin)
    x1 = -2.0*Smin*t + 0.5*evalPhi2p(-Smin,alpha,beta,c,h)
 
    # find s value at region1/ region 2 boundary at time t
    if t > time2:
        # in this case Region 2 extends to the free surface, s2 must be -1.5*c
        s2 = -1.5*c
    else:
        # print "t > time2 search for s2, Smin ", Smin
        s2 = scipy.optimize.brentq(SXT2, Smin, 1.5*c, args=(t, alpha, beta, c, h)) 
    # Calculate x value at region 1/region 2 boundary
    x2 = -(4.0/3.0*s2 - c)*t + evalPhip(s2,alpha,beta,c,h)/(1.5*c + s2 + 1e-20) + (evalPhi(1.5*c,alpha,beta,c,h) - evalPhi(s2,alpha,beta,c,h))/(1.5*c + s2 + 1e-20 )**2
    
    # Calculate x value at region 2/region 3 boundary
    x3 = -c*t
    
    if (x <= x3):
        # In region 3, initial conditions still hold
        r = 1.5*c
        s = 1.5*c
    elif (x > x1):
        # outside free surface, set values to the free surface at this time
        r = -Smin
        s =  Smin
    elif x <= x2:
        # region 2, r = 1.5*c, s varies
        r = 1.5*c
        s = scipy.optimize.brentq(S_from_XT_reg2, Smin, 1.5*c, args=(x, t, alpha, beta, c, h))
    else:
        # find r and s in region 1
        # get initial guess from bounding values, (r,s) is (-Smin, Smin) at x1, (1.5*c, s2) at x2
        R_init = 1.5*c + (-Smin - 1.5*c)*(x - x2)/(x1 - x2)
        S_init = s2 + (Smin - s2)*(x - x2)/(x1 - x2)
        sol = scipy.optimize.root(RS_fromXTreg1, [R_init, S_init], args=(x, t, alpha, beta, c, h))
        r = sol.x[0]
        s = sol.x[1]
    return r, s

def cubic_roots(b0,b1,b2,b3):
    """
    Find roots of cubic b3*x^3 + b2*x^2 + b1*x + b0
    return tuple of ( # roots, [roots,])
    """
    num_roots = 0
    roots = []
    # test for linear or quadratic cases
    if (abs(b3) < 1e-15):
        # test for linear case
        if (abs(b2) < 1e-15):
            # test for constant case
            if (abs(b1) < 1e-15):
                num_roots = 0
            else:
                num_roots = 1
                roots.append(-b0/(b1 + 1e-80))
        else:
            # quadratic case
            discriminant = b1*b1 - 4.0*b0*b2
            if (discriminant >= 0.):
                discriminant = math.sqrt(discriminant)
                
                num_roots = 2
                roots.append((-b1 - discriminant)/(2.0*b2))
                roots.append((-b1 + discriminant)/(2.0*b2))
    else:
        # cubic case
        # put in normalized form 0 = a0 + a1*x + a2*x^2 + x^3
        a0 = b0/b3
        a1 = b1/b3
        a2 = b2/b3
        # print "a0, a1, a2   ", a0, a1, a2
        
        q = a1/3.0 - a2*a2/9.0
        r3 = (a1*a2 - 3.0*a0)/6.0 - a2*a2*a2/27.0

        # print " q, r3  ", q, r3
        
        discriminant = q*q*q + r3*r3
        # print "discriminant  ", discriminant
        if (discriminant/(abs(r3) + 1e-80) > 1e-12):
            # 1 real root
            R = r3 + math.sqrt(discriminant)
            Rthird1 = 0.0
            if (R >= 0.):
                Rthird1 = math.pow(R, 1./3.)
            else:
                Rthird1 = -math.pow(-R, 1./3.)
            
            R = r3 - math.sqrt(discriminant)
            Rthird2=0.
            if (R >= 0.):
                Rthird2 = math.pow(R, 1./3.)
            else:
                Rthird2 = -math.pow(-R, 1./3.)
            # print "1 root: R, Rthird1, Rthird2 ", R, Rthird1, Rthird2
            num_roots = 1
            roots.append(Rthird1 + Rthird2 - a2/3.0)
        elif (abs(discriminant/(r3 + 1e-80)) < 1e-12):
            # 2 real roots, a single and a double root
            Rthird1 = 0
            if (r3 >= 0.):
                Rthird1 = math.pow(r3,1./3.)
            else:
                Rthird1 = -math.pow(-r3,1./3.)
            # print "2 roots: Rthird1 ", Rthird1
            num_roots = 2
            roots.append(2.0*Rthird1 - a2/3.0)
            roots.append(-Rthird1 - a2/3.0) # a double root
        else:
            # three real roots
            # discriminant < 0. */
            discriminant = abs(discriminant)
            R = math.sqrt(r3*r3 + discriminant)
            R = math.pow(R, 1./3.)
            discriminant = math.sqrt(discriminant)
            theta = (math.atan2(discriminant,r3)+ 0.0*math.pi/3.0 )/ 3.0
            
            # print "3 roots: disc, R, theta ", discriminant, R, theta
            num_roots = 3
            roots.append(2.0*R*math.cos(theta) - a2/3.0)
            roots.append(-R*(math.cos(theta) + math.sqrt(3.)*math.sin(theta)) - a2/3.0)
            roots.append(-R*(math.cos(theta) - math.sqrt(3.)*math.sin(theta)) - a2/3.0)
    return (num_roots,roots)

def setdensityEmat(x):
    """
    This function will return the density and energy per volume
    for the coordinate given.
    """
    num_roots, roots = cubic_roots(1.0 - x, 2.0/3.0*alpha, -4.0/9.0*(3. + 2.*alpha + beta ), 8./27*(2. + alpha + beta) )
    R = 0
    count = 0
    for i in range(num_roots):
        if (roots[i] >= 0 and roots[i] <= 1.5):
            R = roots[i]
            count = count + 1
    #print count
    cs = 2.0*R/3.0 # sound speed
    den = cs*cs*cs
    emat = 0.9*cs*cs
    return (den, emat)


