"""
"""

import numpy as np
from numpy import arctan2 as atan2
from astropy import units as u
from Geometry import *



def guessPixDirectionFieldView(pix_x, pix_y, tel_phi, tel_theta, tel_view = 4 * u.degree, array_focus_dist=None, tel_x=None, tel_y=None, array_phi=None, array_theta=None):
    """
    guesses the direction a pixel is looking at from the direction the telescope is pointing to
    
    Parameters
    ----------
    pix_x, pix_y: float
        pixel coordinates on the camera
    
    tel_view: float
        field of view of the telescope: half-angle of the field of view opening cone
    
    tel_phi, tel_theta: float
        angles of telescope viewing direction in local coordinate system
        (phi from north westwards; theta = 0 -> straight up)
    
    
     Assumptions
     -----------
      - the angle of incidence is linear in the pixel plane
      - the direction of the telescope corresponds to the centre of the camera
      - the centre of the camera is at (0,0)
      - the full view angle corresponds ot the outermost pixel (largest distance from the centre)
    
    """
    
    maxR = 0 * u.m
    for x, y in zip(pix_x[:-10], pix_y[:-10]):
        R = (( (x)**2 + (y)**2 )**.5)
        if maxR < R:
           maxR = R


    # beta is the pixel's angular distance to the centre according to beta / tel_view = r / maxR
    # alpha is the polar angle between the y-axis and the pixel
    # to find the direction the pixel is looking at, 
    #  - the pixel direction is set to the telescope direction
    #  - offset by beta towards up
    #  - rotated around the telescope direction by the angle alpha
    pix_alpha = np.array([ atan2(x, y)/u.rad for x,y in zip(pix_x, pix_y) ]) * u.rad
    pix_beta  = np.array([ ( (x**2 + y**2)**.5)/u.m for x,y in zip(pix_x, pix_y) ]) * u.m
    pix_beta  = pix_beta * (tel_view)/(maxR)
    
    tel_dir = SetPhiTheta(tel_phi,tel_theta)

    pix_dirs = []
    
    for a, b in zip(pix_alpha,pix_beta):
        pix_dir = SetPhiTheta( tel_phi, tel_theta + b )
        
        pix_dir = rotateAroundAxis(pix_dir, tel_dir, a)
        pix_dirs.append(pix_dir)
        
    return np.array( pix_dirs )


def guessPixDirectionFocLength(pix_x, pix_y, tel_phi, tel_theta, tel_foclen = 4 * u.m, tel_x=None, tel_y=None, array_phi=None, array_theta=None):
    # beta is the pixel's angular distance to the centre according to beta / tel_view = r / maxR
    # alpha is the polar angle between the y-axis and the pixel
    # to find the direction the pixel is looking at, 
    #  - the pixel direction is set to the telescope direction
    #  - offset by beta towards up
    #  - rotated around the telescope direction by the angle alpha
    
    pix_alpha = np.array([ atan2(x, y)/u.rad for x,y in zip(pix_x, pix_y) ]) * u.rad
    pix_beta  = np.array([ ( (x**2 + y**2)**.5)/u.m for x,y in zip(pix_x, pix_y) ]) * u.m
    pix_beta  = pix_beta / tel_foclen * u.rad 
    
    tel_dir = SetPhiTheta(tel_phi,tel_theta)

    pix_dirs = []
    
    
    for a, b in zip(pix_alpha,pix_beta):
        pix_dir = SetPhiTheta( tel_phi, tel_theta + b )
        
        pix_dir = rotateAroundAxis(pix_dir, tel_dir, a)
        pix_dirs.append(pix_dir)
        
    return np.array( pix_dirs )