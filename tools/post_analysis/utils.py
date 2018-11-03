
import numpy as np


def calc_monthly_PET(Ta):
    ''' Calculate monthly PET based on Ta (monthly average) 
        Input: Ta [deg C]
        Return: PET [mm/mon]'''

    eas = 0.611*np.exp(17.3*Ta/(237.3+Ta))  # saturated vapor pressure [kPa]
    PET = 40.9*eas # PET [mm/mon]
    return PET


def find_1deg_grid_gldas(lat, lon):
    ''' Find the 1deg grid cell that a (lat, lon) point falls in
        according to GLDAS grid

    Input arguments: lat, lon (can be single number or np array)
    Return: lat_grid, lon_grid
    Module requred: import numpy as np
    '''
    lat_grid = np.around(lat-0.5) + 0.5
    lon_grid = np.around(lon-0.5) + 0.5
    return lat_grid, lon_grid

