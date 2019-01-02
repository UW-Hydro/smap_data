
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


def edges_from_centers(centers):
    ''' Return an array of grid edge values from grid center values - note: min and max edge are hardcoded as 0 and 1000
    Parameters
    ----------
    centers: <np.array>
        A 1-D array of grid centers. Typically grid-center lats or lons. Dim: [n]

    Returns
    ----------
    edges: <np.array>
        A 1-D array of grid edge values. Dim: [n+1]
    '''

    edges = np.zeros(len(centers)+1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = 0
    edges[-1] = 1000

    return edges
