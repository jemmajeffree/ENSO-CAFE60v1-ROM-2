import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os.path
import courtney
import scipy.optimize, scipy.signal
import cftime
import datetime
import scipy.linalg

import jemma_ODE_solvers3 as jde
#Epoch 2 25th Sep 2021

full_pacific = np.array((130,280,-5,5))
west_pacific = np.array((120,200,-5,5))
nino3_region = np.array((210,270,-5,5))
nino34_region = np.array((190,240,-5,5))
nino4_region = np.array((160,210,-5,5))
nino12_region = np.array((270,280,-10,0))
    
    
def anomaly_file_name(year0,year1, data_name, e_member):
    '''The filename containing calculated indices
    data_name should be 'sst' or 'mld' or 'i20' 
    Date range is [year1,year0) '''
    iteration = '_'+str(e_member)+'_'
    when = str(year0)+'_'+str(year1)+'_'
    return (when+data_name+iteration+'.nc')
    
def calc_box_anom(var, data_name, e_member = 0, clim_period = slice('1975','2018'), write = True):
    '''Calculate the anomaly for each point within the box
    longitude must be measured in degrees east
    '''
    lonlat = np.array((130,280,-10,10))
    # Wriggle longitude into the coordinates it needs
    lonlat[:2] = (((lonlat[:2]-80.01) %360) -279.99)
    
    assert lonlat[0] < lonlat[1] and lonlat[2] < lonlat[3], 'Region must be defined by sorted longitude/latitude values. If you intend to cross the 80E line this code\'s gonna need some more work'
    
    
    var_box = var.sel(xt_ocean = slice(lonlat[0],lonlat[1]),yt_ocean = slice(lonlat[2],lonlat[3]))
    
    var_clim = var_box.sel(time=clim_period).groupby('time.month').mean('time')
    var_anom = var_box.groupby('time.month') - var_clim
    
    var_anom = var_anom.chunk({'time': 24})
    #var_anom = var_anom.chunk('auto')
    if write:
        var_anom.to_netcdf(anomaly_file_name(clim_period.start,clim_period.stop,data_name,e_member))
    #var_anom_box = var_anom.mean(['xt_ocean','yt_ocean'])
    
    return var_anom
    
def average_region(data, lonlat):
    lonlat[:2] = (((lonlat[:2]-80.01) %360) -279.99)
    return data.sel(xt_ocean = slice(lonlat[0],lonlat[1]),
                    yt_ocean = slice(lonlat[2],lonlat[3]) ).mean(['xt_ocean','yt_ocean'])    
    
def plot_correlation(data_a,data_b, plot = True):
    ''' Calculate the correlation between two data arrays
    and plot if plot=True'''
    
    assert(len(data_a.time)+len(data_b.time))>=48, "I'd appreciate minimum 2 years of data"
    plt.figure('Cross correlation')
    l = data_a.shape[0]
    plt.plot(np.arange(-24,24),scipy.signal.correlate(data_a,data_b)[l-24:l+24])   
    plt.show()
    return
    
def best_fit_matrix(y,dt, verbose = False):
    '''Find the best A for dy/dt = Ay and return this y and the variance
    if h is very small get there in steps instead of trying to approach it all at once
    if verbose, step through what I'm doing and the values achieved so far'''
    
    G = jde.fast_2D_A(y,dt)
    #G = old_jde.linear_lstsq_coefficient_matrix(np.arange(720),y)
    if verbose:
        print(G)
        print('Eigenvalues:' +str(scipy.linalg.eig(G.reshape((y.shape[0],y.shape[0])))[0]))
    
    A1 = scipy.optimize.minimize(jde.timeseries_optimiser,G,(y,dt)).x
    if verbose:
        print(A1.reshape((y.shape[0],y.shape[0])))
        print('Eigenvalues:' +str(scipy.linalg.eig(A1.reshape((y.shape[0],y.shape[0])))[0]))

    
    #Calculate variance
    variance = jde.timeseries_optimiser(A1,y,dt)/y.shape[1]/y.shape[0] 
    
    return A1, variance

def plot_fit(A,t,y,n, h = 0.01):
    ''' Show visually how well it seems to line up'''
   
    for i in range(y.shape[0]):
        plt.figure(i,(20,5))
        plt.plot(t,y[i],'ko')

        plt.title('y'+str(i))
   
   
    for i in range(y.shape[1]-n):
        y_comp = np.zeros((y.shape[0],int(n/h)))
        y0 = y[:,i]

        y_comp = jde.numeric_ODE_system(y0,t[i],t[i+n],0.01,jde.ODE_system,A)
        for f in range(y.shape[0]):
            plt.figure(f)
            plt.plot(np.arange(t[i],t[i+n],h),y_comp[f,:])
            
def isotherm_depth(temp, target_temp=20, depth_name=None, rename = 'i20'):
    """ 
        Written by Dougie
    
        Returns the depth of an isotherm given a target temperature. If no temperatures in the column
        exceed the target temperature, a nan is returned at that point.        | Author: Dougie Squire        Parameters
        ----------
        temp : xarray DataArray
            Array containing values of temperature with at least a depth dimension
        target_temp : value, optional
            Value of temperature used to compute isotherm depth. Default value is 20 degC
        depth_name : str, optional
            Name of depth coordinate        Returns
        -------
        isotherm_depth : xarray DataArray
            Array containing the depth of the requested isotherm        Examples
        --------
        >>> temp = xr.DataArray(20 + np.random.normal(scale=5, size=(4,4,10)), 
        ...                     coords=[('lat', np.arange(-90,90,45)), ('lon', np.arange(0,360,90)), 
        ...                             ('depth', np.arange(2000,0,-200))])
        >>> isotherm_depth(temp)
        <xarray.DataArray 'isosurface' (lat: 4, lon: 4)>
        array([[ 400., 1600., 2000.,  800.],
               [1800., 2000., 1800., 2000.],
               [2000., 2000., 2000., 1600.],
               [1400., 2000., 2000., 2000.]])
        Coordinates:
          * lat      (lat) int64 -90 -45 0 45
          * lon      (lon) int64 0 90 180 270        Notes
        -----------
        | All input array coordinates must follow standard naming
        | If multiple occurrences of target occur along the depth coordinate, only the maximum value of \
                coord is returned
        | The current version includes no interpolation between grid spacing. This should be added as \
                an option in the future
    """    
    def _isosurface(ds, coord, target):
        """
            Returns the max values of a coordinate in the input array where the input array is greater than \
                    a prescribed target. E.g. returns the depth of the 20 degC isotherm. Returns nans for all \.   
                    points in input array where isosurface is not defined. If   
        """        
        mask = ds > target
        ds_mask = mask * ds[coord]
        isosurface = ds_mask.max(coord)
        isosurface = isosurface.rename(rename)
        return isosurface.where(ds.max(dim=coord) > target)    
    return _isosurface(temp, coord=depth_name, target=target_temp)

def interpolate_isotherm_depth(temp, target_temp=20, depth_name='st_ocean', rename = 'i2p'):
    """ 
        Adapted from Dougie's code above
        Interpolates between the lowest point above target_temp and the next recorded temp below
    

    """    

    i0 = temp.where(temp>target_temp).idxmin(dim=depth_name).compute()
    i1 = i0+1

    t0 = temp.isel({depth_name:i0})
    z0 = temp[depth_name].isel({depth_name:i0})

    t1 = temp.isel({depth_name:i1})
    z1 = temp[depth_name].isel({depth_name:i1})

    
    return(target_temp-y0)*(x1-x0)/(y1-y0)+x0

    return jde.total_residual(R,weights)

def build_pairings(Y,months = (2,3,4)):
    ''' Turn data from a timeseries into pairs of start month end month data 
    Y is full set of data (assuming starting in January)
    months is the start months which are used (so MAM is 2,3,4 but finishes on months AMJ'''
    
    mpy = len(months)
    starts = np.zeros((Y.shape[0],Y.shape[1]//12*mpy))
    ends = np.zeros((Y.shape[0],Y.shape[1]//12*mpy))
    
    
    for y in range(Y.shape[1]//12):
        for i,m in enumerate(months):
            if y*12+m+1<Y.shape[1]: #If not starting from the last year bit

                starts[:,y*mpy+i] = Y[:,y*12+m]
                ends[:,y*mpy+i] = Y[:,y*12+m+1] # Doing it this way will automatically jump the year-gap
    return starts, ends

def build_IVP_pairs(sst_index, i20_index ,y0 = 1965,y1 = 2018,season = None):
    ''' Creates an array of start vectors and one of end vectors for a given season and (inclusive) time range
    season is one of  'MAM', 'JJA', 'SON', 'DJF'
    time range is [y0,y1] inclusive (for DJF this will include JF from y1+1)
    Returns Y0 (2,N) start points
            Y1 (2,N) end points
            
    I believe this is effectively the same function as above but with more flexibility and vectorised'''
    
    assert np.all(sst_index.time == i20_index.time)
    
    if season is None:
        j = np.where((sst_index.time.dt.year >=y0)&(sst_index.time.dt.year <=y1))[0]
    else:
        if season == 'DJF':
            j = np.where((sst_index.time.dt.season==season)&(sst_index.time.dt.year >=y0)&(sst_index.time.dt.year <=y1+1))[0]
            j = j[2:-1]  ### potential bug in here? not checked
        else:
            j = np.where((sst_index.time.dt.season==season)&(sst_index.time.dt.year >=y0)&(sst_index.time.dt.year <=y1))[0]
    Y0 = np.array(xr.concat(  [sst_index.isel(time=j).stack({'new':('ensemble_member','time')}),
                   i20_index.isel(time=j).stack({'new':('ensemble_member','time')})],
               dim='var'))
    
    Y1 = np.array(xr.concat(  [sst_index.isel(time=j+1).stack({'new':('ensemble_member','time')}),
                   i20_index.isel(time=j+1).stack({'new':('ensemble_member','time')})],
               dim='var'))
    
    return Y0, Y1


def paired_best_fit_matrix(Y0,Y1,dt, verbose = False):
    '''Find the best A for dy/dt = Ay and return this A and the variance
    Is forecasting forwards from ivp_starts to ivp_ends, which it assumes are a month apart
    if h is very small get there in steps instead of trying to approach it all at once
    if verbose, step through what I'm doing and the values achieved so far'''

    
    # Do a quick linear least squares approximation to ensure scipy.optimize.minimize finds the right local minimum
    dy = (Y1-Y0)/dt
    G = np.linalg.lstsq(((Y0+Y1)/2).T,dy.T, rcond=None)[0].T
    
    if verbose:
        print(G)
        print('Eigenvalues:' +str(scipy.linalg.eig(G.reshape((2,2)))[0]))
    
    # Full optimisation
    A1 = scipy.optimize.minimize(jde.paired_optimiser,G,(Y0,Y1,dt)).x
    if verbose:
        print(A1.reshape((2,2)))
        print('Eigenvalues:' +str(scipy.linalg.eig(A1.reshape((2,2)))[0]))

    
    #Calculate variance
    variance = jde.paired_optimiser(A1,Y0,Y1,dt)/Y0.shape[0]/Y0.shape[1]
    
    return A1, variance

def paired_best_fit_matrix_exp(Y0,Y1,dt, verbose = False):
    '''Find the best A for dy/dt = Ay and return this A and the variance
    Is forecasting forwards from ivp_starts to ivp_ends, which it assumes are a month apart
    if h is very small get there in steps instead of trying to approach it all at once
    if verbose, step through what I'm doing and the values achieved so far'''

    # Do a quick linear least squares approximation to ensure scipy.optimize.minimize finds the right local minimum
    dy = (Y1-Y0)/dt
    G = np.linalg.lstsq(((Y0+Y1)/2).T,dy.T, rcond=None)[0].T
    
    if verbose:
        print(G)
        print('Eigenvalues:' +str(scipy.linalg.eig(G.reshape((2,2)))[0]))
    
    # Full optimisation
    A1 = scipy.optimize.minimize(jde.paired_optimiser_exp,G,(Y0,Y1,dt)).x
    if verbose:
        print(A1.reshape((2,2)))
        print('Eigenvalues:' +str(scipy.linalg.eig(A1.reshape((2,2)))[0]))

    
    #Calculate variance
    variance = jde.paired_optimiser_exp(A1,Y0,Y1,dt)/Y0.shape[0]/Y0.shape[1]
    
    return A1, variance

def fit_matrix_for_data_subset(filename, 
                               sst_index,
                               i20_index,
                            startyears = np.array((1965,)), #Anything iterable, start year of each fit
                            windowlength = 2019-1965, #That's a subtraction deliberately, the length of time included in each fit
                            ensemble = [slice(1,97)], #also has to be iterable, what ensemble members to include in each fit
                            season = None, #None or one of 'DJF','MAM','SON','JJA'
                            verbose = False, #If true, prints updates about how the code's going
                           ): 
    
    # Set up data storage
    coefficient_matrix = np.zeros((len(startyears),len(ensemble),2,2)) # dims: start year, ensemble member, matrix
    variance = np.zeros((len(startyears),len(ensemble))) #dims: start year, ensemble member

    # Iterate through each fit
    for i,y in enumerate(startyears):
        for j,e in enumerate(ensemble):
            if type(e) == int:
                e = np.array((e,))
                
            # Build data structure for optimiser
            Y0,Y1 = build_IVP_pairs(sst_index.sel(ensemble_member = e), 
                                    i20_index.sel(ensemble_member = e),y,y+windowlength-1,season=season)
            
            # Calculate fit
            A, var = paired_best_fit_matrix(Y0,Y1,1,verbose=verbose) 
            
            # Store fit
            coefficient_matrix[i,j,:,:] = A.reshape((2,2))
            variance[i,j] = var
            
            if verbose:  
                print("Year "+str(y)+'-'+str(y+windowlength-1)+', ensemble member '+str(e)+' finished')
                print(A)
                print('----------------------------------------')
            
    # Save output with metadata
    coefficient_matrix_xr = xr.DataArray(coefficient_matrix,name = 'A', dims = ('start_year','ensemble_members', 'matrix_row','matrix_collumn'))
    
    if type(ensemble) == list:
        ensemble = [str(e) for e in ensemble]
    coefficient_matrix_xr = coefficient_matrix_xr.assign_coords({'start_year': startyears, 'ensemble_members':ensemble})
    
    coefficient_matrix_xr.to_netcdf(filename+'_A.nc')

    variance_xr = xr.DataArray(variance,name = 'var', dims = ('start_year','ensemble_members'))
    variance_xr = variance_xr.assign_coords({'start_year': startyears, 'ensemble_members':ensemble})
    variance_xr.to_netcdf(filename+'_var.nc')
    
    return coefficient_matrix,variance


# def ORAS5_fit_matrix_for_data_subset(filename, 
#                                sst_index,
#                                i20_index,
#                             startyears = np.array((1965,)), #Anything iterable, start year of each fit
#                             windowlength = 2019-1965, #That's a subtraction deliberately, the length of time included in each fit
#                             season = None, #None or one of 'DJF','MAM','SON','JJA'
#                             verbose = False, #If true, prints updates about how the code's going
#                            ): 
    
#     coefficient_matrix = np.zeros((len(startyears),1,2,2)) # dims: start year, ensemble member, matrix
#     variance = np.zeros((len(startyears),1)) #dims: start year, ensemble member

#     for i,y in enumerate(startyears):
#         Y0,Y1 = build_IVP_pairs(sst_index, 
#                                 i20_index,y,y+windowlength-1,season=season)

#         A, var = paired_best_fit_matrix(Y0,Y1,1,verbose=verbose) #Calculate fit

#         coefficient_matrix[i,0,:,:] = A.reshape((2,2)) #Save fit
#         variance[i,0] = var

#         if verbose:  
#             print("Year "+str(y)+'-'+str(y+windowlength-1)+' finished')
#             print(A)
#             print('----------------------------------------')
            
#     #Save output
#     coefficient_matrix_xr = xr.DataArray(coefficient_matrix,name = 'A', dims = ('start_year','ensemble_members', 'matrix_row','matrix_collumn'))
    
#     coefficient_matrix_xr = coefficient_matrix_xr.assign_coords({'start_year': startyears, 'ensemble_members':[0]})
    
#     coefficient_matrix_xr.to_netcdf(filename+'_A.nc')

#     variance_xr = xr.DataArray(variance,name = 'var', dims = ('start_year','ensemble_members'))
#     variance_xr = variance_xr.assign_coords({'start_year': startyears, 'ensemble_members':[0]})
#     variance_xr.to_netcdf(filename+'_var.nc')
    
#     return coefficient_matrix,variance