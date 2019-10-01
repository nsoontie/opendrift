# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

#####################################
# NOTE:
# This reader is under development,
# and presently not fully functional
#####################################

import logging
from datetime import datetime

import numpy as np
from netCDF4 import Dataset, MFDataset, num2date
from scipy.interpolate import LinearNDInterpolator

from opendrift.readers.basereader import BaseReader, pyproj

try:
    import xarray as xr
    has_xarray = True
except:
    has_xarray = False
    

class Reader(BaseReader):

    def __init__(self, filename=None, name=None, buffer=0.2,
                 latstep=0.01, lonstep=0.01):

        if filename is None:
            raise ValueError('Need filename as argument to constructor')
        filestr = str(filename)
        if name is None:
            self.name = filestr
        else:
            self.name = name

        self.geobuffer = buffer
        self.latstep = latstep
        self.lonstep = lonstep

        # Due to misspelled standard_name in
        # some (Akvaplan-NIVA) FVCOM files
        variable_aliases = {
            'eastward_sea_water_velocity': 'x_sea_water_velocity',
            'Northward_sea_water_velocity': 'y_sea_water_velocity',
            'eastward wind': 'x_wind',
            'northward wind': 'y_wind'
            }

        # Mapping FVCOM variable names to CF standard_name
        fvcom_mapping = {
            'um': 'x_sea_water_velocity',
            'vm': 'y_sea_water_velocity'}

        self.return_block = True

        try:
            # Open file, check that everything is ok
            logging.info('Opening dataset: ' + filestr)
            if ('*' in filestr) or ('?' in filestr) or ('[' in filestr):
                logging.info('Opening files with MFDataset')
                if has_xarray:
                    try:
                        self.Dataset = \
                            xr.open_mfdataset(filename,
                                              concat_dim='time_counter',
                                              data_vars='minimal',
                                              coords='minimal',
                                              combine='nested')
                    except:
                        self.Dataset = \
                            xr.open_mfdataset(filename,
                                              concat_dim='time',
                                              data_vars='minimal',
                                              coords='minimal',
                                              combine='nested')
                else:
                    self.Dataset = MFDataset(filename)
            else:
                logging.info('Opening file with Dataset')
                if has_xarray:
                    self.Dataset = xr.open_dataset(filename)
                else:
                    self.Dataset = Dataset(filename, 'r')
        except Exception as e:
            raise ValueError(e)

        # We are reading and using lon/lat arrays,
        # and not any projected coordinates
        self.proj4 =  '+proj=latlong'

        logging.debug('Finding coordinate variables.')
        # Find x, y and z coordinates
        for var_name in self.Dataset.variables:
            var = self.Dataset.variables[var_name]
            if var.ndim > 1:
                continue  # Coordinates must be 1D-array
            if has_xarray:
                attributes = var.attrs
                att_dict = var.attrs
            else:
                attributes = var.ncattrs()
                att_dict = var.__dict__
            standard_name = ''
            long_name = ''
            axis = ''
            units = ''
            CoordinateAxisType = ''
            if 'standard_name' in attributes:
                standard_name = att_dict['standard_name']
            if 'long_name' in attributes:
                long_name = att_dict['long_name']
            if 'axis' in attributes:
                axis = att_dict['axis']
            if 'units' in attributes:
                units = att_dict['units']
            if '_CoordinateAxisType' in attributes:
                CoordinateAxisType = att_dict['_CoordinateAxisType']
            if standard_name == 'longitude' or \
                    long_name == 'longitude' or \
                    var_name == 'longitude' or \
                    axis == 'X' or \
                    CoordinateAxisType == 'Lon' or \
                    standard_name == 'projection_x_coordinate':
                self.xname = var_name
                # Fix for units; should ideally use udunits package
                if units == 'km':
                    unitfactor = 1000
                else:
                    unitfactor = 1
                if has_xarray:
                    var_data = var.values
                else:
                    var_data = var[:]
                x = var_data*unitfactor
                self.unitfactor = unitfactor
                self.numx = var_data.shape[0]
            if standard_name == 'latitude' or \
                    long_name == 'latitude' or \
                    var_name == 'latitude' or \
                    axis == 'Y' or \
                    CoordinateAxisType == 'Lat' or \
                    standard_name == 'projection_y_coordinate':
                self.yname = var_name
                # Fix for units; should ideally use udunits package
                if units == 'km':
                    unitfactor = 1000
                else:
                    unitfactor = 1
                if has_xarray:
                    var_data = var.values
                else:
                    var_data = var[:]
                y = var_data*unitfactor
                self.numy = var_data.shape[0]
            if standard_name == 'depth' or axis == 'Z':
                if has_xarray:
                    var_data = var.values
                else:
                    var_data = var[:]
                if 'positive' not in var.ncattrs() or \
                        var.__dict__['positive'] == 'up':
                    self.z = var_data
                else:
                    self.z = -var_data
            if standard_name == 'time' or axis == 'T' or var_name == 'time':
                # Read and store time coverage (of this particular file)
                if has_xarray:
                    var_data = var.values
                else:
                    var_data = var[:]
                time = var_data
                time_units = units
                if has_xarray:
                    self.times = [datetime.utcfromtimestamp((OT -
                        np.datetime64('1970-01-01T00:00:00Z')
                            ) / np.timedelta64(1, 's')) for OT in time]
                else:
                    self.times = num2date(time, time_units)
                self.start_time = self.times[0]
                self.end_time = self.times[-1]
                if len(self.times) > 1:
                    self.time_step = self.times[1] - self.times[0]
                else:
                    self.time_step = None

        if 'x' not in locals():
            raise ValueError('Did not find x-coordinate variable')
        if 'y' not in locals():
            raise ValueError('Did not find y-coordinate variable')

        self.lon = x
        self.lat = y

        # Find all variables having standard_name
        self.variable_mapping = {}
        for var_name in self.Dataset.variables:
            if var_name in [self.xname, self.yname, 'depth']:
                continue  # Skip coordinate variables
            var = self.Dataset.variables[var_name]
            if has_xarray:
                attributes = var.attrs
                att_dict = var.attrs
            else:
                attributes = var.ncattrs()
                att_dict = var.__dict__
            if 'standard_name' in attributes:
                standard_name = str(att_dict['standard_name'])
                if standard_name in variable_aliases:  # Mapping if needed
                    standard_name = variable_aliases[standard_name]
                self.variable_mapping[standard_name] = str(var_name)
            elif var_name in fvcom_mapping:
                self.variable_mapping[fvcom_mapping[var_name]] = \
                    str(var_name)

        self.variables = list(self.variable_mapping.keys())

        self.xmin = self.lon.min()
        self.xmax = self.lon.max()
        self.ymin = self.lat.min()
        self.ymax = self.lat.max()

        # Run constructor of parent Reader class
        super(Reader, self).__init__()

    def get_variables(self, requested_variables, time=None,
                      x=None, y=None, z=None, block=False):

        requested_variables, time, x, y, z, outside = \
            self.check_arguments(requested_variables, time, x, y, z)

        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
            self.nearest_time(time)

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)


        # Finding a subset around the particles, so that
        # we do not interpolate more points than is needed.
        # Performance is quite dependent on the given buffer,
        # but it should not be made too small to make sure
        # particles are inside box
        #buffer = .1  # degrees around given positions

        lonmin = x.min() - self.geobuffer
        lonmax = x.max() + self.geobuffer
        latmin = y.min() - self.geobuffer
        latmax = y.max() + self.geobuffer
        c = np.where((self.lon > lonmin) &
                     (self.lon < lonmax) &
                     (self.lat > latmin) &
                     (self.lat < latmax))[0]

        # Making a lon-lat grid onto which data is interpolated
        #lonstep = .01  # hardcoded for now
        #latstep = .01  # hardcoded for now
        lons = np.arange(lonmin, lonmax, self.lonstep)
        lats = np.arange(latmin, latmax, self.latstep)
        lonsm, latsm = np.meshgrid(lons, lats)

        # Initialising dictionary to contain data
        variables = {'x': lons, 'y': lats, 'z': z,
                     'time': nearestTime}

        # Reader coordinates of subset
        for par in requested_variables:
            var = self.Dataset.variables[self.variable_mapping[par]]
            if var.ndim == 1:
                data = var[c]
            elif var.ndim == 2:
                data = var[indxTime,c]
            elif var.ndim == 3:
                data = var[indxTime,0,c]
            else:
                raise ValueError('Wrong dimension of %s: %i' %
                                 (var_name, var.ndim))
            if has_xarray:
                data = np.asarray(data)
            if 'interpolator' not in locals():
                logging.debug('Making interpolator...')
                interpolator = LinearNDInterpolator((self.lat[c],
                                                     self.lon[c]),
                                                    data)
            else:
                # Re-use interpolator for other variables
                interpolator.values[:,0] = data
            interpolator((0,0))

            variables[par] = interpolator(latsm, lonsm)

        #print variables
        return variables
