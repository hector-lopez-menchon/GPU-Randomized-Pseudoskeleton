% EM_data = user_set_EM_data()
% User function to set ElectroMagnetic data structure
%
% Juan M. Rius, AntennaLab, Universitat Politecnica de Catalunya (Spain), v1.0, August 2007

% /***************************************************************************
%  *   Copyright (C) 2007 by Juan M. Rius                                    *
%  *   AntennaLab, Universitat Politecnica de Catalunya, rius@tsc.upc.edu    *
%  *                                                                         *
%  *   This program is free software; you can redistribute it and/or modify  *
%  *   it under the terms of the GNU General Public License as published by  *
%  *   the Free Software Foundation; either version 2 of the License, or     *
%  *   (at your option) any later version.                                   *
%  *                                                                         *
%  *   This program is distributed in the hope that it will be useful,       *
%  *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
%  *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
%  *   GNU General Public License for more details.                          *
%  *                                                                         *
%  *   You should have received a copy of the GNU General Public License     *
%  *   along with this program; if not, write to the                         *
%  *   Free Software Foundation, Inc.,                                       *
%  *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
%  ***************************************************************************/

function EM_data = user_set_EM_data()

lambda = 1;     % Wavelength
k = 2*pi/lambda; 
eta = 120*pi; 
%field = 1;
field = 3;
Rint_s = 0.2;       % MoM Integration radius (meters). Rint=0 is enough if basis functions are very small.
Rint_f = Rint_s;
Ranal_s = 0;
corr_solid = 0;
flag = 1;
sym_source_field = 1; % Symmetric Green's function: Znm = Zmn.'. Size of compressed Z will be halved.

EM_data = struct('lambda',lambda, 'k',k, 'eta',eta, 'field',field, 'Rint_s',Rint_s, 'Rint_f',Rint_f, 'Ranal_s',Ranal_s, 'corr_solid',corr_solid, 'flag',flag, 'sym_source_field', sym_source_field );