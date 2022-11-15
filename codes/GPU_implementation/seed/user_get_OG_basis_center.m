% [rcx, rcy, rcz, N] = user_get_OG_basis_center(OG_data)
% User function to obtain the center of Object Geometry basis functions
%
% OG_data   = struct containing Object Geometry data 
%
% Output:
% rcx = x-coordinates of basis functions centers (vector)
% rcy = y-coordinates of basis functions centers (vector)
% rcz = z-coordinates of basis functions centers (vector)
% N   = Number of basis functions (length of rcx, rcy, and rcz)
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

function [rcx, rcy, rcz, N] = user_get_OG_basis_center(OG_data)

Tp = OG_data.edges(1,:); Tm = OG_data.edges(2,:);	% T+ and T- triangles corresponding to vertex
pp = OG_data.cent(:,Tp);        % Centers of T+, 1xNe
pm = OG_data.cent(:,Tm);        % Centers of T-, 1xNe
rc = (pp+pm)/2;                 % Centers of basis functions

% Prepare output
rcx = rc(1,:); 
rcy = rc(2,:);
rcz = rc(3,:);
N = OG_data.N;

