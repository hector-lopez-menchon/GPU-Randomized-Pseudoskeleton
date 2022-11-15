% Ei = user_compute_Ei(OG_data, EM_data)
% User function to compute incident field (linear system independent vector)
%
% OG_data   = struct containing Object Geometry data 
% EM_data   = struct containing ElectroMagnetic data 
%
% Output arguments:
% Ei = column vector containing the incident field weighted by MoM testing functions
%
% Juan M. Rius, AntennaLab, Universitat Politecnica de Catalunya (Spain), v1.1, July 2008

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

function [Ei, OG_data] = user_compute_Ei(OG_data, EM_data)

%% BEGIN Parameters
% Incident plane wave direction in deg
theta = 90; phi = -90;

% Electric field angle with respect to xy plane
rot_EH = 90;

%% END Parameters

k = EM_data.k;
th = theta*pi/180; ph = phi*pi/180; rot_EH = rot_EH*pi/180;

k_i = -[sin(th)*cos(ph); sin(th)*sin(ph); cos(th)];

e_i = cos(rot_EH)*[-sin(ph); cos(ph); 0] + sin(rot_EH)*[cos(th)*cos(ph); cos(th)*sin(ph); -sin(th)];

Tp = OG_data.edges(1,:); Tm = OG_data.edges(2,:);	% T+ and T- triangles corresponding to vertex
pp = exp(-j*k* k_i(:).'*OG_data.cent(:,Tp));	% Plane wave at cent of T+, 1xNe
pm = exp(-j*k* k_i(:).'*OG_data.cent(:,Tm));	% Plane wave at cent of T-, 1xNe

rp = OG_data.cent(:,Tp) - OG_data.vertex(:,OG_data.edges(3,:));	% Rho of center in T+
rm = OG_data.vertex(:,OG_data.edges(4,:)) - OG_data.cent(:,Tm);	% Rho of center in T-

Ei = OG_data.ln .* (sum((e_i(:)*pp).*rp) + sum((e_i(:)*pm).*rm)) /2; Ei = Ei.';
