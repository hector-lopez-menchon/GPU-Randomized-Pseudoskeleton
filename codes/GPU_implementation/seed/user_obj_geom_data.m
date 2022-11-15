% OG_data = user_obj_geom_data(EM_data)
% User function to prepare Object Geometry data
%
% Juan M. Rius et al., AntennaLab, Universitat Politecnica de Catalunya (Spain), v1.1, July 2008

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

function OG_data = user_obj_geom_data(EM_data)

%% BEGIN User configurable parameters

%% Pre-defibed objects

N = 3000;  % Desired number of basis functions. Actual number of basis functions will be as approximate as possible to N.
%N = 4*3000;  % Desired number of basis functions. Actual number of basis functions will be as approximate as possible to N.
%N = 4*4*3000;  % Desired number of basis functions. Actual number of basis functions will be as approximate as possible to N.
%N = 4*4*4*3000;  % Desired number of basis functions. Actual number of basis functions will be as approximate as possible to N. %Este es el que venía

%disc_size = 0.1*EM_data.lambda;    % Discretization size (lambdas). Usual discretization
disc_size = 0.05*EM_data.lambda;    % Discretization size (lambdas). Overdiscretization to give advantage to low-rank compression..

object = 'sphere';
% object = 'cube';
% object = 'sq_plate';


%% GiD mesh
% object = 'cylinder.msh';    % GiD mesh file in 'objects' subdirectory
%object = 'cylinder+strip.msh';    % GiD mesh file in 'objects' subdirectory

%% END user configurable parameters

if exist(['objects/' object],'file')
    view_var('GiD mesh file', object);
else
    view_var('Object',object);
    view_var('N target',N);
    view_var('Discretization size',disc_size);
end

%% Prepare geometry parameters
switch object
    case 'sphere'
        n = round(log(N/12)/log(4));
        Ne = 12*4^n;
        R = sqrt(disc_size^2 * Ne/48);
        view_var('Sphere radius',R);
        param = struct('R',R, 'Ne',Ne);

    case 'cube'
        n = round(log(N/12)/log(4));
        Nt = 12*4^n;
        L = 0.8*disc_size*sqrt(Nt/12);
        param = struct('L',L, 'Nt',Nt);

%		Lx = Size in x direction
%		Nx = Number of subdivisions in x
%		Lz = Size in z direction
%		Nz = Number of subdivisions in z
%		x  = x coordinate (default x=0)
%		y  = y coordinate (default y=0)
%		z  = z coordinate (default z=0)
%		cor = Flag for interior edge at corners:
%				0 -> 2 corners have interior edge, 2 don't have
%				1 -> 4 corners have interior edge (default

    case 'sq_plate'
        Nx = round(sqrt(N/1.5));
        Lx = 0.8*disc_size*Nx;
        param = struct('Lx',Lx, 'Lz',Lx, 'Nx',Nx, 'Nz',Nx, 'x',0, 'y',0, 'z',0, 'cor',1);

    otherwise
        if exist(['objects/' object],'file')
            param = object;
            object = 'gid_mesh';
        else
            error('Object geometry file not in objects folder');
        end
end

%% Compute object geometry data
cd objects
OG_data = feval(object,param);
OG_data = get_edge(OG_data);
OG_data.N = length(OG_data.ln);

%% Suggest integration radius suitable for this object
Rint_good = 1.5*(max(OG_data.ln)+mean(OG_data.ln))/2;
OG_data.name = object;
OG_data.Rint_good = Rint_good;
cd ..

view_var('A good Rint for this object is', Rint_good);

end

%% Local functions
%
% function obj=get_edge(obj);
% sets fields of RWG-object (type help object)
%
% input: obj with fields obj.vertex and obj.topol
%
% NOTE:
%
% this version allows edges with > 2 triangles (junctions):
% at junctions, extra triangles are created with the same topology as existing triangles.
% an extra field obj.junctions is created with pointers to the new triangles.
% the new triangles are not referred to in obj.edges, they serve only for the representation of the current.
%
%
% a.heldring, july 2005


function obj=get_edge(obj)

if isfield(obj,'junctions')
    tottrian = 1:size(obj.topol,2);
    nojunc = not(ismember(tottrian,obj.junctions));
    obj.topol = obj.topol(:,nojunc);
end

junctions = [];

obj.trian=[];
obj.edges=[];

% Check for duplicated vertices
%obj = check_duplicated_vertices(obj);% Not necessary if we know that there are none. Very expensive
%for large N.

% get vertices of each triangle
v1=obj.topol(1,:);
v2=obj.topol(2,:);
v3=obj.topol(3,:);
    
% nt = # of triangles

nt = length(v1);

% e = array of coupled vertices
% vv = array of 3rd vertex
% ov = position of third vertex (1,2 or 3)
% tt = array with corresponding triangles

t=1:nt; tt = [t,t,t];
e = [[v1,v2,v3];[v2,v3,v1]];
vv = [v3,v1,v2];
ov = [ones(1,nt)*3,ones(1,nt),ones(1,nt)*2];

e1 = sort(e,1);  % sort e w.r.t. smallest/largest index per pair
[e2,ie2] = sortrows(e1'); % sort e1 'alfabetically' w.r.t. first-then-second vertexc index
e3=(e2(2:end,:)-e2(1:end-1,:))'; % difference of e2
e4=[0,all(not(e3))]==1; % 1-D binary array: a '1' means that the corresponding pair in e2 is equal to its predecessor.
e5=[e4(2:end) 0]==1; % e4, shifted one position to the left.

vv1=vv(ie2(e4)); % first opposite vertices
vv2=vv(ie2(e5)); % second opposite vertices

tt1=tt(ie2(e4)); % first triangles
tt2=tt(ie2(e5)); % second triangles

ov1=ov(ie2(e4)); % first position-of-opp-vertices
ov2=ov(ie2(e5)); % second position-of-opp-vertices

ne = length(tt1);

% BEGIN JUNCTIONS
%
% now we have to find triangles that belong to two or more different basis functions
% defined by the same edge, and with the same 'opposite vertex':

to = [[tt1 tt2];[ov1 ov2]]; % all triangles (+ and -) and their corresponding opp-vert
[to2,ito2] = sortrows(to');
to3=(to2(2:end,:)-to2(1:end-1,:))'; % difference of to2
to4=[0,all(not(to3))]==1; % 1-D binary array: a '1' means that the corresponding pair in to2 is equal to its predecessor.
to5(ito2)=to4; % now to5 points at the 'repeated' pairs of trian/oppvert in to
top = to5(1:ne); % split into pos and neg again
tom = to5(ne+1:end);
tp = find(top);
tm = find(tom);

for t=tp % for all conflicting triangles in tt1
    ntnew = nt+1; % add a triangle at the end...
    obj.topol(:,ntnew) = obj.topol(:,tt1(t)); %... with the same topology as the conflicting triangle
    tt1(t) = ntnew; % and change the reference in tt1
    nt = ntnew;
    junctions = [junctions nt];
end
for t=tm % as above for the triangles in tt2
    ntnew = nt+1;
    obj.topol(:,ntnew) = obj.topol(:,tt2(t));
    tt2(t) = ntnew;
    nt = ntnew;
    junctions  = [junctions nt];
end

obj.junctions = junctions;

obj.edges(1,:) = tt1;
obj.edges(2,:) = tt2;
obj.edges(3,:) = vv1;
obj.edges(4,:) = vv2;

trianp = accumarray([ov1;tt1]',(1:ne),[3,nt]);
trianm = accumarray([ov2;tt2]',(1:ne),[3,nt]);

obj.trian = trianp - trianm;

% fill obj.ln

v1 = obj.vertex(:,e(1,ie2(e4)));
v2 = obj.vertex(:,e(2,ie2(e4)));
obj.ln = sum( (v1-v2).^2, 1).^0.5;

% fill obj.cent, obj.un and obj.ds

v1 = obj.vertex(:,obj.topol(1,:));
v2 = obj.vertex(:,obj.topol(2,:));
v3 = obj.vertex(:,obj.topol(3,:));

obj.cent =(v1+v2+v3)/3;
c  = cross(v3-v1,v2-v1);
obj.un = unitary(c);
obj.ds = sqrt(sum(c.^2))/2;

end

% un = unitary(x)
% Unit vector in the direction of vector x
% Arrays of vectors, 3 x N

function un = unitary(x)
    a_norm =sqrt(sum(x.^2));
    un =[x(1,:)./a_norm;
        x(2,:)./a_norm;
        x(3,:)./a_norm];
end

% obj = check_duplicated_vertices(obj)
% Check and fix duplicated vertices

function obj = check_duplicated_vertices(obj)

    disp('Checking for duplicated vertices....');

    Nv = size(obj.vertex,2);
    x = obj.vertex(1,:);
    y = obj.vertex(2,:);
    z = obj.vertex(3,:);

    xx = repmat(x,[Nv,1])-repmat(x',[1 Nv]);
    yy = repmat(y,[Nv,1])-repmat(y',[1 Nv]);
    zz = repmat(z,[Nv,1])-repmat(z',[1 Nv]);

    dd = (xx.^2 + yy.^2 + zz.^2).^0.5;

    dd = dd + diag(1000*ones(Nv,1));

    crit_vertices = ( norm([max(x)-min(x), max(y)-min(y), max(y)-min(y)]) )*1e-10;

    [ v1, v2 ] = find(dd < crit_vertices);

    dup_ver_referenced = 0;
    for n = 1:length(v2)/2
        if any(obj.topol(:)==v2(n)), dup_ver_referenced = 1;
        end
    end

    if ~isempty(v1)
        if dup_ver_referenced
            view_var('Duplicated vertices list:', [v1 v2]);
            disp('Fixing duplicated vertices...');

            % Replace references to v2 by references to v1
            for n = 1:length(v2)/2
                obj.topol( obj.topol(:)==v2(n) ) = v1(n);
            end
            disp('Duplicated vertices fixed');
        else
            disp('There are duplicated vertices but they are not referenced');
        end
    else
        disp('There are no duplicated vertices');
    end

end

% obj = gid_mesh(p_gid)
%
% Input:
% p_gid: struct with the following field
%		file = GID mesh file
%
% Output: see object.m
% obj = object struct, with fields vertex,topol and Ng
%
% Notes:
% Node numbers must begin with 1 and be correlative
% Mesh must contain only triangular elements
% The vertex matrix begins after the keyword 'Coordinates'
% The topology matrix begins after the keyword 'Elements'
%
% Juan M. Rius, Josep Parron June 1999

function obj = gid_mesh(fich)

if ~ischar(fich) || ~exist(fich,'file'), error('Argument of gid_mesh must be an existing file name');
end

fid = fopen(fich);
if fid==-1, error('Cannot open file %s',fich);
end

obj = struct('vertex',[],'topol',[],'trian',[],'edges',[],'un',[],'ds',[],'ln',[],'cent',[],'feed',[]);

tmp = fgetl(fid);

if ~findstr('Triangle',tmp) || ~findstr('Nnode 3',tmp)
    error('GiD mesh file must contain only 3-node triangle elements');
end

while ~strcmp(tmp,'Coordinates')
   tmp = fgetl(fid);
end
tmp = fscanf(fid,'%f',inf);
tmp = reshape(tmp,4,length(tmp)/4);
if any( tmp(1,:)~=(1:size(tmp,2)) ), error('Node numbers are not correlative');
end
obj.vertex = tmp(2:4,:);

tmp = fgetl(fid);
while ~strcmp(tmp,'Elements')
   tmp = fgetl(fid);
end
tmp = fscanf(fid,'%f',inf);
tmp = reshape(tmp,4,length(tmp)/4);
obj.topol = tmp(2:4,:);

fclose(fid);

end
