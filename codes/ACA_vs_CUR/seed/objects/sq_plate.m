% [obj] = sq_plate(p_sqpl)
% Geometry of square plate in XZ plane
% 
% Input:
% p_sqpl = struct with the following fields
%		Lx = Size in x direction
%		Nx = Number of subdivisions in x
%		Lz = Size in z direction
%		Nz = Number of subdivisions in z
%		x  = x coordinate (default x=0)
%		y  = y coordinate (default y=0)
%		z  = z coordinate (default z=0)
%		cor = Flag for interior edge at corners:
%				0 -> 2 corners have interior edge, 2 don't have
%				1 -> 4 corners have interior edge (default)
%
% Output: see object.m
% object = object struct
%
% Juan M. Rius, Josep Parron June 1999


function [obj] = sq_plate(p_sqpl)

obj = struct('vertex',[],'topol',[],'trian',[],'edges',[],'un',[],'ds',[],'ln',[],'cent',[],'feed',[],'Ng',0);

Lx = p_sqpl.Lx; Nx = p_sqpl.Nx;
Lz = p_sqpl.Lz; Nz = p_sqpl.Nz;
x = p_sqpl.x;
y = p_sqpl.y;
z = p_sqpl.z;
cor = p_sqpl.cor;

Nx1 = floor(Nx/2);
Nz1 = floor(Nz/2);

% Discretization of plate sides
arx = (linspace(x-Lx/2,x+Lx/2, Nx+1)); % Nx funcions base
arz = (linspace(z-Lz/2,z+Lz/2, Nz+1)); % Nz funcions base

% Discretization of plate surface
[c1 c2] = meshgrid(arx,arz);
zax = c1(:).';		% 3 row vectors of vertex coordinates
zay = y*ones(1,(Nx+1)*(Nz+1));
zaz = c2(:).';

% Vertex matrix
vertex = [zax; zay; zaz];

% Topology matrix: vertex for each triangle
topol = [];

if ~cor,	% Division in quadrants not necessary
    for n=1:Nx,
        for r=1:Nz,
            aux = [	(n-1)*(Nz+1)+r  (n-1)*(Nz+1)+r+1 n*(Nz+1)+r;
                    (n-1)*(Nz+1)+r+1  n*(Nz+1)+r+1 n*(Nz+1)+r].';
            topol = [topol aux];
        end;
    end
else
	% 1st quadrant
    for n=1:Nx1,
        for r=1:Nz1,
            aux = [	(n-1)*(Nz+1)+r  n*(Nz+1)+r+1  n*(Nz+1)+r;
                (n-1)*(Nz+1)+r  (n-1)*(Nz+1)+r+1 n*(Nz+1)+r+1].';
            topol = [topol aux];
        end;
    end

	% 2nd quadrant
    for n=1:Nx1,
        for r=Nz1+1:Nz,
            aux = [	(n-1)*(Nz+1)+r  (n-1)*(Nz+1)+r+1 n*(Nz+1)+r;
                (n-1)*(Nz+1)+r+1  n*(Nz+1)+r+1 n*(Nz+1)+r].';
            topol = [topol aux];
        end;
    end

	% 3rd quadrant
    for n=Nx1+1:Nx,
        for r=1:Nz1,
            aux = [	(n-1)*(Nz+1)+r  (n-1)*(Nz+1)+r+1 n*(Nz+1)+r;
                (n-1)*(Nz+1)+r+1  n*(Nz+1)+r+1 n*(Nz+1)+r].';
            topol = [topol aux];
        end;
    end

	% 4th quadrant
    for n=Nx1+1:Nx,
        for r=Nz1+1:Nz,
            aux = [	(n-1)*(Nz+1)+r  n*(Nz+1)+r+1  n*(Nz+1)+r;
                (n-1)*(Nz+1)+r  (n-1)*(Nz+1)+r+1 n*(Nz+1)+r+1].';
            topol = [topol aux];
        end;
    end
end

%For test horizonzal currents
%aux=vertex(2,:);
%vertex(2,:)=vertex(3,:);
%vertex(3,:)=aux;


obj.topol = topol;
obj.vertex = vertex;



