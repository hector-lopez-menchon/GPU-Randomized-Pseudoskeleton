% [obj] = cube(p_cube)
%
% Input:
% p_cube = struct with the following fields
%		L 	= edge (meters)
%		Nt	= Number of triangles. Must be Nt = 12 * 4^n, n integer
%				Number of edges generated will be 1.5*Nt = 18 * 4^n
%
% Output: see object.m
% obj = object struct, with fields vertex,topol and Ng
%
% Juan M. Rius, Josep Parron June 1999

function [obj] = cube(p_cube)

obj = struct('vertex',[],'topol',[],'trian',[],'edges',[],'un',[],'ds',[],'ln',[],'cent',[],'feed',[],'Ng',0);

L = p_cube.L; Nt = p_cube.Nt;
Ng = 0;

n = log(Nt/12)/log(4);
%view_var('n',n);
if floor(n)~=n, error('Must be Nt = 12 x 4^n, n integer'); end
%view_var('Mesh size',L/(2^n));

vertex = L/2 * [  1  1  1  1 -1 -1 -1 -1; % Cartesian coordinates of vertex
                 -1  1  1 -1  1  1 -1 -1;
                  1  1 -1 -1  1 -1  1 -1];

topolin = [ 1 2 1 1 1 2 2 2 5 6 3 3;
		      2 3 4 8 7 7 5 6 7 7 6 8;
		      4 4 8 7 2 5 6 3 6 8 8 4];

Nt = 12; Nv = 8;
topolout = topolin;      
for it = 1:n,		% Divide by 2 mesh size
	for t = 1:Nt,		% For each triangle
		v1 = topolin(1,t); v2 = topolin(2,t); v3 = topolin(3,t);
		r12 = (vertex(:,v1) + vertex(:,v2))/2;
		r23 = (vertex(:,v2) + vertex(:,v3))/2;
		r31 = (vertex(:,v3) + vertex(:,v1))/2;

		% Check if new vertex already exists
		e12 = find(all([r12(1)==vertex(1,:); r12(2)==vertex(2,:); r12(3)==vertex(3,:)]));
		e23 = find(all([r23(1)==vertex(1,:); r23(2)==vertex(2,:); r23(3)==vertex(3,:)]));
		e31 = find(all([r31(1)==vertex(1,:); r31(2)==vertex(2,:); r31(3)==vertex(3,:)]));
		if e12, v12 = e12; else v12 = Nv+1; Nv = Nv+1; vertex = [vertex r12]; end
		if e23, v23 = e23; else v23 = Nv+1; Nv = Nv+1; vertex = [vertex r23]; end
		if e31, v31 = e31; else v31 = Nv+1; Nv = Nv+1; vertex = [vertex r31]; end

		topolout(:,4*t-3) = [v12; v23; v31]; % Replace current triangle
		topolout(:,4*t-2) = [v2 ; v23; v12];
	      topolout(:,4*t-1) = [v23; v3 ; v31];
      	topolout(:,4*t)   = [v31; v1 ; v12];
	end
	topolin=topolout;
	Nt = Nt*4;
end

obj.topol = topolout;
obj.vertex = vertex;
obj.Ng = Ng;

