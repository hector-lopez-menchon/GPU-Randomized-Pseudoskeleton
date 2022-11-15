% function obj=get_edge(obj);
% sets fields of RWG-object (type help object)
%
% input: obj with fields obj.vertex and obj.topol
%
% NOTE:
%
% this version allows edges with > 2 triangles (junctions): 
% at junctions, extra triangles are created with the same topology as existing triangles. 
% an extra field obj.junctions is created with pointers to the new
% triangles.
%
% a.heldring, july 2005


function obj=get_edge(obj)
    
if isfield(obj,'junctions'), tottrian = 1:size(obj.topol,2); nojunc = not(ismember(tottrian,obj.junctions)); obj.topol = obj.topol(:,nojunc); end

junctions = [];

obj.trian=[];
obj.edges=[];

% Check for duplicated vertices
%obj = check_duplicated_vertices(obj);

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

for t=tp, % for all conflicting triangles in tt1 
    ntnew = nt+1; % add a triangle at the end...
    obj.topol(:,ntnew) = obj.topol(:,tt1(t)); %... with the same topology as the conflicting triangle
    tt1(t) = ntnew; % and change the reference in tt1
    nt = ntnew;
    junctions = [junctions nt];
end
for t=tm, % as above for the triangles in tt2
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





