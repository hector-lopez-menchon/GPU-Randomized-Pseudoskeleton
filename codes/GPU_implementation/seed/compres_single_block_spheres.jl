#En este compress_single_block_sphere.jlk vamos a aestudiar los timepos de compresion de un sólo bloque de interacción entre dos esferas. Está basado en compress_main_julia_getZ_almond.jl 

#En este compress_main_julia_getZ_almond.jl, copiado de main_julia_getZ.jl, vamos a hacer la misma prueba en Julia del algoritmo de compresión, pero con los datos de la almendra

#En este main_julia_getZ, copaido de main_julia_2.jl en la carpoeta más intern, vamos a tratar de generar la misma matriz para matlab y julia
#
#

#Lo mio de Julia

using LinearAlgebra
#using Plots
using JLD
push!(LOAD_PATH,"./")
using MatlabFiles
using MATLAB
using StatsBase


mutable struct obj_struct
	vertex
	topol
	trian
	edges
	un
	ds
	ln
	cent
	feed
	Ng
	junctions
	N
	bj
	name
end

mutable struct EM_data_struct 
	lambda
	k
	eta
	field
	Rint_s
	Rint_f
	Ranal_s
	corr_solid
	flag
end

struct param_struct
	R
	Ne
end






function unitary(x)

	# un = unitary(x)
	# Unit vector in the direction of vector x
	# Arrays of vectors, 3 x N


	a_norm = sqrt.(sum(x.^2, dims = 1))
	#println("size(a_norm) = ", size(a_norm))

	return [x[1:1,:]./a_norm; x[2:2,:]./a_norm; x[3:3,:]./a_norm]

end

function cross_matrixwise(A,B)
	#This function emulates the behavior of matlab cross function
	#We assume that A and B have the same dimensions.
	#We assume the number of lines is 3, so we can apply cross
	C = Array{Any}(undef,size(A,1),size(A,2))

	for ii = 1:size(A,2)
		C[:,ii] = cross(A[:,ii],B[:,ii]) 
	end

	return C

end

function meshgridPseudoMatlab(a,b)
         #We assume that a and b are n-element vectors
        X = repeat(transpose(a),length(b))
        Y = collect(transpose(repeat(transpose(b),length(a))))
        return X, Y
end

function vectorized_indices(inds::Array{CartesianIndex{2},1})
        X = zeros(length(inds),1)
        Y = zeros(length(inds),1)
        for i=1:length(inds); X[i]=inds[i][1]; Y[i]=inds[i][2];end
        return X, Y
end


function test_fields(obj::obj_struct, theta_i, phi_i, rot_EH_i, k, eta)
	th = theta_i*pi/180; ph = phi_i/180; rot_EH = rot_EH_i*pi/180

	k_i = -[sin(th)*cos(ph); sin(th)*sin(ph); cos(th)]
	e_i = cos(rot_EH)*[-sin(ph); cos(ph); 0.0] +sin(rot_EH)*[cos(th)*cos(ph); cos(th)*sin(ph); -sin(th)]
	h_i = cross(k_i,e_i)

	Tp = obj.edges[1,:]; Tm = obj.edges[2,:] # T+ and T- triangles correspoding to vertex
	Tp = trunc.(Int,Tp); Tm = trunc.(Int,Tm)

	pp = exp.(-im*k*sum(k_i.*obj.cent[:,Tp],dims=1)) # Plane wave at cent of T+, 1xNe
	pm = exp.(-im*k*sum(k_i.*obj.cent[:,Tm],dims=1)) # Plane wave at cent of T-, 1xNe

	rp = obj.cent[:,Tp] - obj.vertex[:,trunc.(Int,obj.edges[3,:])] # Rho of center in T+
	rm = obj.vertex[:,trunc.(Int,obj.edges[4,:])] - obj.cent[:,Tm]
	
	Ei = transpose(obj.ln).*(sum((e_i.*pp).*rp,dims=1) + sum((e_i.*pm).*rm,dims=1))/2
	Ei = transpose(Ei)


	Hi = transpose(obj.ln).*(sum((h_i*pp).*cross_matrixwise(rp,obj.un[:,Tp]),dims=1) + sum((h_i*pm).*cross_matrixwise(rm,obj.un[:,Tm]),dims=1))/(2*eta)
	Hi = transpose(Hi)

	return collect(Ei), collect(Hi)

end


function sphere(p_sphere::param_struct)::obj_struct
		
	# Input:
	# p_sphere = struct with the following fields
	#               R       = radius (meters)
	#               Ne      = Number of edges = 12 x 4^n, n integer
	#
	# Output: see object.m
	#

	Ng = 0
	R = p_sphere.R
	Ne = p_sphere.Ne

	n = log(Ne/12)/log(4)
	
	#Check if Ne = 12 x 4^n
	if floor(n) != n 
		error("Ne must be = 12 * 4^n")
	end

	#Cartesian coordinates of obj.vertex
	vertex_vec = R*[0 1 0 -1 0 0; 0 0 1 0 -1 0; 1 0 0 0 0 -1 ]
	topol_vec  = [ 1 1 1 1 6 6 6 6; 3 4 5 2 2 3 4 5; 2 3 4 5 3 4 5 2]
	#topol_vec = convert(Array{Int32,2},topol_vec) #We want it to be Int32 instead of 64
	
	Nt = 8
	Nv = 6
	aux_ones = ones(Int64,3,1)

	#Divide by  2 mesh size
	for it = 1:n
		#For each triangle
		for t = 1:Nt
			v1 = topol_vec[1,t]; v2 = topol_vec[2,t]; v3 = topol_vec[3,t]
			r12 = (vertex_vec[:,v1] + vertex_vec[:,v2])/2; r12 = r12*R/norm(r12)
			r23 = (vertex_vec[:,v2] + vertex_vec[:,v3])/2; r23 = r23*R/norm(r23)
			r31 = (vertex_vec[:,v3] + vertex_vec[:,v1])/2; r31 = r31*R/norm(r31)

			e12 = findall((r12[1].==vertex_vec[1,:]).*(r12[2].==vertex_vec[2,:]).*(r12[3].==vertex_vec[3,:]))
			e23 = findall((r23[1].==vertex_vec[1,:]).*(r23[2].==vertex_vec[2,:]).*(r23[3].==vertex_vec[3,:]))
			e31 = findall((r31[1].==vertex_vec[1,:]).*(r31[2].==vertex_vec[2,:]).*(r31[3].==vertex_vec[3,:]))

			#For r12
			#Check if the object we have created already exists
			if (length(e12)==0)
				#It does not exist
				v12 = Nv+1; Nv = Nv+1; vertex_vec = hcat(vertex_vec,r12)
			elseif (length(e12)==1)
				#If this already exists
				v12 = e12[1]	
			else
				#If it exists twice or more
				error("Topology error: a triangle appears at least twice in the topology field")
			end
				
			#For r23
			#Check if the object we have created already exists
			if (length(e23)==0)
				#It does not exist
				v23 = Nv+1; Nv = Nv+1; vertex_vec = hcat(vertex_vec,r23)
			elseif (length(e23)==1)
				#If this already exists
				v23 = e23[1]	
			else
				#If it exists twice or more
				error("Topology error: a triangle appears at least twice in the topology field")
			end
					
			#For r31
			#Check if the object we have created already exists
			if (length(e31)==0)
				#It does not exist
				v31 = Nv+1; Nv = Nv+1; vertex_vec = hcat(vertex_vec,r31)
			elseif (length(e31)==1)
				#If this already exists
				v31 = e31[1]	
			else
				#If it exists twice or more
				error("Topology error: a triangle appears at least twice in the topology field")
			end

			#Replace the current triangle
			topol_vec[:,t] = [v12; v23; v31]
			topol_vec = hcat(topol_vec,[[v2; v23; v12] [v23; v3; v31] [v31; v1; v12]])

		end
		Nt *= 4
	end

	#Build the object to return 
	#Faltaria segurament el campo de nombre
	#return obj_struct(vertex_vec,topol_vec,zeros(3,0),zeros(4,0),zeros(3,0),zeros(1,0),zeros(1,0),zeros(3,0),Any,Any,Any,Any,"sphere")

	return obj_struct(vertex_vec,convert(Array{Int32,2},topol_vec), nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, "sphere")

end

function get_edge_jmr(obj::obj_struct)::obj_struct
	#[obj] = get_edge(obj)
	#
	# Input:
	# topol = topology matrix (vertex of each triangle), 3 x Nt
	# vertex = vertex matrix, 3 x Nv
	#
	# Output:
	# edges = Edges matrix, 4 x Ne. For each edge (column):
	#               Row 1: Triangle T+
	#               Row 2: Triangle T-
	#               Row 3: Global number of opposite vertex in T+
	#               Row 4: Global number of opposite vertex in T-
	# ln    = Length of edges, 1 x Ne
	# trian = triangles matrix. For each triangle (column):
	#               Row 1: Edge 1 (opposite is vertex 1)
	#               Row 2: Edge 2 (opposite is vertex 2)
	#               Row 3: Edge 3 (opposite is vertex 3)
	#               If >0, T+ for that edge; if <0, T- for that edge
	#               If==0, it is edge in open boundary
	# un    = Unit normal to each triangle, 3 x Nt
	# ds    = Area of each triangle,        1 x Nt
	# cent  = Centroid of each triangle,    3 x Nt
	#
	#

	topol = obj.topol
	vertex = obj.vertex
	
	v1 = vertex[:,topol[1,:]]
	v2 = vertex[:,topol[2,:]]
	v3 = vertex[:,topol[3,:]]

	#println("size(v3) = ", size(v3))

	cent = (v1+v2+v3)/3

	c = cross_matrixwise(v3-v1,v2-v1)
	un = unitary(c)
	#csquare = c.^2
	#sumilla = sum(csquare,dims=1)
	#ds = sqrt(sumilla)/2
	ds = sqrt.(sum(c.^2, dims=1))/2

	Nt = length(ds)
	trian = zeros(Int32,3,Nt)
	edges = zeros(Int32,4,ceil(Int, Nt*3/2))

	ln = zeros(0)
	ver = nothing #Si no lo defino aqui pasan cosas curiosas: comrpbar con miniejemplo

	eg = 1				#Global number of current edge
	for Tp = 1:Nt                   #Triangle T+
					# [3 2 1], local edge, this order for compatibility
		for el = [3 2 1]    
			#vprintln("Iteration: Tp = ",Tp," el = ", el)
			if trian[el,Tp]==0  # Edge not found yet
				#println("Iteration: Tp = ",Tp," el = ", el)
				#println("Entered the  if")
				# Find vertex of this edge
				ver = topol[findall(x-> x!=el,[1;2;3]),Tp]
				if el == 2; ver = reverse(ver); end
				#end #Revisar esta estructura condicional en el codigo original

				#println("display ver = ",ver)	
				(tmp,T1) = vectorized_indices(findall(x->x==ver[1],topol)) #T1 = triangles that have ver[1]
				(tmp,T2) = vectorized_indices(findall(x->x==ver[2],topol)) #T2 = triangles that have ver[2]

				#Note: this could be dine as vectorized_indices(findall(ver[1] .== topol))

				(TT1,TT2) = meshgridPseudoMatlab(T1,T2)
				#println("size(T1) = ",size(T1), "size(T2) = ", size(T2))
				Tcom = TT1[TT1.==TT2] # Triangles with common edge
				#println("Tcom = ", Tcom)

				if length(Tcom)==2
					trian[el,Tp] = eg
					#println("eg = ",eg," size(edges) = ", size(edges))

					edges[1,eg] = Tp 			# T+
					edges[3,eg] = topol[el,Tp]

					Tm = Tcom[findall(Tcom .!= Tp)] 	# T-, not equal to Tp

					#Nota, todo esto tan feo viene de la inconsistencia de tipos. Hay que definir como enteros aquello que son enteros
					Tm = trunc(Int, Tm[1])


					#println("Printo Tm ", Tm)

					edges[2,eg] = Tm 

					#Find the local number of eg in T-

					v = findall((topol[:,Tm] .!= ver[1]) .& (topol[:,Tm] .!= ver[2]))[1]
					#println(v)
					trian[v,Tm] = -eg
					edges[4,eg] = topol[v,Tm]
					#append!(ln) NO ES CORRECTO
					

					append!(ln,norm(vertex[:,ver[1]]-vertex[:,ver[2]]))
					#Nota, este append da un poco de miedo, aqui se podria usar sizehint
					eg = eg+1
				elseif length(Tcom) > 2
					#error("More than 2 triangles share an edge")
				end
			end
		end

	end

	edges = edges[:,1:(eg-1)] #Remove void edges, in case of opne object

	obj.edges = edges
	obj.trian = trian
	obj.ln    = ln
	obj.un    = un
	obj.ds    = ds
	obj.cent  = cent

	return obj

end

function twin_spheres!(obj::obj_struct, displacement::Float64)
	#This creates two twin spheres from a single one 
	#Assuming the displacement is in x
	new_vertex = obj.vertex[:,:]
	new_vertex[1,:] = new_vertex[1,:].+displacement
	new_topol = obj.topol[:,:]
	new_topol = new_topol.+findmax(new_topol)[1]
	
	obj.vertex = hcat(obj.vertex,new_vertex)
	obj.topol = hcat(obj.topol,new_topol)
	
	return 
end


#Note, this function translation is not finished, we focus on get_edge_jmr, because we don't need to work with junctions
#function get_edge(obj::obj_struct)

#	#IMPORTANT: the translation of this function from its MATLAB original is not complet. It is foused on a particular case
#
#
#	#=
#	These are notes for developing purposes. In the original MATLAB version, the object is defined in such a way that fields as "junctions" are added dynamically, so here it makes sense to check if it exists or not, as in the code just below. However, in Julia, for theoretical reasons the fields of a struct cannot change (although we can modify their value). The reason is that, if it was possible to modify the fields in this way, we would have a dictionary instead of a struct. Our solution is to define the all the fields from the very beginning and making their value = nothing if necessary
#	=#
#
#	#if is(:junctions,fieldnames(typeof(obj)))
#		#Code
#	#end
#
#	#Be very careful, it is possible that this piece of code will never execute. So it may be wrong without us realizing
#	if obj.junctions != nothing
#		tottrian = 1:size(obj.topol,2)
#		nojunc = false #ESTO ESTA MAL PERO NO SE VA A EJECUTAR DE MOMENTO
#		obj.topol = 0 #ESTO ESTA MAL PERO NO SE VA A EJECUTAR DE MOMENTO
#	end
#
#	obj.junctions = zeros()
#
#
#	return obj
#end
#

function plot_object(obj::obj_struct)

	#This plotting is an utterly naive implementation. This is not industrial code at all!

	topol = obj.topol
	vertex = obj.vertex
	plotlyjs()  #This is the BAD LINE. Comment it and maybe things will be fixed

	xmat = [vertex[1:1,topol[1,:]]; vertex[1:1,topol[2,:]]; vertex[1:1,topol[3,:]]; vertex[1:1,topol[1,:]]]
	ymat = [vertex[2:2,topol[1,:]]; vertex[2:2,topol[2,:]]; vertex[2:2,topol[3,:]]; vertex[2:2,topol[1,:]]]
	zmat = [vertex[3:3,topol[1,:]]; vertex[3:3,topol[2,:]]; vertex[3:3,topol[3,:]]; vertex[3:3,topol[1,:]]]

	println("Plotting: if the plot does not work, try again or restart julia. If it keeps failing go to the function plot_object in the main_julia.jl file and comment the \" bad line \". But wait some time before, it usually takes long. \n ")

	nn = size(obj.topol,2)

	for ii = 1:nn-1
		plot3d!(xmat[:,ii],ymat[:,ii],zmat[:,ii],marker = 2, color = "blue", show = false)
	end
	
	#return plot3d!(xmat[:,nn],ymat[:,nn],zmat[:,nn],marker = 2, color = "blue", show = true)
	return plot3d!(xmat[:,nn],ymat[:,nn],zmat[:,nn],marker = 2, color = "blue", show = true)

end

#############
#Electromagnetic functions
function calling_c_code_example(r1::UnitRange{Int64},r2::UnitRange{Int64},obj::obj_struct,EM_data::EM_data_struct)
	#This function is a wrapper for the C code
	r1 = collect(Cint,r1) #r1 = convert(Array{Cint,1},r1) #A vector of ints 32
	r2 = collect(Cint,r2) # r2 = convert(Array{Cint,1},r2) #A vector of ints 32
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cint,2},obj.topol) #An array of ints 32
	trian = convert(Array{Cint,2},obj.trian) # An array of ints 32
	edges = convert(Array{Cint,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string


	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)


	r1_copy = Array{Int32}(undef,length(r1))
	r2_copy = Array{Int32}(undef,length(r1))
	vertex_copy = Array{Cdouble,2}(undef,rows_vertex,cols_vertex)
	topol_copy = Array{Cint,2}(undef,rows_topol,cols_topol)
	trian_copy = Array{Cint,2}(undef,rows_trian,cols_trian)
	edges_copy = Array{Cint,2}(undef,rows_edges,cols_edges)
	un_copy = Array{Cdouble,2}(undef,rows_un,cols_un)
	ds_copy = Array{Cdouble,2}(undef,rows_ds,cols_ds)
	ln_copy = Array{Cdouble,1}(undef,n_ln)
	cent_copy = Array{Cdouble,2}(undef,rows_cent,cols_cent)
	N_copy = convert(Cint,0)



	#Deconstruct the obj obj_struct
	#res = ccall((:copy_everything,"./funprueba.so"),Int64,(Ptr{ComplexF32},Cint,Cint),x,N,N)
	#res = ccall((:copy_r1, "./funsPrueba.so"),Int64,(Ptr{Cint},Ptr{Cint},Cint),r1_copy,r1,length(r1))
	res = ccall((:copy_r1, "./funsPrueba.so"),
		    Int64,
		    (Ptr{Cint},Ptr{Cint},Cint),
		    r1_copy,
		    r1,
		    length(r1))

	res = ccall((:copy_everything,"./funsPrueba.so"),
		    Int64,
		    (Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Cint,   Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Cint,   Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
		    r1_copy,
		    r2_copy,
		    vertex_copy,
		    topol_copy,
		    trian_copy,
		    edges_copy,
		    un_copy,
		    ds_copy,
		    ln_copy,
		    cent_copy,
		    N_copy,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent)


	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))



	return cent_copy
end

function user_impedance(r1::UnitRange{Int64},r2::UnitRange{Int64},obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
	#This function is a wrapper for the C code

	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)

	#The matrix where we will write the result
	Zr = zeros(Cdouble,N,N)
	Zi = zeros(Cdouble,N,N)

	#Call the external C code
	res = ccall((:impedance_matrix,"./functionsExternal.so"),
		    Int64,
		    (Ptr{Cdouble},Ptr{Cdouble},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
		    Zr,
		    Zi,
		    field,
		    k,
		    eta,
		    Rinteg_s,
		    Ranal_s,
		    Rinteg_f,
		    cor_solid,
		    flag,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent)


	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))



	return Zr+im*Zi
end

function user_impedance2(r1::UnitRange{Int64},r2::UnitRange{Int64},obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
	#This function is a wrapper for the C code
	#In this user_impedance2 we try to solve the proble,m of the previous one, which doesn't allow to compute arbitrary elements of the matrix

	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)

	#The matrix where we will write the result
	Zr = zeros(Cdouble,n_r1,n_r2) #OJO: This is what we fix wrt to the previous version
	Zi = zeros(Cdouble,n_r1,n_r2)

	#Call the external C code
	res = ccall((:impedance_matrix,"./functionsExternal.so"),
		    Int64,
		    (Ptr{Cdouble},Ptr{Cdouble},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
		    Zr,
		    Zi,
		    field,
		    k,
		    eta,
		    Rinteg_s,
		    Ranal_s,
		    Rinteg_f,
		    cor_solid,
		    flag,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent)


	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))



	return Zr+im*Zi
end

function user_impedance3(r1,r2,obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
	#Este user_impedance3 recibe vector en vez de ranges
	#This function is a wrapper for the C code
	#In this user_impedance2 we try to solve the proble,m of the previous one, which doesn't allow to compute arbitrary elements of the matrix

	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)

	#The matrix where we will write the result
	Zr = zeros(Cdouble,n_r1,n_r2) #OJO: This is what we fix wrt to the previous version
	Zi = zeros(Cdouble,n_r1,n_r2)

	#Call the external C code
	res = ccall((:impedance_matrix,"./functionsExternal.so"),
		    Int64,
		    (Ptr{Cdouble},Ptr{Cdouble},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
		    Zr,
		    Zi,
		    field,
		    k,
		    eta,
		    Rinteg_s,
		    Ranal_s,
		    Rinteg_f,
		    cor_solid,
		    flag,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent)


	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))



	return Zr+im*Zi
end



function user_impedance_improved(r1::UnitRange{Int64},r2::UnitRange{Int64},obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
	#Este improved es el paso previo a lo que haremos a continuación con la GPU
	#This function is a wrapper for the C code
	#In this user_impedance2 we try to solve the proble,m of the previous one, which doesn't allow to compute arbitrary elements of the matrix

	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)

	#The matrix where we will write the result
	Zr = zeros(Cdouble,n_r1,n_r2) #OJO: This is what we fix wrt to the previous version
	Zi = zeros(Cdouble,n_r1,n_r2)

	#Call the external C code
	#Notas de prueba:
	#impedance_matrix: el bueno
	#impedance_matrix_wrapper: el de las pruebas de wrapper
	#impedance_matrix_parts: el inicio del paralelismo
	res = ccall((:impedance_matrix_parts,"./functionsExternal_improved.so"),
		    Int64,
		    (Ptr{Cdouble},Ptr{Cdouble},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
		    Zr,
		    Zi,
		    field,
		    k,
		    eta,
		    Rinteg_s,
		    Ranal_s,
		    Rinteg_f,
		    cor_solid,
		    flag,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent)


	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))



	return Zr+im*Zi
end


function user_impedance_pseudoparallel(r1::UnitRange{Int64},r2::UnitRange{Int64},obj::obj_struct,EM_data::EM_data_struct,chunk_size)::Array{Complex{Float64},2}
	#return user_impedance(r1,r2,obj,EM_data)
	N = convert(Cint,obj.N) # An int 32
	#The matrix where we will write the result
	#Zr = zeros(Cdouble,N,N)
	#Zi = zeros(Cdouble,N,N)
	Zmat = zeros(Complex{Float64},N,N)
	num_chunks = div(N,chunk_size)
	surplus = rem(N,chunk_size)
	for ii=1:num_chunks
		Zhere = user_impedance2(r1,r2[(1+(ii-1)*chunk_size):(ii*chunk_size)],obj,EM_data)
		#println("size(Zhere) = ",size(Zhere))
		Zmat[:,(1+(ii-1)*chunk_size):(ii*chunk_size)] = Zhere
	end
	Zhere = user_impedance2(r1,r2[(num_chunks*chunk_size+1):end],obj,EM_data)
	Zmat[:,(num_chunks*chunk_size+1):end] = Zhere
	return Zmat
end


function solve_by_CBF(Z,v)
	#Here, we solve using the Characteristic Basis function technique, assuming that the matrix has only 4 blocks
	#Comprobar que con esto de los bloques no estemos haciendo algo erroneo (particular sistema de punteros de Julia)
	nblock = div(size(Z,1),2)
	Z11 = Z[1:nblock,1:nblock]; Z12 = Z[1:nblock,(nblock+1):end]; Z21 = Z[(nblock+1):end,1:nblock]; Z22 = Z[(nblock+1):end,(nblock+1):end]
	#Z11 = @view Z[1:nblock,1:nblock]; Z12 = @view Z[1:nblock,(nblock+1):end]; Z21 = @view Z[(nblock+1):end,1:nblock]; Z22 = @view Z[(nblock+1):end,(nblock+1):end]
	v1 = v[1:nblock]; v2 = v[(nblock+1):end]
	
	#Compute CBFs, primary and secondary
	fp1 = Z11\v1
	fp2 = Z22\v2
	fs12 = Z11\(Z12*fp2)
	fs21 = Z22\(Z21*fp1)

	#Compute Macro Basis Functions
	Q1 = [fp1 fs12]
	Q2 = [fp2 fs21]
	Q1_adj = Adjoint(Q1)
	Q2_adj = Adjoint(Q2)

	#Compute compressed matrix and vector
	#Z11_comp = Q1_adj*(Z11*Q1)
	Z_compressed = [Q1_adj*(Z11*Q1) Q1_adj*(Z12*Q2); Q2_adj*(Z21*Q1) Q2_adj*(Z22*Q2)]
	v_compressed = [Q1_adj*v1; Q2_adj*v2]
	
	#Solve the system
	@time x_compressed = Z_compressed\v_compressed

	#Computing result in the original basis

	x = x_compressed[1]*[fp1;zeros(nblock)]+x_compressed[2]*[fs12;zeros(nblock)]+x_compressed[3]*[zeros(nblock);fp2]+x_compressed[4]*[zeros(nblock);fs21]
	compression_rate = length(Z)/length(Z_compressed)
	return x, compression_rate

end

function matlab_object()
	#return mxcall(:matlab_get_coefficients,1,xpoints,ypoints,1.0*N)
	(vertex,topol,trian,edges,un,ds,ln,cent,N) = mxcall(:object_wrapper,9,1)
	topol = floor.(Int32,topol)
	trian = floor.(Int32,trian)
	edges = floor.(Int32,edges)
	ln = vec(ln)
	N = floor.(Int32,N)
	return vertex,topol,trian,edges,un,ds,ln,cent,N
end

function matlab_object_almond()
        #return mxcall(:matlab_get_coefficients,1,xpoints,ypoints,1.0*N)
        (vertex,topol,trian,edges,un,ds,ln,cent,N) = mxcall(:object_wrapper_almond,9,1)
        topol = floor.(Int32,topol)
        trian = floor.(Int32,trian)
        edges = floor.(Int32,edges)
        ln = vec(ln)
        N = floor.(Int32,N)
        return vertex,topol,trian,edges,un,ds,ln,cent,N
end


function matlab_object_spheres_interaction()
        #return mxcall(:matlab_get_coefficients,1,xpoints,ypoints,1.0*N)
        (vertex,topol,trian,edges,un,ds,ln,cent,N) = mxcall(:object_wrapper_spheres_interaction,9,1)
        topol = floor.(Int32,topol)
        trian = floor.(Int32,trian)
        edges = floor.(Int32,edges)
        ln = vec(ln)
        N = floor.(Int32,N)
        return vertex,topol,trian,edges,un,ds,ln,cent,N
end


function matlab_object_spheres_interaction2(lambda,radius,d_spheres,Nedges)
        #this one allows to give the parameters of the physical spheres
        #return mxcall(:matlab_get_coefficients,1,xpoints,ypoints,1.0*N)
        (vertex,topol,trian,edges,un,ds,ln,cent,N) = mxcall(:object_wrapper_spheres_interaction2,9,lambda,radius,1.0*d_spheres,1.0*Nedges)
        topol = floor.(Int32,topol)
        trian = floor.(Int32,trian)
        edges = floor.(Int32,edges)
        ln = vec(ln)
        N = floor.(Int32,N)
        return vertex,topol,trian,edges,un,ds,ln,cent,N
end




function matlabZ()
	return mxcall(:call_ACAsolver,1)
end

function matlabZ_almond()
	return mxcall(:call_ACAsolver_almond,1)
end

function getmatrix50()
	(U,V,m,n) = mxcall(:call_getmatrix50,4);
	return U,V,Int64.(m),Int64.(n)
end


function matlab_ACA(aca_threshold,A)
	#Calls the matlab ACA through a wrapper
	(U,V) = mxcall(:ACA_wrapper,2,aca_threshold,A);
	return U,V;
end

function C_ACA(aca_threshold,A)
	#Note: this is not written for performance, as the wrapper code is very inefficient. Only for checking correctness
	#(m,n) = size(A)
	sizeA = size(A)  #OJO	!!!
	if length(sizeA)==1
		m=sizeA[1]
		n=1
	else
		m = sizeA[1]
		n = sizeA[2]
	end
	U = zeros(ComplexF64,m,n); V = zeros(ComplexF64,m,n); #OJO, esta linea modificiad
	#Ac = convert(Array{ComplexF64,2},A) 
	Ac = convert(Array{ComplexF64,length(sizeA)},A)  #OJO
	#aca_order = ccall((:C_ACA_wrapper,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Cdouble,Int64,Int64),U,V,Ac,aca_threshold,m,n)
	aca_order = ccall((:C_ACA_wrapper,"./C_ACA_candidate2.so"),Int64,(Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Cdouble,Int64,Int64),U,V,Ac,aca_threshold,m,n)

	return U[:,1:aca_order], V[1:aca_order,:]
end

function get_Zcomp_data()
	#return mxcall(:matlab_get_coefficients,1,xpoints,ypoints,1.0*N)
	(Z_comp,vertex,topol,trian,edges,un,ds,ln,cent,N) = mxcall(:get_Z_comp,10,1)
	topol = floor.(Int32,topol)
	trian = floor.(Int32,trian)
	edges = floor.(Int32,edges)
	ln = vec(ln)
	N = floor.(Int32,N)
	return Z_comp, vertex,topol,trian,edges,un,ds,ln,cent,N
end


function get_Zcomp_data_extra()
	#This one also returns the incident field
	#return mxcall(:matlab_get_coefficients,1,xpoints,ypoints,1.0*N)
	(Z_comp,vertex,topol,trian,edges,un,ds,ln,cent,N,Ei) = mxcall(:get_Z_comp_extra,11,1)
	topol = floor.(Int32,topol)
	trian = floor.(Int32,trian)
	edges = floor.(Int32,edges)
	ln = vec(ln)
	N = floor.(Int32,N)
	return Z_comp, vertex,topol,trian,edges,un,ds,ln,cent,N,Ei
end


function get_Zcomp_data_extra_almond()
	#This one also returns the incident field
	#return mxcall(:matlab_get_coefficients,1,xpoints,ypoints,1.0*N)
	(Z_comp,vertex,topol,trian,edges,un,ds,ln,cent,N,Ei) = mxcall(:get_Z_comp_extra_almond,11,1)
	topol = floor.(Int32,topol)
	trian = floor.(Int32,trian)
	edges = floor.(Int32,edges)
	ln = vec(ln)
	N = floor.(Int32,N)
	return Z_comp, vertex,topol,trian,edges,un,ds,ln,cent,N,Ei
end



function get_synthetic_Zcomp(Z_comp_original,Z,aca_threshold)
	d = Vector{Dict}()
	nels = length(Z_comp_original)
	for ii=1:nels
		#println("synthetic loop ii= ",ii," out of ",nels)
		Zlocal = Array{Float64}(undef,0,0); U = Array{Float64}(undef,0,0); V = Array{Float64}(undef,0,0)
		#m = Int64.(Z_comp[ii]["m"]); n = Int64.(Z_comp[ii]["n"]) #OJO cambio esta línea
		m = Int64.(Z_comp_original[ii]["m"]); n = Int64.(Z_comp_original[ii]["n"]) 

		if m==n
			#println("m and n are equal")
		else
			#println("m and n ARE NOOOOT EQUAL")
		end
		comp = Int64.(Z_comp[ii]["comp"])
		if comp==1
			println("	about to ACA")
			(U,V) = C_ACA(aca_threshold,Z[m,n])
			#Zsub = Z[m,n]
			#U = zeros(ComplexF64,length(m),length(n)); V = zeros(ComplexF64,length(m),length(n));
			#Ac = convert(Array{ComplexF64,2},Zsub)
			#aca_order = ccall((:C_ACA_wrapper,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Cdouble,Int64,Int64),U,V,Ac,aca_threshold,length(m),length(n))


		elseif comp==0
			#println("	about to Zlocal")
			Zlocal = Z[m,n]
		else
			println("Error in get_synthetic_Zcomp: bad Z_comp index")
			return 
		end
		perdidadetiempo = 0;
		for ii=0:1000000
			perdidadetiempo = perdidadetiempo + cos(1.54354)
		end
		self = Int64(Z_comp_original[ii]["self"])
		#println("ABout to add to dict")
		#Dictlocal = Dict("comp"=>comp,"Z"=>Zlocal,"U"=>U,"V"=>V)
		Dictlocal = Dict("comp"=>comp,"Z"=>Zlocal,"U"=>U,"V"=>V,"m"=>m,"n"=>n,"self"=>self)
		push!(d,Dictlocal)
	end
	return d
end

function get_synthetic_Zcomp_skeleton(Z_comp_original,Z,aca_threshold)
	d = Vector{Dict}()
	nels = length(Z_comp_original)
	for ii=1:nels
		#println("synthetic loop ii= ",ii," out of ",nels)
		Zlocal = Array{Float64}(undef,0,0); U = Array{Float64}(undef,0,0); V = Array{Float64}(undef,0,0)
		Cs = Array{Float64}(undef,0,0);
		Us = Array{Float64}(undef,0,0);
		Rs = Array{Float64}(undef,0,0);

		m = Int64.(Z_comp[ii]["m"]); n = Int64.(Z_comp[ii]["n"])
		if m==n
			#println("m and n are equal")
		else
			#println("m and n ARE NOOOOT EQUAL")
		end
		comp = Int64.(Z_comp[ii]["comp"])
		if comp==1
			#println("	about to ACA")
			#println("ii of Z_comp is: ",ii)
			(U,V) = C_ACA(aca_threshold,Z[m,n])
			#() = #puntero
			#Zsub = Z[m,n]
			#U = zeros(ComplexF64,length(m),length(n)); V = zeros(ComplexF64,length(m),length(n));
			#Ac = convert(Array{ComplexF64,2},Zsub)
			#aca_order = ccall((:C_ACA_wrapper,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Cdouble,Int64,Int64),U,V,Ac,aca_threshold,length(m),length(n))
			(Cs,Us,Rs,rs,cs) = cur_pinv2(Z[m,n],size(U,2))

		elseif comp==0
			#println("	about to Zlocal")
			Zlocal = Z[m,n]
		else
			println("Error in get_synthetic_Zcomp: bad Z_comp index")
			return 
		end
		perdidadetiempo = 0;
		for ii=0:1000000
			perdidadetiempo = perdidadetiempo + cos(1.54354)
		end
		#println("ABout to add to dict")
		Dictlocal = Dict("comp"=>comp,"Z"=>Zlocal,"U"=>U,"V"=>V,"Cs"=>Cs,"Us"=>Us,"Rs"=>Rs,"m"=>m,"n"=>n)
		push!(d,Dictlocal)
	end
	return d
end


function get_C_synthetic_Zcomp(Z_comp_original,Z,aca_threshold)
	nels = length(Z_comp_original)
	vlength_m = zeros(Int32,nels); vlength_n = zeros(Int32,nels); vcompressed = zeros(Int32,nels);
	for ii=1:nels
		mv = Z_comp_original[ii]["m"]; nv = Z_comp_original[ii]["n"]
		vlength_m[ii] = length(mv); vlength_n[ii] = length(nv)
		vcompressed[ii] = Int32(Z_comp_original[ii]["comp"])
	end
	max_m = maximum(vlength_m); max_n = maximum(vlength_n)
	matrix_m = zeros(Int32,max_m,nels); matrix_n = zeros(Int32,max_n,nels)
	for ii=1:nels
	       mv = Z_comp_original[ii]["m"].-1; nv = Z_comp_original[ii]["n"].-1;
	       matrix_m[1:vlength_m[ii],ii]=vec(mv); matrix_n[1:vlength_n[ii],ii]= vec(nv)
	end
	Cmatrix_m = convert(Array{Cint,2},matrix_m)
	Cmatrix_n = convert(Array{Cint,2},matrix_n)	
	Cvlength_m = convert(Array{Cint,1},vlength_m)	
	Cvlength_n = convert(Array{Cint,1},vlength_n)
	Cvcompressed = convert(Array{Cint,1},vcompressed)
	Cnels = convert(Cint,nels)
	Cmax_m = convert(Int64,max_m)
	Cmax_n = convert(Int64,max_n)
	
	(mZ,nZ) = size(Z)
	CZ = convert(Array{ComplexF64,2},Z)
	CmZ = convert(Int64,mZ)
	CnZ = convert(Int64,nZ)

	#kk = ccall((:synthetic_compress,"./C_ACA.so"),Int64,(Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)
	#kk = ccall((:synthetic_compress,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)
	
	#kk = ccall((:synthetic_compress,"./C_ACA_candidate.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)
	##kk = ccall((:copy_to_a_file,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)
	kk = ccall((:copy_to_a_file2,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cvcompressed,Cnels,Cmax_m,Cmax_n)

	kk = ccall((:compare,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)

	println("mZ=",mZ,". nZ=",nZ,". nels=",nels,". maxm=",max_m,". maxn=",max_n)

	return matrix_m, matrix_n, vlength_m, vlength_n
	#return vlength_m, vlength_n

#	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
#	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
#	feed = obj.feed # A nothing
#	Ng = obj.Ng #A nothing
#	N = convert(Cint,obj.N) # An int 32


end

function get_C_synthetic_Zcomp2(Z_comp_original,Z,aca_threshold)
	#Nueva version creada en julio del 22 para sustituir los vicios de las anteriores
	nels = length(Z_comp_original)
	vlength_m = zeros(Int32,nels); vlength_n = zeros(Int32,nels); vcompressed = zeros(Int32,nels);
	for ii=1:nels
		mv = Z_comp_original[ii]["m"]; nv = Z_comp_original[ii]["n"]
		vlength_m[ii] = length(mv); vlength_n[ii] = length(nv)
		vcompressed[ii] = Int32(Z_comp_original[ii]["comp"])
	end
	max_m = maximum(vlength_m); max_n = maximum(vlength_n)
	matrix_m = zeros(Int32,max_m,nels); matrix_n = zeros(Int32,max_n,nels)
	for ii=1:nels
	       mv = Z_comp_original[ii]["m"].-1; nv = Z_comp_original[ii]["n"].-1;
	       if size(mv)==(); mv = [mv]; end #OJO
	       if size(nv)==(); nv = [nv]; end #OJO
	       matrix_m[1:vlength_m[ii],ii]=vec(mv); matrix_n[1:vlength_n[ii],ii]= vec(nv)
	end
	Cmatrix_m = convert(Array{Cint,2},matrix_m)
	Cmatrix_n = convert(Array{Cint,2},matrix_n)	
	Cvlength_m = convert(Array{Cint,1},vlength_m)	
	Cvlength_n = convert(Array{Cint,1},vlength_n)
	Cvcompressed = convert(Array{Cint,1},vcompressed)
	Cnels = convert(Cint,nels)
	Cmax_m = convert(Int64,max_m)
	Cmax_n = convert(Int64,max_n)
	
	(mZ,nZ) = size(Z)
	CZ = convert(Array{ComplexF64,2},Z)
	CmZ = convert(Int64,mZ)
	CnZ = convert(Int64,nZ)

	#kk = ccall((:synthetic_compress,"./C_ACA.so"),Int64,(Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)
	#kk = ccall((:synthetic_compress,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)
	
	#kk = ccall((:synthetic_compress,"./C_ACA_candidate.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)
	kk = ccall((:synthetic_compress2,"./C_ACA_candidate2.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cvcompressed,Cnels,Cmax_m,Cmax_n)

	##kk = ccall((:copy_to_a_file,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)
	#kk = ccall((:copy_to_a_file2,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cvcompressed,Cnels,Cmax_m,Cmax_n)

	#kk = ccall((:compare,"./C_ACA.so"),Int64,(Ptr{ComplexF64},Cint,Cint,Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Cint,Cint,Cint),CZ,CmZ,CnZ,Cmatrix_m,Cmatrix_n,Cvlength_m,Cvlength_n,Cnels,Cmax_m,Cmax_n)

	println("mZ=",mZ,". nZ=",nZ,". nels=",nels,". maxm=",max_m,". maxn=",max_n)

	return matrix_m, matrix_n, vlength_m, vlength_n
	#return vlength_m, vlength_n

#	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
#	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
#	feed = obj.feed # A nothing
#	Ng = obj.Ng #A nothing
#	N = convert(Cint,obj.N) # An int 32


end



function compare_compressions(Z_comp_original,dZ_comp)
	nels = length(Z_comp_original)
	vec_errU = Vector{Float64}()
	vec_errV = Vector{Float64}()

	for ii=1:nels
		if Z_comp_original[ii]["comp"]==1.0
			Uoriginal = Z_comp_original[ii]["U"]; Voriginal = Z_comp_original[ii]["V"]
			Usynth = dZ_comp[ii]["U"]; Vsynth = dZ_comp[ii]["V"]
			errU = norm(Uoriginal-Usynth)/norm(Uoriginal)
			errV = norm(Voriginal-Vsynth)/norm(Voriginal)
			push!(vec_errU,errU); push!(vec_errV,errV)
		end
	end
	return vec_errU, vec_errV
end


function compare_aca_vs_skl(Z_comp,Z)
	#Where Z_comp should be the synthetic that has 
	nels = length(Z_comp)
	vec_err_skl_wrt_aca = Vector{Float64}()
	vec_err_aca_wrt_Z = Vector{Float64}()
	vec_err_skl_wrt_Z = Vector{Float64}()
	vec_ii = Vector{Float64}()
	

	for ii=1:nels
		if Z_comp[ii]["comp"]==1.0
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			Z_local = Z[Int64.(m),Int64.(n)]

			U = Z_comp[ii]["U"]; V = Z_comp[ii]["V"]

			#Cs = Z_comp[ii]["Cs"]; Us = Z_comp[ii]["Us"]; Rs = Z_comp[ii]["Rs"]
			order_aprox = size(U,2)
			(Cs,Us,Rs) = cur_pinv2(Z_local,order_aprox)
			err_skl_wrt_aca = norm(Cs*Us*Rs-U*V)/norm(U*V)
			err_aca_wrt_Z   = norm(U*V-Z_local)/norm(Z_local)
			err_skl_wrt_Z   = norm(Cs*Us*Rs-Z_local)/norm(Z_local)

			#Uoriginal = Z_comp_original[ii]["U"]; Voriginal = Z_comp_original[ii]["V"]
			#Usynth = dZ_comp[ii]["U"]; Vsynth = dZ_comp[ii]["V"]

			#errU = norm(Uoriginal-Usynth)/norm(Uoriginal)
			#errV = norm(Voriginal-Vsynth)/norm(Voriginal)
			#push!(vec_errU,errU); push!(vec_errV,errV)
			push!(vec_err_skl_wrt_aca,err_skl_wrt_aca)
			push!(vec_err_aca_wrt_Z  ,err_aca_wrt_Z)
			push!(vec_err_skl_wrt_Z,err_skl_wrt_Z)
			push!(vec_ii,ii)
		end
	end
	return vec_err_skl_wrt_aca, vec_err_aca_wrt_Z, vec_err_skl_wrt_Z, vec_ii  #vec_errU, vec_errV
end

function rebuild_matrix(Z_comp,Z)
	#Where Z_comp should be the synthetic on
	Zfull_aca = zeros(Complex{Float64},size(Z,1),size(Z,2)); Zfull_pinv = zeros(Complex{Float64},size(Z,1),size(Z,2))
	nels = length(Z_comp)
	number_els_compressed_aca = 0; #number of elements in the ACA compressed matrix	
	number_els_compressed_pinv = 0; #number of elements in the pinv compressed matrix

	time_pinv = 0.0; time_aca = 0.0

	for ii=1:nels
		self = Z_comp[ii]["self"]
		if Z_comp[ii]["comp"]==1.0
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			Z_local = Z[Int64.(m),Int64.(n)]

			U = Z_comp[ii]["U"]; V = Z_comp[ii]["V"]
			
			tt1 = @timed begin
			(Utrash,Vtrash) = C_ACA(1e-3,Z_local)
			end#timed
			time_aca = time_aca + tt1[2]

			#Cs = Z_comp[ii]["Cs"]; Us = Z_comp[ii]["Us"]; Rs = Z_comp[ii]["Rs"]
			order_aprox = size(U,2)
			#(Cs,Us,Rs) = cur_pinv2(Z_local,order_aprox)
			#(Cs,Us,Rs) = cur_pinv2(Z_local,min(length(m),length(n)))
			tt = @timed begin
			#(Cs,Us,Rs) = cur_compress(Z_local,1e-1)
			(Cs,Us,Rs) = cuda_cur_compress(Z_local,1e-1)

			end #timed
			time_pinv = time_pinv + tt[2]
			m = Int64.(m); n = Int64.(n)

			if self == 1
				Zfull_aca[m,n] = U*V;
				Zfull_pinv[m,n] = Cs*Ur*Rs
				number_els_compressed_aca = number_els_compressed_aca + size(U,1)*size(U,2) + size(V,1)*size(V,2)
				number_els_compressed_pinv = number_els_compressed_pinv + size(Cs,1)*size(Cs,2) + size(Rs,1)*size(Rs,2)
			elseif self == 0
				Zfull_aca[m,n] = U*V; Zfull_aca[n,m] = transpose(U*V)
				Zfull_pinv[m,n] = Cs*Us*Rs; Zfull_pinv[n,m] = transpose(Cs*Us*Rs)
				number_els_compressed_aca = number_els_compressed_aca + 1*(size(U,1)*size(U,2) + size(V,1)*size(V,2))
				number_els_compressed_pinv = number_els_compressed_pinv +1*(size(Cs,1)*size(Cs,2) + size(Rs,1)*size(Rs,2))

			else
				error("self field incorrect in rebuild_matrix")
			end

			#Zfull_aca[m,n] = U*V
			#Zfull_pinv[m,n] = Cs*Us*Rs
			#Zfull_aca[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
			#Zfull_aca[m,n] = ones(Complex{Float64},length(m),length(n)); Zfull_pinv[m,n] = ones(Complex{Float64},length(m),length(n))

			#Zfull_pinv[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
		
			#err_skl_wrt_aca = norm(Cs*Us*Rs-U*V)/norm(U*V)
			#err_aca_wrt_Z   = norm(U*V-Z_local)/norm(Z_local)
			#err_skl_wrt_Z   = norm(Cs*Us*Rs-Z_local)/norm(Z_local)

			#Uoriginal = Z_comp_original[ii]["U"]; Voriginal = Z_comp_original[ii]["V"]
			#Usynth = dZ_comp[ii]["U"]; Vsynth = dZ_comp[ii]["V"]

			#errU = norm(Uoriginal-Usynth)/norm(Uoriginal)
			#errV = norm(Voriginal-Vsynth)/norm(Voriginal)
			#push!(vec_errU,errU); push!(vec_errV,errV)
			#push!(vec_err_skl_wrt_aca,err_skl_wrt_aca)
			#push!(vec_err_aca_wrt_Z  ,err_aca_wrt_Z)
			#push!(vec_err_skl_wrt_Z,err_skl_wrt_Z)
			#push!(vec_ii,ii)
		else
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			m = Int64.(m); n = Int64.(n)
			Z_local = Z[m,n]
			
			if self == 1
				Zfull_aca[m,n] = Z_local;
				Zfull_pinv[m,n] = Z_local;
				number_els_compressed_aca = number_els_compressed_aca + size(Z_local,1)*size(Z_local,2)
				number_els_compressed_pinv = number_els_compressed_pinv + size(Z_local,1)*size(Z_local,2)
			elseif self == 0
				Zfull_aca[m,n] = Z_local; Zfull_aca[n,m] = transpose(Z_local)
				Zfull_pinv[m,n] = Z_local; Zfull_pinv[n,m] = transpose(Z_local)
				number_els_compressed_aca = number_els_compressed_aca + 1*(size(Z_local,1)*size(Z_local,2))
				number_els_compressed_pinv = number_els_compressed_pinv + 1*(size(Z_local,1)*size(Z_local,2))

			else
				error("self field incorrect in rebuild_matrix")
			end

		


			#Zfull_aca[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
			#Zfull_aca[m,n] = ones(Complex{Float64},length(m),length(n)); Zfull_pinv[m,n] = ones(Complex{Float64},length(m),length(n))
		end
	end
	compression_aca  = number_els_compressed_aca/(size(Zfull_aca,1)*size(Zfull_aca,2))
	compression_pinv = number_els_compressed_pinv/(size(Zfull_pinv,1)*size(Zfull_pinv,2))
	return Zfull_aca,  Zfull_pinv, compression_aca, compression_pinv, time_pinv, time_aca

end

function user_impedance_cuda3(r1,r2,obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
	#Esta version decide adaptativamente los parámetros de configuarcion del kernel
	#Este improved es el paso previo a lo que haremos a continuación con la GPU
	#This function is a wrapper for the C code
	#In this user_impedance2 we try to solve the proble,m of the previous one, which doesn't allow to compute arbitrary elements of the matrix


	time_uic_1 = @timed begin
	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)

	#The matrix where we will write the result
	Zr = zeros(Cdouble,n_r1,n_r2) #OJO: This is what we fix wrt to the previous version
	Zi = zeros(Cdouble,n_r1,n_r2)

	end
	
	println("time_uic_1 = ",time_uic_1[2])

	#Call the external C code
	#Notas de prueba:
	#impedance_matrix: el bueno
	#impedance_matrix_wrapper: el de las pruebas de wrapper
	#impedance_matrix_parts: el inicio del paralelismo
	#println("Inside julia, time impedance")
	#timeccal = @timed 
	#{
	#Antes impedance_matrix_cuda. ahora impedance_matrix_cuda_elbyel
	timeccal = @timed res = ccall((:impedance_matrix_cuda_elbyel2,"./functionsExternal_cuda.so"),
		    Int64,
		    (Ptr{Cdouble},Ptr{Cdouble},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint,Cint,Cint,Cint,Cint,Cint,Cint),
		    Zr,
		    Zi,
		    field,
		    k,
		    eta,
		    Rinteg_s,
		    Ranal_s,
		    Rinteg_f,
		    cor_solid,
		    flag,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent,
		    n_r1,
		    n_r2,
		    1,
		    1,
		    n_r1,
		    n_r2)
	#}#timed

	println("Julia cc time cuda: ",timeccal[2])
	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))


	println("Time summation")
	return @time  Zr+im*Zi
end

function user_impedance_cuda4(r1,r2,obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
	#Esta version 4 evita calcular por separado la parte real y la imaginaria
	#Esta version decide adaptativamente los parámetros de configuarcion del kernel
	#Este improved es el paso previo a lo que haremos a continuación con la GPU
	#This function is a wrapper for the C code
	#In this user_impedance2 we try to solve the proble,m of the previous one, which doesn't allow to compute arbitrary elements of the matrix


	time_uic_1 = @timed begin
	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)

	#The matrix where we will write the result
	#Zr = zeros(Cdouble,n_r1,n_r2) #OJO: This is what we fix wrt to the previous version
	#Zi = zeros(Cdouble,n_r1,n_r2)
     	Ztotal = zeros(ComplexF64,n_r1,n_r2)

	end
	
	println("time_uic_1 = ",time_uic_1[2])

	#Call the external C code
	#Notas de prueba:
	#impedance_matrix: el bueno
	#impedance_matrix_wrapper: el de las pruebas de wrapper
	#impedance_matrix_parts: el inicio del paralelismo
	#println("Inside julia, time impedance")
	#timeccal = @timed 
	#{
	#Antes impedance_matrix_cuda. ahora impedance_matrix_cuda_elbyel
	timeccal = @timed res = ccall((:impedance_matrix_cuda_elbyel3,"./functionsExternal_cuda.so"),
		    Int64,
		    (Ptr{ComplexF64},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint,Cint,Cint,Cint,Cint,Cint,Cint),
		    Ztotal,
		    field,
		    k,
		    eta,
		    Rinteg_s,
		    Ranal_s,
		    Rinteg_f,
		    cor_solid,
		    flag,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent,
		    n_r1,
		    n_r2,
		    1,
		    1,
		    n_r1,
		    n_r2)
	#}#timed

	println("Julia cc time cuda: ",timeccal[2])
	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))


	println("Time summation")
	#return @time  Zr+im*Zi
	return Ztotal
end

function user_impedance_cuda5(r1,r2,obj::obj_struct,EM_data::EM_data_struct)
	#Esta version 5 cuenta el tiempo que tarda
	#Esta version 4 evita calcular por separado la parte real y la imaginaria
	#Esta version decide adaptativamente los parámetros de configuarcion del kernel
	#Este improved es el paso previo a lo que haremos a continuación con la GPU
	#This function is a wrapper for the C code
	#In this user_impedance2 we try to solve the proble,m of the previous one, which doesn't allow to compute arbitrary elements of the matrix


	#time_uic_1 = @timed begin
	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)

	#The matrix where we will write the result
	#Zr = zeros(Cdouble,n_r1,n_r2) #OJO: This is what we fix wrt to the previous version
	#Zi = zeros(Cdouble,n_r1,n_r2)
     	Ztotal = zeros(ComplexF64,n_r1,n_r2)

	#end timed
	
	#println("time_uic_1 = ",time_uic_1[2])

	vec_time_cuda = zeros(Cdouble,1,1)

	#Call the external C code
	#Notas de prueba:
	#impedance_matrix: el bueno
	#impedance_matrix_wrapper: el de las pruebas de wrapper
	#impedance_matrix_parts: el inicio del paralelismo
	#println("Inside julia, time impedance")
	#timeccal = @timed 
	#{
	#Antes impedance_matrix_cuda. ahora impedance_matrix_cuda_elbyel
	time_tardado = 0.0
	#time_tardado = ccall((:impedance_matrix_cuda_elbyel4,"./functionsExternal_cuda.so"),
	#	    Cdouble,
	#	    (Ptr{ComplexF64},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint,Cint,Cint,Cint,Cint,Cint,Cint),
	#	    vec_time_cuda,
	#	    Ztotal,
	#	    field,
	#	    k,
	#	    eta,
	#	    Rinteg_s,
	#	    Ranal_s,
	#	    Rinteg_f,
	#	    cor_solid,
	#	    flag,
	#	    r1,
	#	    r2,
	#	    vertex,
	#	    topol,
	#	    trian,
	#	    edges,
	#	    un,
	#	    ds,
	#	    ln,
	#	    cent,
	#	    N,
	#	    n_r1,
	#	    n_r2,
	#	    rows_vertex,
	#	    cols_vertex,
	#	    rows_topol,
	#	    cols_topol,
	#	    rows_trian,
	#	    cols_trian,
	#	    rows_edges,
	#	    cols_edges,
	#	    rows_un,
	#	    cols_un,
	#	    rows_ds,
	#	    cols_ds,
	#	    n_ln,
	#	    rows_cent,
	#	    cols_cent,
	#	    n_r1,
	#	    n_r2,
	#	    1,
	#	    1,
	#	    n_r1,
	#	    n_r2)
	##}#timed
	
	timeccal = @timed time_tardado = ccall((:impedance_matrix_cuda_elbyel4,"./functionsExternal_cuda.so"),
		    Cdouble,
		    (Ptr{ComplexF64},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint,Cint,Cint,Cint,Cint,Cint,Cint),
		    Ztotal,
		    field,
		    k,
		    eta,
		    Rinteg_s,
		    Ranal_s,
		    Rinteg_f,
		    cor_solid,
		    flag,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent,
		    n_r1,
		    n_r2,
		    1,
		    1,
		    n_r1,
		    n_r2)

	println("Julia cc time cuda: ",timeccal[2])
	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))


	println("Time summation")
	#return @time  Zr+im*Zi
	println("typeof(Ztotal): ",typeof(Ztotal))
	println("size(Ztotal): ",size(Ztotal))
	println("Time tardado: ",time_tardado)
	return Ztotal, time_tardado
end



function pure_cuda_compression(r1,r2,tol,obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
	#Esta version decide adaptativamente los parámetros de configuarcion del kernel
	#Este improved es el paso previo a lo que haremos a continuación con la GPU
	#This function is a wrapper for the C code
	#In this user_impedance2 we try to solve the proble,m of the previous one, which doesn't allow to compute arbitrary elements of the matrix


	time_uic_1 = @timed begin
	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string
	

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)
	C_tol = convert(Cdouble,tol)

	#The matrix where we will write the result
	Zr = zeros(Cdouble,n_r1,n_r2) #OJO: This is what we fix wrt to the previous version
	Zi = zeros(Cdouble,n_r1,n_r2)

	end

	#Almacenamiento de datos comprimidos de manera ineficiente.
	Ccompressed_r = zeros(Cdouble,n_r1,n_r2)
	Ccompressed_i = zeros(Cdouble,n_r1,n_r2)

	Ucompressed_r = zeros(Cdouble,n_r2,n_r1)
	Ucompressed_i = zeros(Cdouble,n_r2,n_r1)

	Rcompressed_r = zeros(Cdouble,n_r1,n_r2)
	Rcompressed_i = zeros(Cdouble,n_r1,n_r2)

	println("time_uic_1 = ",time_uic_1[2])

	#Call the external C code
	#Notas de prueba:
	#impedance_matrix: el bueno
	#impedance_matrix_wrapper: el de las pruebas de wrapper
	#impedance_matrix_parts: el inicio del paralelismo
	#println("Inside julia, time impedance")
	#timeccal = @timed 
	#{
	#Antes impedance_matrix_cuda. ahora impedance_matrix_cuda_elbyel
timeccal = @timed res = ccall((:cuda_cur_pure,"./functionsExternal_cuda.so"),
	    Int64,
	    (Ptr{Cdouble},Ptr{Cdouble},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}, Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cdouble),
	    Zr,
	    Zi,
	    field,
	    k,
	    eta,
	    Rinteg_s,
	    Ranal_s,
	    Rinteg_f,
	    cor_solid,
	    flag,
	    r1,
	    r2,
	    vertex,
	    topol,
	    trian,
	    edges,
	    un,
	    ds,
	    ln,
	    cent,
	    Ccompressed_r,
	    Ccompressed_i,
	    Ucompressed_r,
	    Ucompressed_i,
	    Rcompressed_r,
	    Rcompressed_i,
	    N,
	    n_r1,
	    n_r2,
	    rows_vertex,
	    cols_vertex,
	    rows_topol,
	    cols_topol,
	    rows_trian,
	    cols_trian,
	    rows_edges,
	    cols_edges,
	    rows_un,
	    cols_un,
	    rows_ds,
	    cols_ds,
	    n_ln,
	    rows_cent,
	    cols_cent,
	    n_r1,
	    n_r2,
	    1,
	    1,
	    n_r1,
	    n_r2,
	    C_tol)
	#}#timed

	#

	println("Julia cc time cuda: ",timeccal[2])
	#aM = ccall((:vectorMean,"./funsPrueba.so"),Float64,(Ptr{Cint},Cint),r1,length(r1))


	println("Time summation")
	return @time  Zr+im*Zi
end

function pure_cuda_compression2(vtest,Avtest_old,current_size,pinv_tol,r1,r2,tol,obj::obj_struct,EM_data::EM_data_struct)
	#Este ya es funcionalmente correcto (o lo pretende) aunque no esté optimizado

	cur_order = current_size

	#pinv_tol = 1e-10
	MM = length(r1)
	NN = length(r2)
        col_samples = sample(1:NN,current_size,replace=false)
        row_samples = sample(1:MM,current_size,replace=false)
        col_samples = sort(col_samples); row_samples = sort(row_samples);
	col_samples = r2[col_samples]; row_samples = r1[row_samples];
        Ccol_samples = convert(Array{Cdouble,1},col_samples)
        Crow_samples = convert(Array{Cdouble,1},row_samples)





#
#	vtest = rand(N) + im*rand(N)
#	D = min(M,N)
#	if D<10
#		start_order = 1
#	else
#		start_order = div(D,10)
#	end
#
#	flag_err=0;
#	flag_size=0
#	err_app = Inf
#	current_size = start_order
#	(C,U,R,err_here) = cuda_cur_(pinv_tol,r1,r2,obj,EM_data)
	
	###


	#time_uic_1 = @timed begin
	#Take elements from EM_data with the proper conversions
	field = convert(Cint,EM_data.field)	
	k = EM_data.k
	eta = EM_data.eta
	Rinteg_s = EM_data.Rint_s
	Ranal_s = EM_data.Ranal_s
	Rinteg_f = EM_data.Rint_f
	cor_solid = convert(Cint,EM_data.corr_solid)
	flag = convert(Cint,EM_data.flag)
	Ccur_order = convert(Cint,cur_order)

	
	#Take elements from obj with the proper conversions
	r1 = collect(Cint,r1); r1 = convert(Array{Cdouble,1},r1) #A vector of doubles
	r2 = collect(Cint,r2); r2 = convert(Array{Cdouble,1},r2) #A vector of doubles
	vertex = convert(Array{Cdouble,2},obj.vertex) #An array of float64
	topol = convert(Array{Cdouble,2},obj.topol) #An array of float64
	trian = convert(Array{Cdouble,2},obj.trian) # An array of float64
	edges = convert(Array{Cdouble,2},obj.edges) #An array of Ints 32
	un = convert(Array{Cdouble,2},obj.un) # An array of float64
	ds = convert(Array{Cdouble,2},obj.ds) #An arry of float64
	ln = convert(Array{Cdouble,1},obj.ln) # An array of Float64
	cent = convert(Array{Cdouble,2},obj.cent) # An array of Float64
	feed = obj.feed # A nothing
	Ng = obj.Ng #A nothing
	N = convert(Cint,obj.N) # An int 32
	name = obj.name # A string

	#Integers describing sizes
	n_r1 = length(r1)
	n_r2 = length(r2)
	rows_vertex = size(vertex,1)
	cols_vertex = size(vertex,2)
	rows_topol = size(topol,1)
	cols_topol = size(topol,2)
	rows_trian = size(topol,1)
	cols_trian = size(topol,2)
	rows_edges = size(edges,1)
	cols_edges = size(edges,2)
	rows_un = size(un,1)
	cols_un = size(un,2)
	rows_ds = size(ds,1)
	cols_ds = size(ds,2)
	n_ln = length(ln)
	rows_cent = size(cent,1)
	cols_cent = size(cent,2)

	#The matrix where we will write the result
	#Zr = zeros(Cdouble,n_r1,n_r2) #OJO: This is what we fix wrt to the previous version
	#Zi = zeros(Cdouble,n_r1,n_r2)
     	Ztotal = zeros(ComplexF64,n_r1,n_r2)

	#end
	

	#Almacenamiento de datos comprimidos de manera ineficiente.
	Ccompressed = zeros(ComplexF64,MM,cur_order)
	Ucompressed = zeros(ComplexF64,cur_order,cur_order)
	Rcompressed = zeros(ComplexF64,cur_order,NN)

	#vectores test
	vtest_c = convert(Array{ComplexF64,1},vtest)
	Avtest_old_c = convert(Array{ComplexF64,1},Avtest_old)
	Avtest_new = zeros(ComplexF64,MM) #OJO a dimension de esto

	#Vector para almacenar tamaño final de la compresion
	#vec_cur_order = zeros(Cint,1,1)

	#Vector para almacenar el tiempo transcurrido
	vec_time_cuda = zeros(Cdouble,1,1)

	#Call the external C code
	#Notas de prueba:
	#impedance_matrix: el bueno
	#impedance_matrix_wrapper: el de las pruebas de wrapper
	#impedance_matrix_parts: el inicio del paralelismo
	#println("Inside julia, time impedance")
	#timeccal = @timed 
	#{
	#Antes impedance_matrix_cuda. ahora impedance_matrix_cuda_elbyel
#	timeccal = @timed err_here = ccall((:cuda_complete_compression,"./functionsExternal_cuda.so"),
#		    Cdouble,
#		    (Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Cint,Ptr{Cdouble},Cdouble,Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{Cdouble},Ptr{Cdouble},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,
#		      Ptr{ComplexF64},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint,Cint,Cint,Cint,Cint,Cint,Cint),
#		    Ccompressed, #Añadido
#		    Ucompressed, #Id
#		    Rcompressed, #Id
#		    Ccur_order, #Id (OJO, no asignado aún)
#		    vec_time_cuda, #Id
#		    pinv_tol, #Id
#		    Avtest_old_c, #Id
#		    Avtest_new, #Id
#		    vtest_c, #Id
#		    Ccol_samples, #Id
#		    Crow_samples, #Id
#		    n_r1, #Id asd threads C
#		    Ccur_order, #Id treadhs C
#		    1, #Id blocks C
#		    1, #Id blocks C
#		    Ccur_order, #Id threads U
#		    Ccur_order, #Id threads U
#		    1, #Id blocks U
#		    1, #Id blocks U 
#		    Ccur_order, #Id threads R
#		    n_r2, #Id threads R,
#		    1, #Id blocks R
#		    1, #Id Blocks R
#		    Ztotal,
#		    field,
#		    k,
#		    eta,
#		    Rinteg_s,
#		    Ranal_s,
#		    Rinteg_f,
#		    cor_solid,
#		    flag,
#		    r1,
#		    r2,
#		    vertex,
#		    topol,
#		    trian,
#		    edges,
#		    un,
#		    ds,
#		    ln,
#		    cent,
#		    N,
#		    n_r1,
#		    n_r2,
#		    rows_vertex,
#		    cols_vertex,
#		    rows_topol,
#		    cols_topol,
#		    rows_trian,
#		    cols_trian,
#		    rows_edges,
#		    cols_edges,
#		    rows_un,
#		    cols_un,
#		    rows_ds,
#		    cols_ds,
#		    n_ln,
#		    rows_cent,
#		    cols_cent,
#		    n_r1,
#		    n_r2,
#		    1,
#		    1,
#		    n_r1,
#		    n_r2)
###para distinta configuracion blocks threads


	#	C_blocks_x = n_r1
	#	C_blocks_y = Ccur_order
	#	C_threads_x = 1
	#	C_threads_y = 1

	#	U_blocks_x = Ccur_order
	#	U_blocks_y = Ccur_order
	#	U_threads_x = 1
	#	U_threads_y = 1

	#	R_blocks_x = Ccur_order
	#	R_blocks_y = n_r2
	#	R_threads_x = 1
	#	R_threads_y = 1

		#divfactor = 16
		println("divfactor is ",divfactor)

		C_blocks_x = ceil(Cint,n_r1/divfactor)
		C_blocks_y = ceil(Cint,Ccur_order/divfactor)
		C_threads_x = divfactor
		C_threads_y = divfactor

		U_blocks_x = ceil(Cint,Ccur_order/divfactor)
		U_blocks_y = ceil(Cint,Ccur_order/divfactor)
		U_threads_x = divfactor
		U_threads_y = divfactor

		R_blocks_x = ceil(Cint,Ccur_order/divfactor)
		R_blocks_y = ceil(Cint,n_r2/divfactor)
		R_threads_x = divfactor
		R_threads_y = divfactor



		timeccal = @timed err_here = ccall((:cuda_complete_compression,"./functionsExternal_cuda.so"),
		    Cdouble,
		    (Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Cint,Ptr{Cdouble},Cdouble,Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{Cdouble},Ptr{Cdouble},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,
		      Ptr{ComplexF64},    Cint,Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Cint, Cint,    Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},  Ptr{Cdouble},  Ptr{Cdouble},    Cint,  Cint,  Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint,Cint,Cint,Cint,Cint,Cint,Cint),
		    Ccompressed, #Añadido
		    Ucompressed, #Id
		    Rcompressed, #Id
		    Ccur_order, #Id (OJO, no asignado aún)
		    vec_time_cuda, #Id
		    pinv_tol, #Id
		    Avtest_old_c, #Id
		    Avtest_new, #Id
		    vtest_c, #Id
		    Ccol_samples, #Id
		    Crow_samples, #Id
		    C_blocks_x, #Id asd threads C
		    C_blocks_y, #Id treadhs C
		    C_threads_x, #Id blocks C
		    C_threads_y, #Id blocks C
		    U_blocks_x, #Id threads U
		    U_blocks_y, #Id threads U
		    U_threads_x, #Id blocks U
		    U_threads_y, #Id blocks U 
		    R_blocks_x, #Id threads R
		    R_blocks_y, #Id threads R,
		    R_threads_x, #Id blocks R
		    R_threads_y, #Id Blocks R
		    Ztotal,
		    field,
		    k,
		    eta,
		    Rinteg_s,
		    Ranal_s,
		    Rinteg_f,
		    cor_solid,
		    flag,
		    r1,
		    r2,
		    vertex,
		    topol,
		    trian,
		    edges,
		    un,
		    ds,
		    ln,
		    cent,
		    N,
		    n_r1,
		    n_r2,
		    rows_vertex,
		    cols_vertex,
		    rows_topol,
		    cols_topol,
		    rows_trian,
		    cols_trian,
		    rows_edges,
		    cols_edges,
		    rows_un,
		    cols_un,
		    rows_ds,
		    cols_ds,
		    n_ln,
		    rows_cent,
		    cols_cent,
		    n_r1,
		    n_r2,
		    1,
		    1,
		    n_r1,
		    n_r2)


	#}#timed

	println("err_here (inside pure_cuda_compression2): ",err_here)

	###
	return Ccompressed, Ucompressed, Rcompressed, row_samples, col_samples, Avtest_new, err_here, vec_time_cuda[1] 

end


function rebuild_matrix_cuda(Z_comp,Z,obj,EM_data)
	#Where Z_comp should be the synthetic on
	Zfull_aca = zeros(Complex{Float64},size(Z,1),size(Z,2)); Zfull_pinv = zeros(Complex{Float64},size(Z,1),size(Z,2))
	nels = length(Z_comp)
	number_els_compressed_aca = 0; #number of elements in the ACA compressed matrix	
	number_els_compressed_pinv = 0; #number of elements in the pinv compressed matrix

	time_pinv = 0.0; time_aca = 0.0

	for ii=1:nels #antes 1:nels
		self = Z_comp[ii]["self"]
		if Z_comp[ii]["comp"]==1.0
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			Z_local = Z[Int64.(m),Int64.(n)]
        #Ccol_samples = convert(Array{Cint,1},col_samples)
        #Crow_samples = convert(Array{Cint,1},row_samples)
			if size(m)==(); m=[m]; end
			if size(n)==(); n=[n]; end
			Z_local_cuda = user_impedance_cuda3(Int64.(m),Int64.(n),obj,EM_data)
			println("ii: ",ii," out of ",nels)
			#println("m: ",m)
			#println("n: ",n)
			#println("Z_local: ",Z_local[1])
			#println("Z_local_cuda: ",Z_local_cuda[1])
			#println("size(m): ",size(m))
			#println("size(n): ",size(n))
			#println("size(Z_local_cuda): ",size(Z_local_cuda))
			#println("size(Z_local): ",size(Z_local))
			(s1help,s2help) = size(Z_local_cuda)
			if s1help==1 || s2help==1
				println("Err cuda mat(case vec): ",norm(vec(Z_local_cuda)- vec(Z_local))/norm(vec(Z_local)) )
			else
				println("Err cuda mat(case normal): ",norm(Z_local_cuda- Z_local)/norm(Z_local))
			end

			U = Z_comp[ii]["U"]; V = Z_comp[ii]["V"]
			
			tt1 = @timed begin
			(Utrash,Vtrash) = C_ACA(1e-3,Z_local)
			end#timed
			time_aca = time_aca + tt1[2]

			#Cs = Z_comp[ii]["Cs"]; Us = Z_comp[ii]["Us"]; Rs = Z_comp[ii]["Rs"]
			order_aprox = size(U,2)
			#(Cs,Us,Rs) = cur_pinv2(Z_local,order_aprox)
			#(Cs,Us,Rs) = cur_pinv2(Z_local,min(length(m),length(n)))
			tt = @timed begin
			#(Cs,Us,Rs) = cur_compress(Z_local,1e-1)
			#(Cs,Us,Rs) = cuda_cur_compress(Z_local,1e-1)
			#(Cs,Us,Rs) = cuda_cur_compress2(Z_local,1e-1) #Estaba
			println("about to cuda_cur_compress3")
			(Cs,Us,Rs,trash,trash,trash,time_cur3_partial) = cuda_cur_compress3(Int64.(m),Int64.(n),1e-1,obj,EM_data)
			#(Cs,Us,Rs,trash,trash,trash,time_cur3_partial) = cur_compress2(Int64.(m),Int64.(n),1e-1,obj,EM_data)

			#println("Error cuda_cur_compress3: ",norm(Cs*Us*Rs- Z_local_cuda)/norm(Z_local_cuda))
			end #timed
			#time_pinv = time_pinv + tt[2]
			time_pinv = time_pinv + time_cur3_partial
			m = Int64.(m); n = Int64.(n)

			if self == 1
				Zfull_aca[m,n] = U*V;
				Zfull_pinv[m,n] = Cs*Ur*Rs
				number_els_compressed_aca = number_els_compressed_aca + size(U,1)*size(U,2) + size(V,1)*size(V,2)
				number_els_compressed_pinv = number_els_compressed_pinv + size(Cs,1)*size(Cs,2) + size(Rs,1)*size(Rs,2)
			elseif self == 0
				Zfull_aca[m,n] = U*V; Zfull_aca[n,m] = transpose(U*V)
				Zfull_pinv[m,n] = Cs*Us*Rs; Zfull_pinv[n,m] = transpose(Cs*Us*Rs)
				number_els_compressed_aca = number_els_compressed_aca + 1*(size(U,1)*size(U,2) + size(V,1)*size(V,2))
				number_els_compressed_pinv = number_els_compressed_pinv +1*(size(Cs,1)*size(Cs,2) + size(Rs,1)*size(Rs,2))

			else
				error("self field incorrect in rebuild_matrix")
			end

			#Zfull_aca[m,n] = U*V
			#Zfull_pinv[m,n] = Cs*Us*Rs
			#Zfull_aca[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
			#Zfull_aca[m,n] = ones(Complex{Float64},length(m),length(n)); Zfull_pinv[m,n] = ones(Complex{Float64},length(m),length(n))

			#Zfull_pinv[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
		
			#err_skl_wrt_aca = norm(Cs*Us*Rs-U*V)/norm(U*V)
			#err_aca_wrt_Z   = norm(U*V-Z_local)/norm(Z_local)
			#err_skl_wrt_Z   = norm(Cs*Us*Rs-Z_local)/norm(Z_local)

			#Uoriginal = Z_comp_original[ii]["U"]; Voriginal = Z_comp_original[ii]["V"]
			#Usynth = dZ_comp[ii]["U"]; Vsynth = dZ_comp[ii]["V"]

			#errU = norm(Uoriginal-Usynth)/norm(Uoriginal)
			#errV = norm(Voriginal-Vsynth)/norm(Voriginal)
			#push!(vec_errU,errU); push!(vec_errV,errV)
			#push!(vec_err_skl_wrt_aca,err_skl_wrt_aca)
			#push!(vec_err_aca_wrt_Z  ,err_aca_wrt_Z)
			#push!(vec_err_skl_wrt_Z,err_skl_wrt_Z)
			#push!(vec_ii,ii)
		else
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			m = Int64.(m); n = Int64.(n)
			Z_local = Z[m,n]
			
			if self == 1
				Zfull_aca[m,n] = Z_local;
				Zfull_pinv[m,n] = Z_local;
				number_els_compressed_aca = number_els_compressed_aca + size(Z_local,1)*size(Z_local,2)
				number_els_compressed_pinv = number_els_compressed_pinv + size(Z_local,1)*size(Z_local,2)
			elseif self == 0
				Zfull_aca[m,n] = Z_local; Zfull_aca[n,m] = transpose(Z_local)
				Zfull_pinv[m,n] = Z_local; Zfull_pinv[n,m] = transpose(Z_local)
				number_els_compressed_aca = number_els_compressed_aca + 1*(size(Z_local,1)*size(Z_local,2))
				number_els_compressed_pinv = number_els_compressed_pinv + 1*(size(Z_local,1)*size(Z_local,2))

			else
				error("self field incorrect in rebuild_matrix")
			end

		


			#Zfull_aca[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
			#Zfull_aca[m,n] = ones(Complex{Float64},length(m),length(n)); Zfull_pinv[m,n] = ones(Complex{Float64},length(m),length(n))
		end
	end
	compression_aca  = number_els_compressed_aca/(size(Zfull_aca,1)*size(Zfull_aca,2))
	compression_pinv = number_els_compressed_pinv/(size(Zfull_pinv,1)*size(Zfull_pinv,2))
	return Zfull_aca,  Zfull_pinv, compression_aca, compression_pinv, time_pinv, time_aca

end


function rebuild_matrix_cuda2(Z_comp,Z,obj,EM_data)
	#Calculamos los elementos no comprimidos con CPU y GPU
	#Where Z_comp should be the synthetic on
	Zfull_aca = zeros(Complex{Float64},size(Z,1),size(Z,2)); Zfull_pinv = zeros(Complex{Float64},size(Z,1),size(Z,2))
	nels = length(Z_comp)
	number_els_compressed_aca = 0; #number of elements in the ACA compressed matrix	
	number_els_compressed_pinv = 0; #number of elements in the pinv compressed matrix

	time_pinv = 0.0; time_aca = 0.0
	total_time_uncompressed_cuda = 0.0
	total_time_uncompressed_cpu = 0.0


	for ii=1:nels #antes 1:nels
		self = Z_comp[ii]["self"]
		if Z_comp[ii]["comp"]==1.0
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			Z_local = Z[Int64.(m),Int64.(n)]
        #Ccol_samples = convert(Array{Cint,1},col_samples)
        #Crow_samples = convert(Array{Cint,1},row_samples)
			if size(m)==(); m=[m]; end
			if size(n)==(); n=[n]; end
			Z_local_cuda = user_impedance_cuda3(Int64.(m),Int64.(n),obj,EM_data)
			println("ii: ",ii," out of ",nels)
			#println("m: ",m)
			#println("n: ",n)
			#println("Z_local: ",Z_local[1])
			#println("Z_local_cuda: ",Z_local_cuda[1])
			#println("size(m): ",size(m))
			#println("size(n): ",size(n))
			#println("size(Z_local_cuda): ",size(Z_local_cuda))
			#println("size(Z_local): ",size(Z_local))
			(s1help,s2help) = size(Z_local_cuda)
			if s1help==1 || s2help==1
				println("Err cuda mat(case vec): ",norm(vec(Z_local_cuda)- vec(Z_local))/norm(vec(Z_local)) )
			else
				println("Err cuda mat(case normal): ",norm(Z_local_cuda- Z_local)/norm(Z_local))
			end

			U = Z_comp[ii]["U"]; V = Z_comp[ii]["V"]
			
			tt1 = @timed begin
			(Utrash,Vtrash) = C_ACA(1e-3,Z_local)
			end#timed
			time_aca = time_aca + tt1[2]

			#Cs = Z_comp[ii]["Cs"]; Us = Z_comp[ii]["Us"]; Rs = Z_comp[ii]["Rs"]
			order_aprox = size(U,2)
			#(Cs,Us,Rs) = cur_pinv2(Z_local,order_aprox)
			#(Cs,Us,Rs) = cur_pinv2(Z_local,min(length(m),length(n)))
			tt = @timed begin
			#(Cs,Us,Rs) = cur_compress(Z_local,1e-1)
			#(Cs,Us,Rs) = cuda_cur_compress(Z_local,1e-1)
			#(Cs,Us,Rs) = cuda_cur_compress2(Z_local,1e-1) #Estaba
			#println("about to cuda_cur_compress3")
			(Cs,Us,Rs,trash,trash,trash,time_cur3_partial) = cuda_cur_compress3(Int64.(m),Int64.(n),1e-1,obj,EM_data)
			#(Cs,Us,Rs,trash,trash,trash,time_cur3_partial) = cur_compress2(Int64.(m),Int64.(n),1e-1,obj,EM_data)

			#println("Error cuda_cur_compress3: ",norm(Cs*Us*Rs- Z_local_cuda)/norm(Z_local_cuda))
			end #timed
			#time_pinv = time_pinv + tt[2]
			time_pinv = time_pinv + time_cur3_partial
			m = Int64.(m); n = Int64.(n)

			if self == 1
				Zfull_aca[m,n] = U*V;
				Zfull_pinv[m,n] = Cs*Ur*Rs
				number_els_compressed_aca = number_els_compressed_aca + size(U,1)*size(U,2) + size(V,1)*size(V,2)
				number_els_compressed_pinv = number_els_compressed_pinv + size(Cs,1)*size(Cs,2) + size(Rs,1)*size(Rs,2)
			elseif self == 0
				Zfull_aca[m,n] = U*V; Zfull_aca[n,m] = transpose(U*V)
				Zfull_pinv[m,n] = Cs*Us*Rs; Zfull_pinv[n,m] = transpose(Cs*Us*Rs)
				number_els_compressed_aca = number_els_compressed_aca + 1*(size(U,1)*size(U,2) + size(V,1)*size(V,2))
				number_els_compressed_pinv = number_els_compressed_pinv +1*(size(Cs,1)*size(Cs,2) + size(Rs,1)*size(Rs,2))

			else
				error("self field incorrect in rebuild_matrix")
			end

			#Zfull_aca[m,n] = U*V
			#Zfull_pinv[m,n] = Cs*Us*Rs
			#Zfull_aca[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
			#Zfull_aca[m,n] = ones(Complex{Float64},length(m),length(n)); Zfull_pinv[m,n] = ones(Complex{Float64},length(m),length(n))

			#Zfull_pinv[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
		
			#err_skl_wrt_aca = norm(Cs*Us*Rs-U*V)/norm(U*V)
			#err_aca_wrt_Z   = norm(U*V-Z_local)/norm(Z_local)
			#err_skl_wrt_Z   = norm(Cs*Us*Rs-Z_local)/norm(Z_local)

			#Uoriginal = Z_comp_original[ii]["U"]; Voriginal = Z_comp_original[ii]["V"]
			#Usynth = dZ_comp[ii]["U"]; Vsynth = dZ_comp[ii]["V"]

			#errU = norm(Uoriginal-Usynth)/norm(Uoriginal)
			#errV = norm(Voriginal-Vsynth)/norm(Voriginal)
			#push!(vec_errU,errU); push!(vec_errV,errV)
			#push!(vec_err_skl_wrt_aca,err_skl_wrt_aca)
			#push!(vec_err_aca_wrt_Z  ,err_aca_wrt_Z)
			#push!(vec_err_skl_wrt_Z,err_skl_wrt_Z)
			#push!(vec_ii,ii)
		else
			println("ii: ",ii," out of ",nels)
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			m = Int64.(m); n = Int64.(n)
			if size(m)==(); m=[m]; end
			if size(n)==(); n=[n]; end
			Z_local = Z[m,n]
			println("UNCOMPRESSED CASE")
			println("size(m): ",size(m))
			println("size(n): ",size(n))
			println("m: ",m)
			println("n: ",n)
			#println("user_impedance_cuda3")
			#(Z_local_cuda3) = user_impedance_cuda3(Int64.(m),Int64.(n),obj,EM_data)
			#println("user_impedance_cuda5")
			(Z_local_cuda5,time_uncompressed_cuda) = user_impedance_cuda5(Int64.(m),Int64.(n),obj,EM_data)
			
			tt_time_uncompressed_cpu = @timed begin
				(Z_local_cpu) = user_impedance3(Int64.(m),Int64.(n),obj,EM_data)
			end
			#println("IN UNCOMPRESSED")
			#println("error between cuda5 and cuda3: ",norm(Z_local_cuda3- Z_local_cuda5)/norm(Z_local_cuda3))
			#println("error between cuda5 and cpu", norm(Z_local_cuda5- Z_local_cpu)/norm(Z_local_cpu))
			#println("error between cuda3 and cpu", norm(Z_local_cuda3- Z_local_cpu)/norm(Z_local_cpu))
			#println("error between cuda5 and direct: ",norm(Z_local_cuda5-Z_local)/norm(Z_local))
			#println("error between cpu and direct: ",norm(Z_local_cpu-Z_local)/norm(Z_local))


			total_time_uncompressed_cuda = total_time_uncompressed_cuda + time_uncompressed_cuda
			total_time_uncompressed_cpu  = total_time_uncompressed_cpu  + tt_time_uncompressed_cpu[2]

			if self == 1
				Zfull_aca[m,n] = Z_local;
				Zfull_pinv[m,n] = Z_local;
				number_els_compressed_aca = number_els_compressed_aca + size(Z_local,1)*size(Z_local,2)
				number_els_compressed_pinv = number_els_compressed_pinv + size(Z_local,1)*size(Z_local,2)
			elseif self == 0
				Zfull_aca[m,n] = Z_local; Zfull_aca[n,m] = transpose(Z_local)
				Zfull_pinv[m,n] = Z_local; Zfull_pinv[n,m] = transpose(Z_local)
				number_els_compressed_aca = number_els_compressed_aca + 1*(size(Z_local,1)*size(Z_local,2))
				number_els_compressed_pinv = number_els_compressed_pinv + 1*(size(Z_local,1)*size(Z_local,2))

			else
				error("self field incorrect in rebuild_matrix")
			end

		


			#Zfull_aca[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
			#Zfull_aca[m,n] = ones(Complex{Float64},length(m),length(n)); Zfull_pinv[m,n] = ones(Complex{Float64},length(m),length(n))
		end
	end
	compression_aca  = number_els_compressed_aca/(size(Zfull_aca,1)*size(Zfull_aca,2))
	compression_pinv = number_els_compressed_pinv/(size(Zfull_pinv,1)*size(Zfull_pinv,2))
	return Zfull_aca,  Zfull_pinv, compression_aca, compression_pinv, time_pinv, time_aca, total_time_uncompressed_cuda, total_time_uncompressed_cpu

end


function rebuild_matrix_cuda_timed(Z_comp,Z,obj,EM_data)
	#Where Z_comp should be the synthetic on
	Zfull_aca = zeros(Complex{Float64},size(Z,1),size(Z,2)); Zfull_pinv = zeros(Complex{Float64},size(Z,1),size(Z,2))
	nels = length(Z_comp)
	number_els_compressed_aca = 0; #number of elements in the ACA compressed matrix	
	number_els_compressed_pinv = 0; #number of elements in the pinv compressed matrix

	time_pinv = 0.0; time_aca = 0.0

	time_create_matrix = 0.0; 

	for ii=1:nels #antes 1:nels
		self = Z_comp[ii]["self"]
		if Z_comp[ii]["comp"]==1.0
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			Z_local = Z[Int64.(m),Int64.(n)]
        		#Ccol_samples = convert(Array{Cint,1},col_samples)
				#Crow_samples = convert(Array{Cint,1},row_samples)
			if size(m)==(); m=[m]; end
			if size(n)==(); n=[n]; end
			t_create_aux = @timed begin
			#Z_local_cuda = user_impedance_cuda3(Int64.(m),Int64.(n),obj,EM_data)
			Z_local_cuda = user_impedance3(Int64.(m),Int64.(n),obj,EM_data)
			end #timed
			time_create_matrix = time_create_matrix + t_create_aux[2]
			#println("ii: ",ii)
			#println("m: ",m)
			#println("n: ",n)
			#println("Z_local: ",Z_local[1])
			#println("Z_local_cuda: ",Z_local_cuda[1])
			#println("size(m): ",size(m))
			#println("size(n): ",size(n))
			#println("size(Z_local_cuda): ",size(Z_local_cuda))
			#println("size(Z_local): ",size(Z_local))
			#(s1help,s2help) = size(Z_local_cuda)
			#if s1help==1 || s2help==1
			#	println("Err cuda mat(case vec): ",norm(vec(Z_local_cuda)- vec(Z_local))/norm(vec(Z_local)) )
			#else
			#	println("Err cuda mat(case normal): ",norm(Z_local_cuda- Z_local)/norm(Z_local))
			#end

			U = Z_comp[ii]["U"]; V = Z_comp[ii]["V"]
			
			tt1 = @timed begin
			(Utrash,Vtrash) = C_ACA(1e-3,Z_local)
			end#timed
			time_aca = time_aca + tt1[2]

			#Cs = Z_comp[ii]["Cs"]; Us = Z_comp[ii]["Us"]; Rs = Z_comp[ii]["Rs"]
			order_aprox = size(U,2)
			#(Cs,Us,Rs) = cur_pinv2(Z_local,order_aprox)
			#(Cs,Us,Rs) = cur_pinv2(Z_local,min(length(m),length(n)))
			tt = @timed begin
			(Cs,Us,Rs) = cur_compress(Z_local,1e-1)
			#(Cs,Us,Rs) = cuda_cur_compress(Z_local,1e-1)
			end #timed
			time_pinv = time_pinv + tt[2]
			m = Int64.(m); n = Int64.(n)

			if self == 1
				Zfull_aca[m,n] = U*V;
				Zfull_pinv[m,n] = Cs*Ur*Rs
				number_els_compressed_aca = number_els_compressed_aca + size(U,1)*size(U,2) + size(V,1)*size(V,2)
				number_els_compressed_pinv = number_els_compressed_pinv + size(Cs,1)*size(Cs,2) + size(Rs,1)*size(Rs,2)
			elseif self == 0
				Zfull_aca[m,n] = U*V; Zfull_aca[n,m] = transpose(U*V)
				Zfull_pinv[m,n] = Cs*Us*Rs; Zfull_pinv[n,m] = transpose(Cs*Us*Rs)
				number_els_compressed_aca = number_els_compressed_aca + 1*(size(U,1)*size(U,2) + size(V,1)*size(V,2))
				number_els_compressed_pinv = number_els_compressed_pinv +1*(size(Cs,1)*size(Cs,2) + size(Rs,1)*size(Rs,2))

			else
				error("self field incorrect in rebuild_matrix")
			end

			#Zfull_aca[m,n] = U*V
			#Zfull_pinv[m,n] = Cs*Us*Rs
			#Zfull_aca[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
			#Zfull_aca[m,n] = ones(Complex{Float64},length(m),length(n)); Zfull_pinv[m,n] = ones(Complex{Float64},length(m),length(n))

			#Zfull_pinv[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
		
			#err_skl_wrt_aca = norm(Cs*Us*Rs-U*V)/norm(U*V)
			#err_aca_wrt_Z   = norm(U*V-Z_local)/norm(Z_local)
			#err_skl_wrt_Z   = norm(Cs*Us*Rs-Z_local)/norm(Z_local)

			#Uoriginal = Z_comp_original[ii]["U"]; Voriginal = Z_comp_original[ii]["V"]
			#Usynth = dZ_comp[ii]["U"]; Vsynth = dZ_comp[ii]["V"]

			#errU = norm(Uoriginal-Usynth)/norm(Uoriginal)
			#errV = norm(Voriginal-Vsynth)/norm(Voriginal)
			#push!(vec_errU,errU); push!(vec_errV,errV)
			#push!(vec_err_skl_wrt_aca,err_skl_wrt_aca)
			#push!(vec_err_aca_wrt_Z  ,err_aca_wrt_Z)
			#push!(vec_err_skl_wrt_Z,err_skl_wrt_Z)
			#push!(vec_ii,ii)
		else
			m = Z_comp[ii]["m"]; n = Z_comp[ii]["n"]
			m = Int64.(m); n = Int64.(n)
			Z_local = Z[m,n]
			
			if self == 1
				Zfull_aca[m,n] = Z_local;
				Zfull_pinv[m,n] = Z_local;
				number_els_compressed_aca = number_els_compressed_aca + size(Z_local,1)*size(Z_local,2)
				number_els_compressed_pinv = number_els_compressed_pinv + size(Z_local,1)*size(Z_local,2)
			elseif self == 0
				Zfull_aca[m,n] = Z_local; Zfull_aca[n,m] = transpose(Z_local)
				Zfull_pinv[m,n] = Z_local; Zfull_pinv[n,m] = transpose(Z_local)
				number_els_compressed_aca = number_els_compressed_aca + 1*(size(Z_local,1)*size(Z_local,2))
				number_els_compressed_pinv = number_els_compressed_pinv + 1*(size(Z_local,1)*size(Z_local,2))

			else
				error("self field incorrect in rebuild_matrix")
			end

		


			#Zfull_aca[m,n] = copy(Z_local); Zfull_pinv[m,n] = copy(Z_local)
			#Zfull_aca[m,n] = ones(Complex{Float64},length(m),length(n)); Zfull_pinv[m,n] = ones(Complex{Float64},length(m),length(n))
		end
	end
	compression_aca  = number_els_compressed_aca/(size(Zfull_aca,1)*size(Zfull_aca,2))
	compression_pinv = number_els_compressed_pinv/(size(Zfull_pinv,1)*size(Zfull_pinv,2))
	return Zfull_aca,  Zfull_pinv, compression_aca, compression_pinv, time_pinv, time_aca, time_create_matrix

end



function execute_pseudomain()
	asd = 6;
	Casd = convert(Int64,6)
	kk = ccall((:pseudo_main,"./C_ACA_candidate.so"),Int64,(Int64,),Casd);	
	return kk;
end

function cur_smart(A,nsamples)
        (m,n) = size(A)
        col_samples = sample(1:n,nsamples)
        row_samples = sample(1:m,nsamples)
        C = A[:,col_samples];
        #R = A[row_samples,:];
        #U = inv(A[col_samples,row_samples])
        row_samples = [];
        for ii=1:nsamples
                (maxval,maxind) = findmax(abs.(C[:,ii]))
                push!(row_samples,maxind)
        end
        R = A[row_samples,:];
        U = inv(A[row_samples,col_samples])
        return C, U, R, row_samples, col_samples
end

function cur_smart_pinv(A,nsamples)
        (m,n) = size(A)
        col_samples = sample(1:n,nsamples)
        row_samples = sample(1:m,nsamples)
        C = A[:,col_samples];
        #R = A[row_samples,:];
        #U = inv(A[col_samples,row_samples])
        row_samples = [];
        for ii=1:nsamples
                (maxval,maxind) = findmax(abs.(C[:,ii]))
                push!(row_samples,maxind)
        end
        R = A[row_samples,:];
        U = pinv(A[row_samples,col_samples])
        return C, U, R, row_samples, col_samples
end

function cur_smart_pinv2(A,nsamples)
	#Arregla problema de permutaciones
        (m,n) = size(A)
        col_samples = sample(1:n,nsamples)
	col_samples = sort(col_samples)
        #row_samples = sample(1:m,nsamples)
        C = A[:,col_samples];
        #R = A[row_samples,:];
        #U = inv(A[col_samples,row_samples])
        row_samples = [];
        for ii=1:nsamples
                (maxval,maxind) = findmax(abs.(C[:,ii]))
                push!(row_samples,maxind)
        end
        R = A[row_samples,:];
        U = pinv(A[row_samples,col_samples])
        return C, U, R, row_samples, col_samples
end


function cur_pinv(A,nsamples)
        (m,n) = size(A)
        col_samples = sample(1:n,nsamples)
        row_samples = sample(1:m,nsamples)
        C = A[:,col_samples];
        #R = A[row_samples,:];
        #U = inv(A[col_samples,row_samples])
       # row_samples = [];
       # for ii=1:nsamples
       #         (maxval,maxind) = findmax(abs.(C[:,ii]))
       #         push!(row_samples,maxind)
       # end
        R = A[row_samples,:];
        U = pinv(A[row_samples,col_samples])
        return C, U, R, row_samples, col_samples
end


function cur_pinv2(A,nsamples)
	#Here we sort the columns
        #(m,n) = size(A)  #OJO
	sizeA = size(A)  #OJO	!!!
	if length(sizeA)==1
		m=sizeA[1]
		n=1
	else
		m = sizeA[1]
		n = sizeA[2]
	end

        #col_samples = sample(1:n,nsamples)
        #row_samples = sample(1:m,nsamples)
	col_samples = sample(1:n,nsamples,replace=false)
        row_samples = sample(1:m,nsamples,replace=false)

	col_samples = sort(col_samples); row_samples = sort(row_samples);
        C = A[:,col_samples];
        #R = A[row_samples,:];
        #U = inv(A[col_samples,row_samples])
       # row_samples = [];
       # for ii=1:nsamples
       #         (maxval,maxind) = findmax(abs.(C[:,ii]))
       #         push!(row_samples,maxind)
       # end
        R = A[row_samples,:];
        #U = pinv(A[row_samples,col_samples])
	U = my_pinv(A[row_samples,col_samples])
        return C, U, R, row_samples, col_samples
end


function cur_pinv3(nsamples,r1,r2,obj,EM_data)
	#Here we sort the columns
        #(m,n) = size(A)  #OJO
#	sizeA = size(A)  #OJO	!!!
#	if length(sizeA)==1
#		m=sizeA[1]
#		n=1
#	else
#		m = sizeA[1]
#		n = sizeA[2]
#	end

	m = length(r1)
	n = length(r2)

        #col_samples = sample(1:n,nsamples)
        #row_samples = sample(1:m,nsamples)
	col_samples = sample(1:n,nsamples,replace=false)
        row_samples = sample(1:m,nsamples,replace=false)

	col_samples = sort(col_samples); row_samples = sort(row_samples);
	col_samples = r2[col_samples]; row_samples = r1[row_samples]

        #C = A[:,col_samples];
        #R = A[row_samples,:];
        #U = inv(A[col_samples,row_samples])
       # row_samples = [];
       # for ii=1:nsamples
       #         (maxval,maxind) = findmax(abs.(C[:,ii]))
       #         push!(row_samples,maxind)
       # end
        #R = A[row_samples,:];
        #U = pinv(A[row_samples,col_samples])
	#U = my_pinv(A[row_samples,col_samples])
	C = user_impedance3(Int64.(r1),Int64.(col_samples),obj,EM_data)
	U = user_impedance3(Int64.(row_samples),Int64.(col_samples),obj,EM_data)	
	U = my_pinv(U)
	R = user_impedance3(Int64.(row_samples),Int64.(r2),obj,EM_data)

        return C, U, R, row_samples, col_samples
end


function cuda_cur_pinv2(A,nsamples,tolerance)
        #(m,n) = size(A)
       	sizeA = size(A)  #OJO	!!!
	if length(sizeA)==1
		m=sizeA[1]
		n=1
	else
		m = sizeA[1]
		n = sizeA[2]
	end
	
	m = convert(Cint,m)
        n = convert(Cint,n)
        nsamples = convert(Cint,nsamples)
        Ctolerance = convert(Cdouble,tolerance)
        col_samples = sample(1:n,nsamples,replace=false)
        row_samples = sample(1:m,nsamples,replace=false)
        col_samples = sort(col_samples); row_samples = sort(row_samples);
        Ccol_samples = convert(Array{Cint,1},col_samples)
        Crow_samples = convert(Array{Cint,1},row_samples)

        #Cexact = A[:,col_samples];
        #Rexact = A[row_samples,:];
        #Uexact = pinv(A[row_samples,col_samples])

        C = zeros(ComplexF64,m,nsamples)
        U = zeros(ComplexF64,nsamples,nsamples)
        R = zeros(ComplexF64,nsamples,n)

        ress = ccall((:aux_pinv_compression,"./functions_cur_external_cuda.so"),
                     Int64,
                     (Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{Int64},Ptr{Int64},Cint,Cint,Cint,Cdouble),
                     C,
                     U,
                     R,
                     A,
                     Crow_samples,
                     Ccol_samples,
                     nsamples,
                     m,
                     n,
                     Ctolerance)
       # println("m: ",m)
       # println("n: ",n)
       # println("nsamples: ",nsamples)
       # println("A: ",A)
       # println("row_samples: ",row_samples)
       # println("col_samples: ",col_samples)
       # println("error C: ",norm(Cexact-C)/norm(Cexact))
       # println("C: ",C)
       # println("error U: ",norm(Uexact-U)/norm(Uexact))
       # println("error R: ",norm(Rexact-R)/norm(Rexact))
       # println("R: ",R)
       # println("error intersection: ",norm(U-A[row_samples,col_samples])/norm(A[row_samples,col_samples]))
       # println("U: ",U)

        return C, U, R, row_samples, col_samples
end


function cur_compress(A,tol)
	#(M,N) = size(A)
	sizeA = size(A)  #OJO	!!!
	if length(sizeA)==1
		M=sizeA[1]
		N=1
	else
		M = sizeA[1]
		N = sizeA[2]
	end

	vtest = rand(N) + im*rand(N)
	D = min(M,N)
	if D<10
		start_order = 1
	else
		start_order = div(D,10)
	end

	#par = log2(D,start_order)
	#Aumentamos el orden por potencias de 2
	#vv = start_order*2 .^(0:Int64(floor(par))); push!(vv,D)

	flag_err=0; flag_size=0
	err_app = Inf
	current_size = start_order

	(C,U,R) = cur_pinv2(A,current_size)
	Avtest_old = C*U*R*vtest;
	if 2*current_size < D
		current_size = 2*current_size
	else
		current_size = D

	end
	
	row_samples = 0; col_samples = 0

	flag_size = 0
	while (err_app > tol) & (flag_size == 0)
		println("Entering loop")
		println("err before comp:", err_app)
		(C,U,R, row_samples, col_samples) = cur_pinv2(A,current_size)
		Avtest_new = C*U*R*vtest
		err_app = norm(Avtest_old-Avtest_new)/norm(Avtest_new)
		println("err after comp:",err_app)
		Avtest_old = copy(Avtest_new)
	#	if current_size > D
	#		error("current_size larger than D, someting went wrong")
	#	elseif 2*current_size < D
	#		current_size = 2*current_size
	#	elseif 2*current_size >= D
	#		current_size = D
	#		flag_size = 1
	#	else
	#		error("something weird happened")
	#	end

		if current_size > D
			error("current_size larger than D, something went wrong")
		elseif current_size==D
			flag_size=1
		elseif current_size < D
			if 2*current_size <= D
				current_size = 2*current_size
			else
				current_size = D
			end
		else
			error("something mathematically impossible happened. You are in the wrong universe")
		end
/
	end
	
	return C, U, R, row_samples, col_samples, err_app

end

function cur_compress2(r1,r2,tol,obj::obj_struct,EM_data::EM_data_struct)
	#Aquí vamos a hacer que se tengan que generar los elementos de la matriz
	#(M,N) = size(A)
	#sizeA = size(A)  #OJO	!!!
	#if length(sizeA)==1
	#	M=sizeA[1]
	#	N=1
	#else
	#	M = sizeA[1]
	#	N = sizeA[2]
	#end
	M = length(r1)
	N = length(r2)


	vtest = rand(N) + im*rand(N)
	D = min(M,N)
	if D<10
		start_order = 1
	else
		start_order = div(D,10)
	end

	#par = log2(D,start_order)
	#Aumentamos el orden por potencias de 2
	#vv = start_order*2 .^(0:Int64(floor(par))); push!(vv,D)

	flag_err=0; flag_size=0
	err_app = Inf
	current_size = start_order

	#(C,U,R) = cur_pinv2(A,current_size)
	(C,U,R) = cur_pinv3(current_size,r1,r2,obj,EM_data)
	Avtest_old = C*U*R*vtest;
	if 2*current_size < D
		current_size = 2*current_size
	else
		current_size = D

	end
	
	row_samples = 0; col_samples = 0

	flag_size = 0
	while (err_app > tol) & (flag_size == 0)
		println("Entering loop")
		println("err_app (before compression): ",err_app)
		#(C,U,R, row_samples, col_samples) = cur_pinv2(A,current_size)
		(C,U,R, row_samples, col_samples) = cur_pinv3(current_size,r1,r2,obj,EM_data)

		Avtest_new = C*U*R*vtest
		err_app = norm(Avtest_old-Avtest_new)/norm(Avtest_new)
		println("err_app (after comrpession): ",err_app)
	#	if current_size > D
	#		error("current_size larger than D, someting went wrong")
	#	elseif 2*current_size < D
	#		current_size = 2*current_size
	#	elseif 2*current_size >= D
	#		current_size = D
	#		flag_size = 1
	#	else
	#		error("something weird happened")
	#	end

		if current_size > D
			error("current_size larger than D, something went wrong")
		elseif current_size==D
			flag_size=1
		elseif current_size < D
			if 2*current_size <= D
				current_size = 2*current_size
			else
				current_size = D
			end
		else
			error("something mathematically impossible happened. You are in the wrong universe")
		end

	end
	
	return C, U, R, row_samples, col_samples, err_app, Inf

end


function cuda_cur_compress(A,tol)
	#(M,N) = size(A)
	sizeA = size(A)  #OJO	!!!
	if length(sizeA)==1
		M=sizeA[1]
		N=1
	else
		M = sizeA[1]
		N = sizeA[2]
	end

	vtest = rand(N) + im*rand(N)
	D = min(M,N)
	if D<10
		start_order = 1
	else
		start_order = div(D,10)
	end

	#par = log2(D,start_order)
	#Aumentamos el orden por potencias de 2
	#vv = start_order*2 .^(0:Int64(floor(par))); push!(vv,D)

	flag_err=0; flag_size=0
	err_app = Inf
	current_size = start_order

	#(C,U,R) = cur_pinv2(A,current_size)
	(C,U,R) = cuda_cur_pinv2(A,current_size,1e-10)
	Avtest_old = C*U*R*vtest;
	if 2*current_size < D
		current_size = 2*current_size
	else
		current_size = D

	end
	
	row_samples = 0; col_samples = 0

	flag_size = 0
	while (err_app > tol) & (flag_size == 0)
		#println("Entering loop")
		(C,U,R, row_samples, col_samples) = cur_pinv2(A,current_size)
		Avtest_new = C*U*R*vtest
		err_app = norm(Avtest_old-Avtest_new)/norm(Avtest_new)
		Avtest_old = copy(Avtest_new)
	#	if current_size > D
	#		error("current_size larger than D, someting went wrong")
	#	elseif 2*current_size < D
	#		current_size = 2*current_size
	#	elseif 2*current_size >= D
	#		current_size = D
	#		flag_size = 1
	#	else
	#		error("something weird happened")
	#	end

		if current_size > D
			error("current_size larger than D, something went wrong")
		elseif current_size==D
			flag_size=1
		elseif current_size < D
			if 2*current_size <= D
				current_size = 2*current_size
			else
				current_size = D
			end
		else
			error("something mathematically impossible happened. You are in the wrong universe")
		end

	end
	
	return C, U, R, row_samples, col_samples, err_app

end


function cuda_cur_compress2(A,tol)
	#En este el calculo del error tambien se lleva a cabo en CUDA, de forma modular
	#(M,N) = size(A)
	sizeA = size(A)  #OJO	!!!
	if length(sizeA)==1
		M=sizeA[1]
		N=1
	else
		M = sizeA[1]
		N = sizeA[2]
	end

	vtest = rand(N) + im*rand(N)
	D = min(M,N)
	if D<10
		start_order = 1
	else
		start_order = div(D,10)
	end

	#par = log2(D,start_order)
	#Aumentamos el orden por potencias de 2
	#vv = start_order*2 .^(0:Int64(floor(par))); push!(vv,D)

	flag_err=0; flag_size=0
	err_app = Inf
	current_size = start_order

	#(C,U,R) = cur_pinv2(A,current_size)
	(C,U,R) = cuda_cur_pinv2(A,current_size,1e-10)
	Avtest_old = C*U*R*vtest;
	if 2*current_size < D
		current_size = 2*current_size
	else
		current_size = D

	end
	
	row_samples = 0; col_samples = 0

	flag_size = 0
	while (err_app > tol) & (flag_size == 0)
		println("Entering loop")
		#(C,U,R, row_samples, col_samples) = cur_pinv2(A,current_size)
		(C,U,R) = cuda_cur_pinv2(A,current_size,1e-10) #OJO!!! cambiada
		Avtest_new = C*U*R*vtest
		err_app = norm(Avtest_old-Avtest_new)/norm(Avtest_new)
		(err_app_cuda,Avtest_new_cuda) = compute_error_cuda(C,U,R,Avtest_old,vtest)
		println("NNN error Avtest_new normal and cuda: ",norm(Avtest_new-Avtest_new_cuda)/norm(Avtest_new))
		println("NNN error of error: ",abs(err_app_cuda-err_app)/abs(err_app))
		Avtest_old = copy(Avtest_new)
	#	if current_size > D
	#		error("current_size larger than D, someting went wrong")
	#	elseif 2*current_size < D
	#		current_size = 2*current_size
	#	elseif 2*current_size >= D
	#		current_size = D
	#		flag_size = 1
	#	else
	#		error("something weird happened")
	#	end

		if current_size > D
			error("current_size larger than D, something went wrong")
		elseif current_size==D
			flag_size=1
		elseif current_size < D
			if 2*current_size <= D
				current_size = 2*current_size
			else
				current_size = D
			end
		else
			error("something mathematically impossible happened. You are in the wrong universe")
		end

	end
	
	return C, U, R, row_samples, col_samples, err_app

end
#(err_app_cuda,Avtest_new_cuda) = compute_error_cuda(C,U,R,Avtest_old)


function cuda_cur_compress3(r1,r2,tol,obj::obj_struct,EM_data::EM_data_struct)
	#En este las operaciones importantes se llevan a cabo en CUDA con automtiming
	#En este el calculo del error tambien se lleva a cabo en CUDA, de forma modular
	
	time_total = 0.0

	pinv_tol = 1e-10
	M = length(r1)
	N = length(r2)

	vtest = rand(N) + im*rand(N)
	D = min(M,N)
	if D<10
		start_order = 1
	else
		start_order = div(D,10)
	end

	flag_err=0; flag_size=0
	err_app = Inf
	current_size = start_order

	#(C,U,R) = cur_pinv2(A,current_size)
	#(C,U,R) = cuda_cur_pinv2(A,current_size,1e-10)
	Avtest_old = ones(M) + im*zeros(M)
	(C,U,R,row_samples,col_samples,Avtest_new,err_here,time_partial) = pure_cuda_compression2(vtest,Avtest_old,current_size,pinv_tol,r1,r2,tol,obj,EM_data)
	time_total = time_total + time_partial
	#Avtest_old = C*U*R*vtest;
	Avtest_old = copy(Avtest_new)

	if 2*current_size < D
		current_size = 2*current_size
	else
		current_size = D

	end
	
	row_samples = 0; col_samples = 0

	flag_size = 0
	while (err_app > tol) & (flag_size == 0)
		println("Entering loop")
		println("Current size: ",current_size)
		println("err_app before compression: ",err_app)
		#
		#(C,U,R, row_samples, col_samples) = cur_pinv2(A,current_size)
		#Avtest_new = C*U*R*vtest
		#err_app = norm(Avtest_old-Avtest_new)/norm(Avtest_new)
		#(err_app_cuda_independent,Avtest_new_cuda) = compute_error_cuda(C,U,R,Avtest_old,vtest)
		(C,U,R,row_samples,col_samples,Avtest_new,err_app,time_partial) = pure_cuda_compression2(vtest,Avtest_old,current_size,pinv_tol,r1,r2,tol,obj,EM_data)
		#(err_app_cuda_independent,Avtest_new_cuda) = compute_error_cuda(C,U,R,Avtest_old,vtest)
		#Avtest_new_cpu = C*U*R*vtest
		#err_app_cpu = norm(Avtest_old-Avtest_new_cpu)/norm(Avtest_new_cpu)
		println("err_app after compression: ",err_app)
		#println("err_app_cpu: ",err_app_cpu)
		#println("err_app_cuda_independent: ",err_app_cuda_independent)
		#println("error Avtest cpu vs cuda independent: ",norm(Avtest_new_cpu-Avtest_new_cuda)/norm(Avtest_new_cpu))
		#println("error Avtest cpu vs cuda integrated: ",norm(Avtest_new_cpu-Avtest_new)/norm(Avtest_new_cpu))
		#println("error Avtest cuda integrated vs cuda independent: ",norm(Avtest_new_cuda-Avtest_new)/norm(Avtest_new_cpu))
		#println("Avtest_new_cuda: ",Avtest_new_cuda[1:10])
		#println("Avtest_new: ",Avtest_new[1:10])
		#println("Avtest_new_cpu: ",Avtest_new_cpu[1:10])

		time_total = time_total + time_partial
		
		Avtest_old = copy(Avtest_new)
	#	if current_size > D
	#		error("current_size larger than D, someting went wrong")
	#	elseif 2*current_size < D
	#		current_size = 2*current_size
	#	elseif 2*current_size >= D
	#		current_size = D
	#		flag_size = 1
	#	else
	#		error("something weird happened")
	#	end

		if current_size > D
			error("current_size larger than D, something went wrong")
		elseif current_size==D
			flag_size=1
		elseif current_size < D
			if 2*current_size <= D
				current_size = 2*current_size
			else
				current_size = D
			end
		else
			error("something mathematically impossible happened. You are in the wrong universe")
		end

	end
	
	return C, U, R, row_samples, col_samples, err_app, time_total

end




function compute_error_cuda(C,U,R,Avtest_old,vtest)
	#Calcula  Avtest_new = C*U*R*vtest y el error norm(Avtest_old-Avtest_new)/norm(Avtest_new)
	
	N = length(Avtest_old)

	Avtest_new = zeros(ComplexF64,N,1)
#	m = convert(Cint,m)
	(M1,N1) = size(C)
	(trash,M2) = size(R)
	M1 = convert(Cint,M1)
	N1 = convert(Cint,N1)
	M2 = convert(Cint,M2)

	#errhere = ccall((:compute_error_compression,"./functions_cur_external_cuda.so"), OJO!!!
	errhere = ccall((:compute_error_compression,"./functionsExternal_cuda.so"),
			Cdouble,
			(Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Ptr{ComplexF64},Cint,Cint,Cint),
			C,U,R,Avtest_old,Avtest_new,vtest,M1,N1,M2)
	return errhere, Avtest_new

end


function invert_svec(svec,tol)
        svec_inverted = zeros(length(svec))
        for ii=1:length(svec)
                if svec[ii]>=tol
                        svec_inverted[ii] = 1/svec[ii]
                end
        end
        return svec_inverted
end



function my_pinv(A)
	tol = 1e-10
        F = svd(A)
        U = F.U
        S = F.S
        Vt = F.Vt
        Sinv = invert_svec(S,tol)
        (m,n) = size(A)
        return Vt'*Diagonal(Sinv)*U'
end

function retrieve_indices(a,along)
	v = zeros(Int64,length(a))
	for ii=1:length(a)
		v[ii] = findall(x->x==a[ii],along)[1]
	end
	return vec(v)
end

function loadMatlab_interaction()
        return mxcall(:loadMatlab_interaction,1)
end




#function 

function get_case_and_divfactor()
	f = open("parameters.txt")
	lines = readlines(f)
	close(f)
	case_code = parse(Int64,lines[1])
	divfactor = parse(Int64,lines[2])
	return case_code, divfactor
end

function assign_lambda_Nedges(case_code)

	if case_code == 1
		lambda = 4
		Nedges = 12*4^3
	elseif case_code == 2
		lambda = 2
		Nedges = 12*4^4
	elseif case_code == 3
		lambda = 1
		Nedges = 12*4^5
	else 
		error("Error in assigning case_code")
	end
	
	return lambda, Nedges

end

############################################ MAIN ############################################################


###loading from parameters

(case_code,divfactor) = get_case_and_divfactor()

(lambda,Nedges) = assign_lambda_Nedges(case_code)

println("case_code: ",case_code)
println("lambda: ",lambda)
println("Nedges: ",Nedges)
println("divfactor: ",divfactor)

global divfactor


#lambda = 2
#lambda = 1;
#lambda = 0.5;
#lambda = 0.25
distance_parameter = 1;
k = 2*pi/lambda;
eta = 120*pi;
field = 1; # EFIE->1, MFIE->2, CFIE->3
radius = 1.0* distance_parameter;

Rint_s = 1;       # MoM Integration radius (meters). Rint=0 is enough if basis functions are very small.
Rint_f = Rint_s;
Ranal_s = 1;
corr_solid = 0;
flag = 0;

println("tag1")

#obj = obj_struct(nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing) #Initialize object
EM_data = EM_data_struct(lambda,k,eta,field,Rint_s,Rint_f,Ranal_s,corr_solid,flag)
#(vertex,topol,trian,edges,un,ds,ln,cent,N) = matlab_object_spheres_interaction()
#obj.vertex = vertex
#obj.topol = topol
#obj.trian = trian
#obj.edges = edges
#obj.un = un
#obj.ds = ds
#obj.ln = ln
#obj.cent = cent
#obj.N = N
#
#number_edges = N;
#
#println("tag2")
#
#
##Z = user_impedance3(1:number_edges/2,number_edges/2+1:number_edges,obj,EM_data);
#
#println("tag3")

#Zmatlab = loadMatlab_interaction()

#println("tag4")

#println("error Z: ",norm(Z-Zmatlab)/norm(Zmatlab))

d_spheres = 12* distance_parameter;
#Nedges = 12*4^3  #For lambda = 2
#Nedges = 12*4^4  #For lambda = 1
#Nedges = 4*12*4^4  #For lambda = 0.5
#Nedges = 4*12*4^5  #For lambda = 0.25




obj2 = obj_struct(nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing) #Initialize object

println("tag5")
(vertex2,topol2,trian2,edges2,un2,ds2,ln2,cent2,N2) = matlab_object_spheres_interaction2(lambda,radius,d_spheres,Nedges)

obj2.vertex = vertex2
obj2.topol = topol2
obj2.trian = trian2
obj2.edges = edges2
obj2.un = un2
obj2.ds = ds2
obj2.ln = ln2
obj2.cent = cent2
obj2.N = N2

number_edges = N2


println("tag6")

#Z2 = user_impedance3(1:number_edges/2,number_edges/2+1:number_edges,obj,EM_data);
(Z2,time_trash) = user_impedance_cuda5(1:number_edges/2,number_edges/2+1:number_edges,obj2,EM_data)
println("direct compression")
(Cdirect,Udirect,Rdirect,trash1,trash2,trash3) = cur_compress(Z2,1e-1)
println("error direct: ",norm(Cdirect*Udirect*Rdirect-Z2)/norm(Z2))
println("size(Cdirect): ",size(Cdirect))


println("tag7")

#println("error Z2: ",norm(Z2-Zmatlab)/norm(Zmatlab))


m = 1:number_edges/2
n = number_edges/2+1:number_edges

println("about to cuda compress")

(Cs_gpu,Us_gpu,Rs_gpu,trash,trash,trash,time_cur3_partial_cuda) = cuda_cur_compress3(Int64.(m),Int64.(n),1e-1,obj2,EM_data)

println("about to cpu compress")

time_cpu = @timed begin
(Cs_cpu,Us_cpu,Rs_cpu,trash,trash,trash,time_cur3_partial_cpu) = cur_compress2(Int64.(m),Int64.(n),1e-1,obj2,EM_data)
end

println("about to compute errors")

println("error GPU: ",norm(Cs_gpu*Us_gpu*Rs_gpu-Z2)/norm(Z2))
println("error CPU: ",norm(Cs_cpu*Us_cpu*Rs_cpu-Z2)/norm(Z2))

println("time GPU (ms): ",time_cur3_partial_cuda)
println("time CPU (s): ",time_cpu[2])


println("size(Cs_gpu): ",size(Cs_gpu))
println("size(Cs_cpu): ",size(Cs_cpu))



########################a partir de aquí lo antiguo
#
##Incident field
#
#th_i = 90 #180 Antes
#ph_i = 0
#rot_EH = 90
#
##Electromagnetic
##lambda = 1
#lambda = 0.0035*20; #alemndra pequeña
##lambda = 0.0019*20; #almendra grande
#
#k = 2*pi/lambda
#eta = 120*pi
##field = 1 #EFIE -> 1, MFIE -> 2, CFIE -> 3
#field = 3;
#
## Rint_s = 0.2       # MoM Integration radius (meters). Rint=0 is enough if basis functions are very small.
## Rint_f = Rint_s
## Ranal_s = 0
##Rint_s = 1.0       # MoM Integration radius (meters). Rint=0 is enough if basis functions are very small.
##Rint_s = 10.0 #El origiinal
#Rint_s = 0.2*lambda; #0.2
#
#Rint_f = Rint_s
#Ranal_s = 0.0; #1.0
##Ranal_s = 10.0 #el original
#Ranal_s = 0.0;
#corr_solid = 0  ##Esto esta bien que sea Int
##flag = 0 #el original
#flag = 1
#
#EM_data = EM_data_struct(lambda,k,eta,field,Rint_s,Rint_f,Ranal_s,corr_solid,flag)
#geom = "sphere"
##Ne = 192*4
##Ne = 192*4 #original
#Ne = 3072
##radio = 2e-4
##radio=5*lambda
#radio = 0.4; #Obtenido heurísticamente del caso que queremos emular
#param = param_struct(radio,Ne)
#
##Old object (sphere)
#####################
##obj = sphere(param)
##displacement = 3.0*radio
###This modifies the object to contain twin spheres
###twin_spheres!(obj,displacement)
##obj = get_edge_jmr(obj)
##obj.N = convert(Int32,length(obj.ln))
##N = obj.N
######################
#
#obj = obj_struct(nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing) #Initialize object
#
#(vertex,topol,trian,edges,un,ds,ln,cent,N) = matlab_object_almond()
#obj.vertex = vertex
#obj.topol = topol
#obj.trian = trian
#obj.edges = edges
#obj.un = un
#obj.ds = ds
#obj.ln = ln
#obj.cent = cent
#obj.N = N
#
#
#
#
#(Ei,Hi) = test_fields(obj ,th_i, ph_i, rot_EH, k, eta)
#
##println("aaa = ",aaa)
#
##println("Time normal")
##@time Ze = user_impedance(1:N,1:N,obj,EM_data)
##println("Time pseudoparallel")
##@time Ze = user_impedance_pseudoparallel(1:N,1:N,obj,EM_data,10)
##println("Time imrpoved")
##@time Ze = user_impedance_improved(1:N,1:N,obj,EM_data)
#
##Je = -Ze\Ei;
##Zmatlab = matlabZ();
#
#println("CUDA version")
#
##Zmatlab = matlabZ_almond(); #estaba
##save("Zmatlab.jld","Zmatlab",Zmatlab)
#println("about to load matrix")
#dddd = load("Zmatlab.jld")
#Zmatlab = dddd["Zmatlab"]
#println("Matlab matrix loaded")
#
##Zcuda = user_impedance_cuda3(collect(1:N),collect(1:N),obj,EM_data)
##Zcuda4 = user_impedance_cuda4(collect(1:N),collect(1:N),obj,EM_data)
#
##println("error cuda: ",norm(Zmatlab-Zcuda)/norm(Zmatlab))
##println("error cuda4: ",norm(Zmatlab-Zcuda4)/norm(Zmatlab))
#
#
#
#
#m = [2716.0, 2751.0, 2752.0, 2763.0, 2765.0, 2766.0, 2767.0, 2811.0, 2812.0, 2841.0, 2842.0, 2843.0, 2900.0, 2901.0, 2928.0, 2931.0, 2932.0, 2947.0, 2948.0, 2949.0, 2993.0, 2994.0, 3018.0, 3019.0, 3020.0, 3079.0, 3080.0, 3081.0, 3111.0, 3112.0, 3156.0, 3157.0, 3158.0, 3167.0, 3168.0, 3169.0, 3183.0, 3184.0, 3186.0, 3248.0, 3249.0, 3310.0, 3311.0, 3340.0, 3365.0, 3366.0, 3367.0, 3368.0, 3369.0, 3452.0, 3454.0, 3535.0, 3536.0, 3579.0]
# n = [2684.0, 2685.0, 2700.0, 2822.0, 2919.0, 3099.0, 3144.0, 3424.0, 3979.0, 3980.0, 3981.0, 3982.0, 3983.0, 3984.0, 3985.0, 3986.0, 3987.0, 3988.0, 3989.0, 3990.0, 3991.0, 3992.0, 3993.0, 3994.0, 3995.0, 3996.0, 3997.0, 3998.0, 3999.0, 4000.0, 4001.0, 4002.0, 4003.0, 4005.0, 4006.0, 4007.0, 4008.0, 4010.0, 4011.0, 4012.0, 4013.0, 4016.0, 4017.0, 4018.0, 4019.0, 4020.0, 4028.0, 4029.0, 4191.0, 4192.0, 4193.0, 4194.0, 4195.0, 4197.0, 4202.0, 4212.0, 4213.0, 4356.0, 4412.0, 4413.0, 4414.0]
#m = [2716.0, 2751.0, 2752.0, 2763.0, 2765.0, 2766.0, 2767.0, 2811.0]
#n = [2684.0, 2685.0, 2700.0, 2822.0, 2919.0, 3099.0, 3144.0, 3424.0, 3979.0]
#
#
##Ze = user_impedance_cuda3(Int64.(m),Int64.(n),obj,EM_data)
#Ze4 = user_impedance_cuda4(Int64.(m),Int64.(n),obj,EM_data)
#
#(Cs3,Us3,Rs3,rs3,cs3,trash,trash) = cuda_cur_compress3(Int64.(m),Int64.(n),1e-1,obj,EM_data)
#m_mini = collect(1:length(m))
#n_mini = collect(1:length(n))
#rs_mini = retrieve_indices(Int64.(rs3),Int64.(m))
#cs_mini = retrieve_indices(Int64.(cs3),Int64.(n))
#
#println("========Computing Intersection")
##Uinter = user_impedance_cuda4(Int64.(rs3),Int64.(cs3),obj,EM_data)
#
#
#println("Us3: ",Us3)
#
#println("rs3: ",rs3)
#println("cs3: ",cs3)
#println("m_mini: ",m_mini)
#println("n_mini: ",n_mini)
#
#Cs_synt = Ze4[Int64.(m_mini),cs_mini];
#Us_synt = Ze4[rs_mini,cs_mini];
#Rs_synt = Ze4[rs_mini,Int64.(n_mini)]
#println("Us_synt: ",Us_synt)
#println("size(Cs_synt): ",size(Cs_synt))
#println("size(Cs3): ", size(Cs3))
#println("size(Us_synt): ",size(Us_synt))
#println("size(Us3): ", size(Us3))
#println("size(Rs_synt): ",size(Rs_synt))
#println("size(Rs3): ", size(Rs3))
#
#
#println("Err Cs_synt: ",norm(Cs_synt-Cs3)/norm(Cs_synt))
#println("Err Us_synt: ",norm(Us_synt-Us3)/norm(Us_synt))
#println("Err Rs_synt: ",norm(Rs_synt-Rs3)/norm(Rs_synt))
##println("Err Uinter: ",norm(Us_synt-Uinter)/norm(Us_synt))
#
#
#
#
#
##save("comparisons_cuda3vs4.jld","Ze",Ze,"Ze4",Ze4)
#
##Zee = pure_cuda_compression(Int64.(m),Int64.(n),1e-3,obj,EM_data)
##println("Ze: ",Ze)
#
##println("")
##println("===========================")
##println("")
#
##println("Ze4: ",Ze4)
##println("error Ze Ze4: ",norm(Ze-Ze4)/norm(Ze))
#
#
#aca_threshold = 1e-3;
#
#
#
#
###(Z_comp, vertextrash,topoltrash,triantrash,edgestrash,untrash,dstrash,lntrash,centtrash,Ntrash) = get_Zcomp_data();
#println("tag1")
#(Z_comp, vertextrash,topoltrash,triantrash,edgestrash,untrash,dstrash,lntrash,centtrash,Ntrash,Ei) = get_Zcomp_data_extra_almond();
#println("tag1.1")
####staba
##println("tag2")
##tcomp_aca = @timed (Z_comp_synth) = get_synthetic_Zcomp(Z_comp,Zmatlab,aca_threshold)
##tcomp_skl = @timed (Z_comp_skl) = get_synthetic_Zcomp_skeleton(Z_comp,Zmatlab,aca_threshold)
##println("tcomp_aca: ",tcomp_aca[2])
##println("tcomp_skl: ",tcomp_skl[2])
##println("tag3")
##(vec_err_skl_wrt_aca, vec_err_aca_wrt_Z, vec_err_skl_wrt_Z, vec_ii) = compare_aca_vs_skl(Z_comp_skl,Zmatlab)
###estaba
#
#
##ppp = plot(1:length(vec_err_skl_wrt_aca),vec_err_skl_wrt_aca,yaxis=:log,label="skl_vs_aca")
##ppp = plot!(1:length(vec_err_skl_wrt_aca),vec_err_aca_wrt_Z,yaxis=:log,label="aca_vs_Z")
##ppp = plot!(1:length(vec_err_skl_wrt_Z),vec_err_skl_wrt_Z,yaxis=:log,label="skl_vs_Z")
###staba2
##println("tag4")
##(Zfull_aca,Zfull_pinv, compression_aca, compression_pinv,time_pinv,time_aca) = rebuild_matrix(Z_comp,Zmatlab)
##(Zfull_aca,Zfull_pinv, compression_aca, compression_pinv,time_pinv,time_aca) = rebuild_matrix_cuda(Z_comp,Zmatlab,obj,EM_data)
#(Zfull_aca,Zfull_pinv, compression_aca, compression_pinv,time_pinv,time_aca,tu_cuda,tu_cpu) = rebuild_matrix_cuda2(Z_comp,Zmatlab,obj,EM_data)
#
##(Zfull_aca,Zfull_pinv, compression_aca, compression_pinv,time_pinv,time_aca,time_create) = rebuild_matrix_cuda_timed(Z_comp,Zmatlab,obj,EM_data)
#println("time_pinv: ",time_pinv)
#println("time_aca: ",time_aca)
#println("time_uncompressed_cuda: ",tu_cuda)
#println("time_uncompressed_cpu: ",tu_cpu)
##println("time_create: ",time_create)
#
#println("tag5")
#println("error ACA: ",norm(Zmatlab-Zfull_aca)/norm(Zmatlab))
#println("error pinv: ",norm(Zmatlab-Zfull_pinv)/norm(Zmatlab))
#println("error ACA-pinv: ",norm(Zfull_pinv-Zfull_aca)/norm(Zfull_aca))
###staba2
#
##(Z_comp_synth) = get_synthetic_Zcomp(Z_comp,Zmatlab,aca_threshold)
##(kk1,kk2,kk3,kk4) = get_C_synthetic_Zcomp2(Z_comp,Zmatlab,aca_threshold) #Funciona
#
#println("Acabamos get_C_synthetic_Zcomp2")
#
#
##(errU,errV) = compare_compressions(Z_comp,Z_comp_synth)
##
##(Zfull_aca,Zfull_pinv, compression_aca, compression_pinv,time_pinv,time_aca) = rebuild_matrix(Z_comp_synth,Zmatlab)
##
##(U50,V50,m,n) = getmatrix50();
##(U50C,V50C) = C_ACA(aca_threshold,Zmatlab[m,n])
##Z50 = Ze[m,n];
##(Csharp,Usharp,Rsharp) = cur_pinv2(Z50,24)
##println("Error ACA: ", norm(Z50-U50*V50)/norm(Z50))
##println("Error CUR: ",norm(Z50-Csharp*Usharp*Rsharp)/norm(Z50))
##println("U matlab vs U C: ",opnorm(U50-U50C)/opnorm(U50))
##println("V matlab vs V C: ",opnorm(V50-V50C)/opnorm(V50))
##
##println("Computational finished")
#
#
#
##save("result_Je.jld","Je",Je)
#
##create_matlab_file(real(Je),"real_Je.txt")
##create_matlab_file(imag(Je),"imag_Je.txt")
##
##create_matlab_file(real(Ei),"real_Ei.txt")
##create_matlab_file(imag(Ei),"imag_Ei.txt")
##
##create_matlab_file(real(Ze[:]),"real_Ze.txt")
##create_matlab_file(imag(Ze[:]),"imag_Ze.txt")
##
##println("Computation has successfully finished")
#
#
#
##
##println(" Comparing solution with whole system and by using Macro Basis Functions (Characteristic Basis Functions)\n")
##println(" WHOLE SYSTEM:")
##@time x_exact = -Ze\Ei
##println(" MACRO BASIS FUNCTIONS: ")
##(x_CBF, compression_rate) = solve_by_CBF(-Ze,Ei)
##println(" Error of CBF: sum(abs.(x_exact-x_CBF))/(sum(abs.(x_exact)))  =  ", sum(abs.(x_exact-x_CBF))/(sum(abs.(x_exact))) )
##println(" Compression rate: (size_whole_matrix)/(size_compressed_matrix) = ",compression_rate)
##
###save("matriz_single.jld","Ze_single",Ze)
##
#
