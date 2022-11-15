
#In this example we compare a GPU implementation of the CUR algorithm with a CPU implementation of the same technique



using LinearAlgebra
using JLD
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

function user_impedance4(r1,r2,obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
	#Basado en user_impedance3, pero implementa una nueva funcion de C, impedance_matrix_freed, que libera eficientemente la memoria reservada
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
	res = ccall((:impedance_matrix_freed,"./functionsExternal.so"),
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

	return Zr+im*Zi
end


function user_impedance4_chunked(r1,r2,obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
        #Lo mismo que user_impedance3_chunked pero llama a user_impedance4. Es eficiente en terminos de uso de memoria
        #Esta funcion aplica user_impedance3 por partes para evitar gasto ineficiente de memoria

        #Integers describing sizes
        n_r1 = length(r1)
        n_r2 = length(r2)

        Ztotal = zeros(Complex{Float64},n_r1,n_r2)

        for ii=1:n_r2
                if rem(ii,100)==0; println("ii = ",ii," out of ",length(r2)); GC.gc(); end
                Zhere = user_impedance4(r1,Int64.(collect(r2[ii]:r2[ii])),obj,EM_data)
                #Zhere = rand(length(r1))+im*rand(length(r1))
                Ztotal[:,ii] = Zhere
        end

        return Ztotal
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

	return Ztotal, time_tardado
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
	czols_cent = size(cent,2)

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
	C = user_impedance4_chunked(Int64.(r1),Int64.(col_samples),obj,EM_data)
	U = user_impedance4_chunked(Int64.(row_samples),Int64.(col_samples),obj,EM_data)	
	U = my_pinv(U)
	R = user_impedance4_chunked(Int64.(row_samples),Int64.(r2),obj,EM_data)

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

        return C, U, R, row_samples, col_samples
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

############################################ MAIN ############################################################


###loading from parameters

(case_code,divfactor) = get_case_and_divfactor()

(lambda,Nedges) = assign_lambda_Nedges(case_code)

println("case_code: ",case_code)
println("lambda: ",lambda)
println("Nedges: ",Nedges)
println("divfactor: ",divfactor)

global divfactor

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

EM_data = EM_data_struct(lambda,k,eta,field,Rint_s,Rint_f,Ranal_s,corr_solid,flag)


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

final_time_cuda = time_cur3_partial_cuda
final_time_cpu = time_cpu[2]

save("results.jld","final_time_cuda",final_time_cuda,"final_time_cpu",final_time_cpu)


