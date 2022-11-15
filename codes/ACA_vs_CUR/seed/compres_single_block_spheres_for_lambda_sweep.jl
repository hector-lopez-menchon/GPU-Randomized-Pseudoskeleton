#This script considers the interaction between two PEC spheres, and compares several compression techniques

#Corresponds to example 4.1.1 of the associated paper



using LinearAlgebra
using JLD
using MATLAB
using StatsBase
include("ACA_julia.jl")

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

function user_impedance3(r1,r2,obj::obj_struct,EM_data::EM_data_struct)::Array{Complex{Float64},2}
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
	Zr = zeros(Cdouble,n_r1,n_r2)
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
	return Zr+im*Zi
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


function matlab_object_spheres_interaction(lambda,radius,d_spheres,Nedges)
        #Wrapper to the matlab routine that computes the double sphere object
	(vertex,topol,trian,edges,un,ds,ln,cent,N) = mxcall(:object_wrapper_spheres_interaction2,9,lambda,radius,1.0*d_spheres,1.0*Nedges)
        topol = floor.(Int32,topol)
        trian = floor.(Int32,trian)
        edges = floor.(Int32,edges)
        ln = vec(ln)
        N = floor.(Int32,N)
        return vertex,topol,trian,edges,un,ds,ln,cent,N
end



function matlab_ACA(aca_threshold,A)
	#Calls the matlab ACA through a wrapper
	(U,V) = mxcall(:ACA_wrapper,2,aca_threshold,A);
	return U,V;
end

function matlab_postcompress(C,U,R,tol)
        (Cp,Up,Rp) = mxcall(:postcompress_wrapper,3,C,U,R,tol)
        return Cp, Up, Rp
end

function C_ACA(aca_threshold,A)
	#Note: this is not written for performance, as the wrapper code is very inefficient. Only for checking correctness
	sizeA = size(A) 
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

function ACA_julia_timed(ACA_tol,m,n,obj,EM_data)
        time_here =  @timed begin
                (U,V) = ACA_julia(ACA_tol,m,n,obj,EM_data)
        end
        return U,V, time_here[2]
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

function cur_pinv3(nsamples,r1,r2,obj,EM_data)
	#Applies randomized pseudoskeleton approximation
	m = length(r1)
	n = length(r2)

	col_samples = sample(1:n,nsamples,replace=false)
        row_samples = sample(1:m,nsamples,replace=false)

	col_samples = sort(col_samples); row_samples = sort(row_samples);
	col_samples = r2[col_samples]; row_samples = r1[row_samples]

       	C = user_impedance3(Int64.(r1),Int64.(col_samples),obj,EM_data)
	U = user_impedance3(Int64.(row_samples),Int64.(col_samples),obj,EM_data)	
	U = my_pinv(U)
	R = user_impedance3(Int64.(row_samples),Int64.(r2),obj,EM_data)

        return C, U, R, row_samples, col_samples
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

function get_case_lambda()
	f = open("parameters.txt")
	lines = readlines(f)
	close(f)
	case_code = parse(Int64,lines[1])
	return case_code
end

function assign_lambda(case_code,av_edge)
	lambda = 0.0
	if case_code==1
		lambda = 5* av_edge
	elseif case_code == 2
		lambda = 10 * av_edge
	elseif case_code == 3
		lambda = 15 *av_edge
	elseif case_code == 4
		lambda = 20 * av_edge
	end
	return lambda
end

function sweep_CUR_compress(r1,r2,obj::obj_struct,EM_data::EM_data_struct,vpc,Z)
	#Compresses the block Z applying several compression ranges r using the randomized pseudoskeleton approximation method

	#r1: row indices of the subblock to be compute with respect to the total matrix
	#r2: column indices of the subblock to be compute with respect to the total matrix
	#obj_struct: object storing object data
	#EM_data: object storing electromagnetic data
	#vpc: vector percentage compression
	#Z: subblock to be compressed
	ss = length(vpc)
	err_vec = zeros(ss)
	time_vec = zeros(ss)
	nels_vec = zeros(ss)
	sZ = length(r1)
	normZ2 = norm(Z2)

	for ii=1:ss
		println("sweeping CUR ii=",ii," out of ",ss)
		println("=====================")
		nels_here = Int64(ceil(1.0*sZ*vpc[ii]/100))
		time_here = @timed begin
		(C,U,R) = cur_pinv3(nels_here,r1,r2,obj,EM_data)
		end
		err_here = norm(C*U*R-Z2)/normZ2
		err_vec[ii] = err_here
		time_vec[ii] = time_here[2]
		nels_vec[ii] = nels_here
	end
	return err_vec, time_vec, nels_vec
end

function sweep_CUR_compress_postcompressed(r1,r2,obj::obj_struct,EM_data::EM_data_struct,vpc,Z)
        #vpc, vector percentage compression
        ss = length(vpc)
        err_vec = zeros(ss)
        time_vec = zeros(ss)
        nels_vec = zeros(ss)
        nels_post_vec = zeros(ss)
        err_post_vec = zeros(ss)
        sZ = length(r1)
        normZ2 = norm(Z2)

        for ii=1:ss
                println("sweeping CUR ii=",ii," out of ",ss)
                println("=====================")
                nels_here = Int64(ceil(1.0*sZ*vpc[ii]/100))
                time_here = @timed begin
                (C,U,R) = cur_pinv3(nels_here,r1,r2,obj,EM_data)
                end
                err_here = norm(C*U*R-Z2)/normZ2
                err_vec[ii] = err_here
                time_vec[ii] = time_here[2]
                nels_vec[ii] = nels_here
                (Cpost,Upost,Rpost) = matlab_postcompress(C,U,R,0.1*err_here);
                nels_post_here = size(Cpost,2)
                err_post_here = norm(Cpost*Upost*Rpost-Z2)/norm(Z2)
                nels_post_vec[ii] = nels_post_here
                err_post_vec[ii] = err_post_here
        end
        return err_vec, time_vec, nels_vec, err_post_vec, nels_post_vec
end



function sweep_ACA_compress(r1,r2,obj::obj_struct,EM_data::EM_data_struct,vtols,Z)
	#Compresses the block Z applying several compression ranges r using the adaptive cross approximation method

	#r1: row indices of the subblock to be compute with respect to the total matrix
	#r2: column indices of the subblock to be compute with respect to the total matrix
	#obj_struct: object storing object data
	#EM_data: object storing electromagnetic data
	#vpc: vector percentage compression
	#Z: subblock to be compressed


	ss = length(vtols)
	err_vec = zeros(ss)
	time_vec = zeros(ss)
	nels_vec = zeros(ss)
	sZ = length(r1)
	normZ2 = norm(Z)

	for ii=1:ss
		println("sweeping ACA ii=",ii," out of ",ss)
		println("=====================")

		#nels_here = Int64(ceil(1.0*sZ/vpc[ii]))
		time_here = @timed begin
		#(C,U,R) = cur_pinv3(nels_here,r1,r2,obj,EM_data)
		#(U,V) = matlab_ACA(vtols[ii],Z)
		(U,V,time_here) = ACA_julia_timed(vtols[ii],r1,r2,obj,EM_data)
		end
		err_here = norm(U*V-Z2)/normZ2
		err_vec[ii] = err_here
		time_vec[ii] = time_here[2]
		nels_vec[ii] = size(U,2)
	end
	return err_vec, time_vec, nels_vec
end

function sweep_CUR_compress_postcompressed(r1,r2,obj::obj_struct,EM_data::EM_data_struct,vpc,Z)
        #vpc, vector percentage compression
        ss = length(vpc)
        err_vec = zeros(ss)
        time_vec = zeros(ss)
        nels_vec = zeros(ss)
        nels_post_vec = zeros(ss)
        err_post_vec = zeros(ss)
        sZ = length(r1)
        normZ2 = norm(Z2)

        for ii=1:ss
                println("sweeping CUR ii=",ii," out of ",ss)
                println("=====================")
                nels_here = Int64(ceil(1.0*sZ*vpc[ii]/100))
                time_here = @timed begin
                (C,U,R) = cur_pinv3(nels_here,r1,r2,obj,EM_data)
                #(C,U,R) = cur_pinv2(Z,nels_here)
                end
                err_here = norm(C*U*R-Z2)/normZ2
                err_vec[ii] = err_here
                time_vec[ii] = time_here[2]
                nels_vec[ii] = nels_here
                (Cpost,Upost,Rpost) = matlab_postcompress(C,U,R,0.1*err_here);
                nels_post_here = size(Cpost,2)
                err_post_here = norm(Cpost*Upost*Rpost-Z2)/norm(Z2)
                nels_post_vec[ii] = nels_post_here
                err_post_vec[ii] = err_post_here
        end
        return err_vec, time_vec, nels_vec, err_post_vec, nels_post_vec
end

function sweep_ACA_compress_postcompressed(r1,r2,obj::obj_struct,EM_data::EM_data_struct,vtols,Z)
        #vtols, vector desired_tolerances
        #ss = length(vpc)
        ss = length(vtols)
        err_vec = zeros(ss)
        time_vec = zeros(ss)
        nels_vec = zeros(ss)
        nels_post_vec = zeros(ss)
        err_post_vec = zeros(ss)
        sZ = length(r1)
        normZ2 = norm(Z)

        for ii=1:ss
                println("sweeping ACA ii=",ii," out of ",ss)
                println("=====================")

                (U,V,time_here) = matlab_ACA_timed(vtols[ii],Z)
                (U,V,time_here) = ACA_julia_timed(vtols[ii],r1,r2,obj,EM_data)

                err_here = norm(U*V-Z2)/normZ2
                err_vec[ii] = err_here
                #time_vec[ii] = time_here[2]
                time_vec[ii] = time_here

                nels_vec[ii] = size(U,2)
                (Upost,Spost,Vpost) = matlab_postcompress(U,collect(Diagonal(ones(Complex{Float64},size(U,2)))),V,0.1*err_here);
                nels_post_here = size(Upost,2)
                err_post_here = norm(Upost*Spost*Vpost-Z2)/norm(Z2)
                nels_post_vec[ii] = nels_post_here
                err_post_vec[ii] = err_post_here
        end
        return err_vec, time_vec, nels_vec, err_post_vec, nels_post_vec
end



############################################ MAIN ############################################################



case_code = get_case_lambda(); #Load from file parameters.txt a variable defining lambda
lambda = assign_lambda(case_code,0.06048) 
distance_parameter = 1;
k = 2*pi/lambda;
eta = 120*pi;
field = 1; # EFIE->1, MFIE->2, CFIE->3
#radius = 1.5* distance_parameter;
radius = 1.0

#Parameters for Petrov-Galerkin discretizations
Rint_s = 1;       # MoM Integration radius (meters). Rint=0 is enough if basis functions are very small.
Rint_f = Rint_s;
Ranal_s = 1;
corr_solid = 0;
flag = 0;

println("tag1")

d_spheres = 12* distance_parameter;
#Nedges = 12*4^3 
#Nedges = 12*4^4 
Nedges = 4*12*4^4 
#Nedges = 4*12*4^5 

#Initialize object 
obj2 = obj_struct(nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing) #Initialize object

println("tag5")
(vertex2,topol2,trian2,edges2,un2,ds2,ln2,cent2,N2) = matlab_object_spheres_interaction(lambda,radius,d_spheres,Nedges)

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

#Compute the low-rank impedance matrix to be compressed
(Z2) = user_impedance3(Int64.(1:number_edges/2),Int64.(number_edges/2+1:number_edges),obj2,EM_data)

m = 1:number_edges/2
n = number_edges/2+1:number_edges

vec_perc_compression = [0.1, 0.5, 1, 5]

(errvecCUR,timevecCUR,nels_vecCUR) = sweep_CUR_compress(Int64.(m),Int64.(n),obj2,EM_data,vec_perc_compression,Z2)
(errvecCUR,timevecCUR,nels_vecCUR,errvecCUR_post,nels_vecCUR_post) = sweep_CUR_compress_postcompressed(Int64.(m),Int64.(n),obj2,EM_data,vec_perc_compression,Z2)


(errvecACA,timevecACA,nels_vecACA) = sweep_ACA_compress(Int64.(m),Int64.(n),obj2,EM_data,errvecCUR,Z2)
(errvecACA,timevecACA,nels_vecACA,errvecACA_post,nels_vecACA_post) = sweep_ACA_compress_postcompressed(Int64.(m),Int64.(n),obj2,EM_data,errvecCUR,Z2)


#save results to disk
#save("results_sweep_compression.jld","errvecCUR",errvecCUR,"timevecCUR",timevecCUR,"nels_vecCUR",nels_vecCUR,"errvecACA",errvecACA,"timevecACA",timevecACA,"nels_vecACA",nels_vecACA)
save("results_sweep_compression.jld","errvecCUR",errvecCUR,"timevecCUR",timevecCUR,"nels_vecCUR",nels_vecCUR,"errvecCUR_post",errvecCUR_post,"nels_vecCUR_post",nels_vecCUR_post,
     "errvecACA",errvecACA,"timevecACA",timevecACA,"nels_vecACA",nels_vecACA,"errvecACA_post",errvecACA_post,"nels_vecACA_post",nels_vecACA_post)




