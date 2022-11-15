module MatlabFiles

export create_matlab_file

function create_matlab_file(A_arg,name)
	#prerequisite, A should be array of floats
	A = vec(A_arg)
	open(name,"w") do f
		for i=1:length(A)
			aa=A[i]
			write(f," $aa ")
		end
	end
	return  
end

end


