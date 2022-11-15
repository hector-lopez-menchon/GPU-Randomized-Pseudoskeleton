
function ACA_julia(ACA_thres,m,n,OG_data,EM_data)

	M = length(m)
	N = length(n)

	#If Z is a vector, there is nothing to compress
	#if M==1 || N==1; U = user_impedance(m,n,OG_data,EM_data); V = 1; return U,V; end
	#if M==1 || N==1; U = pseudo_user_impedance(m,n,Z); V = 1; return U,V; end
	if M==1 || N==1; U = user_impedance3(m,n,OG_data,EM_data); V = 1; return U,V; end

	
	J = zeros(Int64,N); #Indices of columns picked up from Z
	I = zeros(Int64,M); #Indices of rows picked up from Z
	i = collect(2:M); #Row indices to search for maximum in R
	j = collect(1:N); #Column indices to search for maximum in R


	## Initialization

	# Initialize the 1st row index I(1) = 1
	I[1] = 1;

	# Initialize the 1st row of the approximate error matrix
	Rik = user_impedance3([m[I[1]]],n,OG_data,EM_data);


	# Find the 1st column index J(1)
	(trash,col) = findmax(abs.(Rik[j]))
	J[1] = j[col[1]];

	j = remove_index(j,J[1]);

	# First row of V
	V = Rik/Rik[J[1]];

	# Initialize the 1st column of the approximate error matrix
	Rjk = user_impedance3(m,[n[J[1]]],OG_data,EM_data);


	# First column of U
	U = Rjk;

	# Norm of (approximate) Z, to test error
	normZ = norm(U)^2 * norm(V)^2;

	# Find 2nd row index I(2)
	#row = find( abs(Rjk(i)) == max(abs(Rjk(i))) );
	(trash,row) = findmax(abs.(Rjk[i]))
	I[2] = i[row[1]];
	i = remove_index(i,I[2]);


	# Iteration
	for k=2:min(M,N)

		
	    	# Update (Ik)th row of the approximate error matrix:
		Rik = user_impedance3([m[I[k]]],n,OG_data,EM_data)

		auxvar = - collect(transpose(U[I[k],:]))*V;
		Rik = Rik + auxvar

	   	# Find kth column index Jk
		(trash,col) = findmax(abs.(Rik[j]));

	    	J[k] = j[col[1]];
		j = remove_index(j,J[k]);

	    	# Terminate if R(I(k),J(k)) == 0
		if(Rik[J[k]] == 0)
			return U, V
			break
	    	end

	    	# Set k-th row of V equal to normalized error
	    	Vk = Rik/Rik[J[k]];

	    	# Update (Jk)th column of the approximate error matrix
		Rjk = user_impedance3(m,[n[J[k]]],OG_data,EM_data) - U*V[:,J[k]]



	    	# Set k-th column of U equal to updated error
	    	Uk = Rjk;

	    	# Norm of approximate Z
		normZ = normZ + 2*sum(real.((U'*Uk).*transpose(Vk*V'))) + norm(Uk)^2*norm(Vk)^2
 

	   	# Update U and V
	    	U = [U Uk]; V = [V; Vk];

	   	# Check convergence
	   	if norm(Uk)*norm(Vk) <= ACA_thres*sqrt(normZ)
			return U, V
			break
	   	end

	   	if k==min(M,N)
			return U, V
			break
	   	end

	    	# Find next row index
		(trash,row) = findmax(abs.(Rjk[i]))
		I[k+1] = i[row[1]];
		i = remove_index(i,I[k+1]);
	end

	return U,V
end



function remove_index(i,e)
	N = length(i)
	i2 = zeros(N-1);
	row = findall(i.==e)
	i2[1:row[1]-1] = i[1:row[1]-1]
	i2[row[1]:N-1] = i[row[1]+1:N]
	return Int64.(i2)
end

function pseudo_user_impedance(m,n,Z)
	M = length(m)
	N = length(n)

	return reshape(Z[m,n],M,N)
end



