
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>


void remove_index(int * i, int e, int * nels);
void printU(double complex **U,int M,int ACAorder);
void printV(double complex **V,int N,int ACAorder);
void fillrange(int *v, int vsize, int initial_value);
int find_max_abs(double complex *A,int nrows, int ncols, int *indices, int n_indices);
void add_a_line(double complex **pp, int x, int nels, double complex *v);
void mult_and_row(double complex **pp, int x, int nels, double complex *v,double complex alpha);
void UrowV(double complex *vec, double complex **U, int row_of_U, int Ufixedsize, int Uorder, double complex **V, int Vfixedsize, int Vorder);
void UVcol(double complex *vec, double complex **U, int Ufixedsize, int Uorder, double complex **V, int col_of_V, int Vfixedsize, int Vorder);
void substract(double complex *vec, double complex *a, double complex *b, int N);
double compute_norm(double complex *W, int lengthW);
double normU(double complex **U, int Ufixedsize, int Uorder);
double normV(double complex **V, int Vfixedsize, int Vorder);
double adhoc_operation(double complex **U, double complex *Uk,int Ufixedsize, int Uorder, double complex **V, double complex *Vk,int Vfixedsize, int Vorder);
int pseudoACA(double complex **U, double complex **V, int * ACA_order, double complex *Z,int M, int N, double ACA_thres);
//int synthetic_compress(int *matrix_m, int *matrix_n,int *vlength_m,int *vlength_n, int nels, int maxm, int maxn);
int synthetic_compress(double complex *Z, int mZ, int nZ, int *matrix_m, int *matrix_n,int *vlength_m,int *vlength_n, int nels, int maxm, int maxn);

void remove_index(int * i, int e, int *nels){
	//Adaptation of the remove index function from the MATLAB ACA
	//This function does not reduce the size of the array, but simply copies the new array on the first position of the older one
	//i: set of indices
	//e: index to be removed
	//nels: original number of elements
	int to_be_removed=0; int flag_found=0;
	for (int ii=0; ii<*nels; ii++){
		if (i[ii]==e){
			to_be_removed=ii;
			flag_found=1;
		}
	}
	if (flag_found==0){
		printf("Critical error: remove index badly applied. Index not found.\n");
	}
	for (int ii=to_be_removed; ii<*nels-1; ii++){
		i[ii]=i[ii+1];
	}
	*nels = *nels - 1;
	return;
}

void printU(double complex **U,int M,int ACAorder){
	printf("Displaying a U kind matrix\n");
	for (int ii=0; ii<ACAorder; ii++){
		for (int jj=0; jj<M; jj++){
			printf("[%f + %f i] ",creal(U[ii][jj]),cimag(U[ii][jj]));
		}
		printf("\n");
	}
	return;
}

void printV(double complex **V,int N,int ACAorder){
	printf("Displaying a V kind matrix\n");
	for (int ii=0; ii<ACAorder; ii++){
		for (int jj=0; jj<N; jj++){
			printf("[%f + %f i] ",creal(V[ii][jj]),cimag(V[ii][jj]));
		}
		printf("\n");
	}
	return;
}


void pseudo_ui_row(double complex *v, int m_ind, int *n,int size_row,int N, double complex *Z, int nrowsZ, int ncolsZ){
	//Computes a row of the matrix
	//m_ind; row to be computed
	//n: vector of indices
	//size_row: should be one (useless)
	//N: lenght of the n vector
	//Z: matrix to be samples
	//nrowsZ: number of rows of Z
	//ncolsZ: number of cols of Z
	
	for (int ii=0; ii<N; ii++){
		v[ii] = Z[m_ind+nrowsZ*n[ii]];
	}
	return;
}


void pseudo_ui_col(double complex *v, int *m, int n_ind,int M,int size_col, double complex *Z, int nrowsZ, int ncolsZ){
	//Computes a row of the matrix
	//m: vector of indices
	//n_ind: col to be computed
	//size_col: should be one (useless)
	//M: lenght of the n vector
	//Z: matrix to be samples
	//nrowsZ: number of rows of Z
	//ncolsZ: number of cols of Z
	
	for (int ii=0; ii<M; ii++){
		printf("ii=%d. M=%d. m[ii]=%d\n",ii,M,m[ii]);
		v[ii] = Z[nrowsZ*n_ind+m[ii]];
	}
	return;
}


void fillrange(int *v, int vsize, int initial_value){
	//Fills an array with consecituve integers starting from a givven initial value
	//v: array to e filled
	//vsize: lenght of that array
	//initial_value: value of the first element
	for (int ii=0; ii<vsize; ii++){
		v[ii] = ii+initial_value;
	}
	return;
}

int find_max_abs(double complex *A,int nrows, int ncols, int *indices, int n_indices){
	//Finds the maximum with absolute value in an array, over a restriscted set of indices
	//A: matrix for findinx max element abs
	//nrows: number of rows of the matrix
	//ncols: number of cols of the matrix
	//indices: vector of indicies
	//n_indices: length of the vector of indices
	//nota: puede que algún parámetro sea innecesario
	int position_max=-1;
	double max_val=0.0;
	for (int ii=0; ii<n_indices; ii++){
			if (cabs(A[indices[ii]])>max_val){
			max_val = cabs(A[indices[ii]]);
			position_max = ii;
		}
	}
	return position_max;
}

void add_a_line(double complex **pp, int x, int nels, double complex *v){
	//Adds a columns (or a line, in general) to the 2D array based on double pointer
	//pp: 2D array double pointer
	//x: index corresponding to the line we will add
	//nels: number of rows, the fixed length
	//v: col to add
	pp[x] = (double complex *)malloc(nels  * sizeof(double complex));
	for (int ii=0; ii<nels; ii++){
		pp[x][ii] = v[ii];
	}
	return;
}

void mult_line(int numels, double complex *v, double complex alpha){
	//Multiplies the vector v for the scalar alpha
	//numels: number of elements of v
	//v: vector
	//alpha: scalar ofinterest
	for (int ii=0; ii<numels; ii++){
		v[ii] = alpha*v[ii];
	}
	return;
}


void mult_and_row(double complex **pp, int x, int nels, double complex *v,double complex alpha){
	//Nota, usar esta función es ineficiente porque implica copiar muchos datos, a tener en cuenta para optimización
	//Adds a row to pp and multiplies by alpha
	//pp: array to array of pointers
	//x: index of the row to be added
	//nels: number of elements of the vector to be added
	//v: vector to be added
	//alpha: scalar to multiply for
	pp[x] = (double complex *)malloc(nels * sizeof(double complex));
	for (int ii=0; ii<nels; ii++){
		pp[x][ii] = alpha*v[ii];
	}
	return; 
}

void UrowV(double complex *vec, double complex **U, int row_of_U, int Ufixedsize, int Uorder, double complex **V, int Vfixedsize, int Vorder){
	//performs the operation vec=U[row_of_U,:]*V
	//vec:  where the result is stored
	//U: U matrix
	//row_of_U: row of U that we want to multiply
	//Ufixedsize: number of rows of U. Is fixed. Corresponds to M
	//Uorder: number of cols of U that we have added
	//V: Vmatrix
	//Vfixedsize: number of cols of V. Is fixed. Corresponds to N
	//Vorder: number of rows that we have added to V
	
	if (Vorder!=Uorder){
		printf("Error in UrowV\n");
		return;
	}

	double complex aux;
	
	//Se podrían modificar el orden de los bucles para evitar muchos pagafaults
	for (int ii=0; ii<Vfixedsize; ii++){
		aux = 0.0 +0.0 * I;
		for (int jj=0; jj<Vorder; jj++){
			aux = aux + U[jj][row_of_U]*V[jj][ii];
		}
		
		vec[ii] = aux;
	}
	return;
}

void UVcol(double complex *vec, double complex **U, int Ufixedsize, int Uorder, double complex **V, int col_of_V, int Vfixedsize, int Vorder){
	//performs the operation vec=U*V[:,col_of_V]
	//vec:  where the result is stored
	//U: U matrix
	//Ufixedsize: number of rows of U. Is fixed. Corresponds to M
	//Uorder: number of cols of U that we have added
	//V: Vmatrix
	//col_of_V: column of V that we want to mulitply
	//Vfixedsize: number of cols of V. Is fixed. Corresponds to N
	//Vorder: number of rows that we have added to V
	if (Vorder!=Uorder){
		printf("Error in UVcol\n");
		return;
	}
	
	double complex aux;
	for (int ii=0; ii<Vfixedsize; ii++){
		aux = 0.0 + 0.0 * I;
		for (int jj=0; jj<Vorder; jj++){
			aux = aux + U[jj][ii]*V[jj][col_of_V];
		}
		vec[ii] = aux;
	}
	return;
}


void substract(double complex *vec, double complex *a, double complex *b, int N){
	//Performs the operation v = a - b,
	//vec: where the result is stored
	//a: first vector
	//b: second vector
	//N: length of a and b
	for (int ii=0; ii<N; ii++){
		vec[ii] = a[ii] - b[ii];
	}
	return;
}

double compute_norm(double complex *W, int lengthW){
	double aux = 0.0;
	for (int ii=0; ii<lengthW; ii++){
		aux = aux + pow(cabs(W[ii]),2);
	}
	return sqrt(aux);
}

double normU(double complex **U, int Ufixedsize, int Uorder){
	//Computes the norm of a U type matrix
	//U: U matrix
	//Ufixedsize: number of rows of U (usually M)
	//Uorder: number of cols that we have added to U
	double max = 0.0;
	double aux = 0.0;
	printf("Within normU\n");
	for (int ii=0; ii<Ufixedsize; ii++){
		aux = 0.0;
		for (int jj=0; jj<Uorder; jj++){
			aux = aux + pow(cabs(U[jj][ii]),2);
		}	
		if (aux > max){
			max = aux;
		}
	}
	return sqrt(max);
}

double normV(double complex **V, int Vfixedsize, int Vorder){
	//Compute the norm of a V type matrix
	//V: Vmatrix
	//Vfixedsize: number of cols of V (in general N)
	//Vorder: number of rows that we have added to U
	double max = 0.0;
	double aux = 0.0;
	for (int ii=0; ii<Vorder; ii++){
		aux = compute_norm(V[ii],Vfixedsize);
		if (aux > max){
			max = aux;
		}
	}
	return sqrt(max);
}

double adhoc_operation(double complex **U, double complex *Uk,int Ufixedsize, int Uorder, double complex **V, double complex *Vk,int Vfixedsize, int Vorder){
	//Performs the operation 2*sum(abs((U'*Uk).*(Vk*V')')) + norm(Uk)^2*norm(Vk)^2
	//U: U matrix
	//Uk: candidate column of length Ufixedsize
	//Ufixedsize: number of rows of U. Is fixed. Corresponds to M
	//Uorder: number of cols of U that we have added
	//V: Vmatrix
	//Vk: candidate row oflength Vfixedsize
	//Vfixedsize: number of cols of V. Is fixed. Corresponds to N
	//Vorder: number of rows that we have added to V
	
	double normUk = 0.0;
       	double normVk = 0.0;
	double aux    = 0.0;
	double part1  = 0.0;
	double complex sub_aux1 = 0.0 + 0.0 * I;
	double complex sub_aux2 = 0.0 + 0.0 * I;
	//First part. The 2*sum(abs((U'*Uk).*(Vk*V')')) part
	for (int ii=0; ii<Uorder; ii++){
		sub_aux1 = 0.0 + 0.0 * I; sub_aux2 = 0.0 + 0.0 * I;
		//Computing U element
		for (int jj=0; jj<Ufixedsize; jj++){
			sub_aux1 = sub_aux1 + conj(U[ii][jj])*Uk[jj];
		}
		for (int jj=0; jj<Vfixedsize; jj++){
			sub_aux2 = sub_aux2 + Vk[jj]*conj(V[ii][jj]);
		}
		aux = aux + cabs(sub_aux1*conj(sub_aux2)); //OJO: cabs
	}
	aux = 2*aux;
	part1 = aux;

	normUk = compute_norm(Uk,Ufixedsize);			
	normVk = compute_norm(Vk,Vfixedsize);			
	return part1 + pow(normUk,2) * pow(normVk,2); 
}

int pseudoACA(double complex **U, double complex **V, int * ACA_order, double complex *Z,int M, int N, double ACA_thres){
	
	printf("	Inside psaudoACA\n");	
	int *m, *n, *II, *J, *i, *j;
	int col, row, minMN, ncolsU, nrowsV, nelsi, nelsj;

	double normZ;

	printf("	Doing some malloc\n");
	double complex *Rik, *Rjk, *aux_lineN, *aux_lineM;
	Rik = (double complex *)malloc(N*sizeof(double complex));
	Rjk = (double complex *)malloc(M*sizeof(double complex));
	aux_lineN = (double complex *)malloc(N*sizeof(double complex));
	aux_lineM = (double complex *)malloc(M*sizeof(double complex));

	m = (int *)malloc(M * sizeof(int));
	n = (int *)malloc(N * sizeof(int));

	fillrange(m,M,0); fillrange(n,N,0);

	if (M==1 || N==1){
		//Do something short
	}

	printf("	Callocing a little bit	\n");
	J = (int *)calloc(N,sizeof(int)); //Indices of columns picked up from Z	
	II = (int *)calloc(M,sizeof(int)); //Indices of rows picked up from Z
	i = (int *)malloc((M-1)*sizeof(int)); //Row indices to search for maximum in R  
	j = (int *)malloc(N*sizeof(int)); //Column indices to search for maximum in R 
	nelsi = M-1; nelsj = N;
	fillrange(i,M-1,1); fillrange(j,N,0);
	//Initialize the 1st row index I(1) = 1
	II[0] = 0;
	
	//Initialize the 1st row of the approximate error matrix
	pseudo_ui_row(Rik,m[II[0]],n,1,N,Z,M,N);
	
	//% Find the 1st column index J[0]
	col = find_max_abs(Rik,1,N,j,nelsj);
	J[0] = j[col];
	remove_index(j,J[0],&nelsj);
	
	//First row of V	
	mult_and_row(V,0,N,Rik,1/Rik[J[0]]);
	nrowsV=1;
	
	pseudo_ui_col(Rjk,m,n[J[0]],M,1,Z,M,N);
	//First column of U
	add_a_line(U,0,M,Rjk);
	ncolsU = 1;

		
	normZ = pow(compute_norm(Rjk,M),2) * pow(compute_norm(V[0],N),2);
	
	row = find_max_abs(Rjk,1,M,i,nelsi);
	II[1] = i[row];
	remove_index(i,II[1],&nelsi);

	minMN = (M > N) ? N : M;


	printf("	Starting loop\n");
	for (int k = 1; k < minMN ; k++){
		//Compute the product of matrices
		//Update (Ik)th row of the approximate error matrix:
		UrowV(aux_lineN,U,II[k],M,ncolsU,V,N,nrowsV);	
		pseudo_ui_row(Rik,m[II[k]],n,1,N,Z,M,N);
		substract(Rik,Rik,aux_lineN,N);

		//Find kth column index Jk
		col = find_max_abs(Rik,1,N,j,nelsj);
		J[k] = j[col];
		remove_index(j,J[k],&nelsj);
		printf("	loop tag 1\n");
		//Terminate if R(I(k)==0
		if (Rik[J[k]]== 0.0 + 0.0 * I){
			break;
		}

		//Set k-th row of V equal to normalized error
		mult_line(N,Rik,1/Rik[J[k]]);

		printf("	loope tag 2\n");
		//Update (Jk)th column of the approximate error matrix
		UVcol(aux_lineM,U,M,ncolsU,V,J[k],N,nrowsV);
		printf("	loop tag 2.1\n");
		pseudo_ui_col(Rjk,m,n[J[k]],M,1,Z,M,N);
		printf("	loop tag2.2\n");
		substract(Rjk,Rjk,aux_lineM,M);
		
		//Norm of approximate Z
		printf("	loop tag2.3\n");
		normZ = normZ + adhoc_operation(U,Rjk,M,ncolsU,V,Rik,N,nrowsV); 
		
		//Update U and V
		printf("	loop tag 3\n");
		printf("	add line to V. nrowsV = %d. N = %d\n",nrowsV,N);
		add_a_line(V,nrowsV,N,Rik); nrowsV++;
		printf("	add line to U. ncolsU = %d. M = %d\n",ncolsU,M);
		add_a_line(U,ncolsU,M,Rjk); ncolsU++;

		//Check convergence
		if (compute_norm(Rik,N)*compute_norm(Rjk,M)<ACA_thres*sqrt(normZ)){
			break;
		}

		if (k==minMN-1){
			break;
		}
		//Find next row index
		printf("	loop tag 4\n");
		row = find_max_abs(Rjk,1,M,i,nelsi);
		II[k+1]=i[row]; //OJO
		remove_index(i,II[k+1],&nelsi);
		printf("	exiting loop \n");
	}
	

	*ACA_order = ncolsU;	
	//Los comentamos por si acaso
	free(Rik);
	free(Rjk);
	
	free(m);
	free(n);
	free(II);
	free(J);
	free(i);
	free(j);
	free(aux_lineN);
	free(aux_lineM);

	//int *m, *n, *II, *J, *i, *j;
	//int col, row, minMN, ncolsU, nrowsV, nelsi, nelsj;



	return 1; 

}

//Free Allocated memory
void freeAllocatedMemory(double complex **piBuffer, int nRow)
{
    int iRow = 0;
    for (iRow =0; iRow < nRow; iRow++)
    {
        free(piBuffer[iRow]); // free allocated memory
    }
    free(piBuffer);
}

int C_ACA_wrapper(double complex *U, double complex *V, double complex *A, double aca_threshold, int m, int n){
	
	double complex **Uaca = NULL; double complex **Vaca = NULL;

	//U and V are stored with an array of pointers to arrays. In this way, we can increase their size dynamically. U is stored as U[column][row] and V as V[row][column]. 

	//U tiene M filas, y como  máximo N columnas 
	printf("About to malloc in wrapper\n");
	Uaca = (double complex **)malloc(n*sizeof(double complex *));
	//V tiene N columnas, y como máximo M filas
	printf("Second malloc in wrapper\n");
	Vaca = (double complex **)malloc(m*sizeof(double complex *));
	
	//Check for allocation errors 
	if (Uaca==NULL || Vaca==NULL){
		fprintf(stderr,"Error in memory allocation \n");
	}

	int ACA_order, k;

	printf("Executing pseudoca\n");
	k = pseudoACA(Uaca,Vaca,&ACA_order,A,m,n,aca_threshold);
	printf("Pseudoaca executed\n"),

	//Now, copy to the proper format
	
	printf("About to copy U\n");
	for (int jj=0; jj<ACA_order; jj++){
		for (int ii=0; ii<m; ii++){
			U[ii+m*jj]=Uaca[jj][ii];
		}
	}

	printf("About co copy V\n");
	double complex auxforV;
	for (int jj=0; jj<n; jj++){
		for (int ii=0; ii<ACA_order; ii++){
			//V[ii+m*jj]=Vaca[ii][jj];
			printf("Comp auxforV\n");
			auxforV = Vaca[ii][jj];
			printf("inserting auxforV\n");
			V[ii+m*jj] = auxforV;
		}
	}
	printf("About to free memory");
	freeAllocatedMemory(Uaca,ACA_order);
	freeAllocatedMemory(Vaca,ACA_order);
	return ACA_order;
}

//Este funciona
//int synthetic_compress(int *matrix_m, int *matrix_n,int *vlength_m,int *vlength_n, int nels, int maxm, int maxn){
//	//printf("matrix_m[0][end] = %d\n",matrix_m[maxm*(nels-1)]);
//	//printf("matrix_m[1][end] = %d\n",matrix_m[maxm*(nels-1)+1]);
//	//printf("matrix_m[2][end] = %d\n",matrix_m[maxm*(nels-1)+2]);
//	double complex **Uaca = NULL; double complex **Vaca = NULL;
//	//Uaca = (double complex **)malloc(1000*sizeof(double complex *));
//	//Vaca = (double complex **)malloc(1000*sizeof(double complex *));
//
//	int k;
//	for (int ii=0; ii<nels; ii++){
//		Uaca = (double complex **)malloc(vlength_n[ii]*sizeof(double complex *));	;
//		Vaca = (double complex **)malloc(vlength_m[ii]*sizeof(double complex *));
//		for (int jj = 0; jj<vlength_n[ii]; jj++){
//			Uaca[jj] = (double complex *)malloc(vlength_m[ii]*sizeof(double complex*));
//		}
//		for (int jj = 0; jj<vlength_m[ii]; jj++){
//			Vaca[jj] = (double complex *)malloc(vlength_n[ii]*sizeof(double complex*));
//		}
//
//		//free(Uaca);
//		//free(Vaca);
//		freeAllocatedMemory(Uaca,vlength_n[ii]);
//		freeAllocatedMemory(Vaca,vlength_m[ii]);
//	}
//	printf("With frees. ended loop of allocations and deallocations\n");
//	return 1;
//}


int synthetic_compress(double complex *Z, int mZ, int nZ, int *matrix_m, int *matrix_n,int *vlength_m,int *vlength_n, int nels, int maxm, int maxn){
	//printf("matrix_m[0][end] = %d\n",matrix_m[maxm*(nels-1)]);
	//printf("matrix_m[1][end] = %d\n",matrix_m[maxm*(nels-1)+1]);
	//printf("matrix_m[2][end] = %d\n",matrix_m[maxm*(nels-1)+2]);
	double complex **Uaca = NULL; double complex **Vaca = NULL; 
	double complex *Zlocal;

	int ACA_order;

	//Uaca = (double complex **)malloc(1000*sizeof(double complex *));
	//Vaca = (double complex **)malloc(1000*sizeof(double complex *));

	int k;
	for (int ii=0; ii<nels; ii++){
		//Creating Zlocal
		Zlocal = (double complex *)malloc(vlength_n[ii]*vlength_m[ii]*sizeof(double complex));
		for (int jj=1; jj<vlength_n[ii]; jj++){
			for (int kk=1;kk<vlength_m[ii]; kk++){
				//Zlocal[kk+jj*vlength_m[ii]] = Z[matrix_m[kk+vlength_m[ii]*jj]]; //OJO 22
				Zlocal[kk+jj*vlength_m[ii]] = Z[matrix_m[kk+maxm*ii]+mZ*matrix_n[jj+maxn*ii]];
			}

		}	

		Uaca = (double complex **)malloc(vlength_n[ii]*sizeof(double complex *));	;
		Vaca = (double complex **)malloc(vlength_m[ii]*sizeof(double complex *));
	
		k = pseudoACA(Uaca,Vaca,&ACA_order,Zlocal,vlength_m[ii],vlength_n[ii],0.01);
	
		
	//	for (int jj = 0; jj<vlength_n[ii]; jj++){
	//		Uaca[jj] = (double complex *)malloc(vlength_m[ii]*sizeof(double complex*));
	//	}
	//	for (int jj = 0; jj<vlength_m[ii]; jj++){
	//		Vaca[jj] = (double complex *)malloc(vlength_n[ii]*sizeof(double complex*));
	//	}

		

		//free(Uaca);
		//free(Vaca);
		
		//freeAllocatedMemory(Uaca,vlength_n[ii]);
		//freeAllocatedMemory(Vaca,vlength_m[ii]);
			
		freeAllocatedMemory(Uaca,ACA_order);
		freeAllocatedMemory(Vaca,ACA_order);
	
		free(Zlocal);
	}
	printf("With ACA.With frees. ended loop of allocations and deallocations\n");
	return 1;
}

void write_integer_matrix_to_files(int *A, int M, int N, int code){
	FILE *f;
	if (code==1){
		f = fopen("matrix_mFILE.txt","w");
	} else if (code==2){
		f = fopen("matrix_nFILE.txt","w");
	} else if (code==3){
		f = fopen("vlength_mFILE.txt","w");
	} else if (code==4){
		f = fopen("vlength_nFILE.txt","w");
	} else if (code==5) {
		f = fopen("vcompressed_FILE.txt","w");
	} else {
		printf("Error in write_integer_matrix_to_files\n");
		return;
	}

	for (int jj=0; jj<N; jj++){
		for (int ii=0; ii<M; ii++){
			fprintf(f,"%d,",A[ii+jj*M]);
		}
	}
	fclose(f);
	return;
}


void read_integer_matrix_from_files(int *A, int M, int N, int code){
	FILE *f;
	if (code==1){
		f = fopen("matrix_mFILE.txt","r");
	} else if (code==2){
		f = fopen("matrix_nFILE.txt","r");
	} else if (code==3){
		f = fopen("vlength_mFILE.txt","r");
	} else if (code==4){
		f = fopen("vlength_nFILE.txt","r");
	} else {
		printf("Error in write_integer_matrix_to_files\n");
		return;
	}

	int aux;

	for (int jj=0; jj<N; jj++){
		for (int ii=0; ii<M; ii++){
			fscanf(f,"%d,",&aux);
			A[ii+jj*M]=aux;
		}
	}
	fclose(f);
	return;
}


void write_complex_matrix_to_files(complex double *Z, int M, int N){
        FILE *fr, *fi;
        fr = fopen("Z_real_part.txt","w");
        fi = fopen("Z_imag_part.txt","w");

        //Write real
        for (int jj=0; jj<N; jj++){
                for (int ii=0;ii<M;ii++){
                        //Z[ii+jj*M] = rand() + I*rand();
                        fprintf(fr,"%lf,",creal(Z[ii+jj*M]));
                }
        }

        //Write imag
        for (int jj=0; jj<N; jj++){
                for (int ii=0;ii<M;ii++){
                        //Z[ii+jj*M] = rand() + I*rand();
                        fprintf(fi,"%lf,",cimag(Z[ii+jj*M]));
                }
        }
        fclose(fr);
        fclose(fi);
        return;
}

void read_complex_matrix_from_files(complex double *Z, int M, int N){
        FILE *fr, *fi;
        fr = fopen("Z_real_part.txt","r");
        fi = fopen("Z_imag_part.txt","r");

        double aux;

        //Write real
        for (int jj=0; jj<N; jj++){
                for (int ii=0;ii<M;ii++){
                        fscanf(fr,"%lf,",&aux);//rand() + I*rand();
                        printf("Executing write. aux = %lf\n",&aux);
                        Z[ii+jj*M] = aux;
                        //fprintf(fr,"%f,",creal(Z[ii+jj*M]));
                }
        }

        //Write imag
        for (int jj=0; jj<N; jj++){
                for (int ii=0;ii<M;ii++){
                        //Z[ii+jj*M] = rand() + I*rand();
                        //fprintf(fi,"%f,",cimag(Z[ii+jj*M]));
                        fscanf(fi,"%lf,",&aux);//rand() + I*rand();
                        Z[ii+jj*M] += I*aux;

                }
        }
        return;

}

double compare_complex_matrices(double complex *Z1, double complex *Z2, int M, int N){
        double cum_error = 0.0;
        for (int jj=0; jj<N; jj++){
                for (int ii=0; ii<M; ii++){
                        printf("Comparing elements [%d][%d]. Z1-> %lf + i%lf. Z2-> %lf + i%lf\n",ii,jj,creal(Z1[ii+M*jj]),cimag(Z1[ii+M*jj]),creal(Z2[ii+M*jj]),cimag(Z2[ii+M*jj]));
                        cum_error=cum_error+cabs(Z1[ii+M*jj]-Z2[ii+M*jj]);
                }
        }
        return cum_error;
}

double compare_integer_matrices(int *Z1, int *Z2, int M, int N){
        double cum_error = 0.0;
        for (int jj=0; jj<N; jj++){
                for (int ii=0; ii<M; ii++){
                        //printf("Comparing elements [%d][%d]. Z1-> %lf + i%lf. Z2-> %lf + i%lf\n",ii,jj,creal(Z1[ii+M*jj]),cimag(Z1[ii+M*jj]),creal(Z2[ii+M*jj]),cimag(Z2[ii+M*jj]));
                        //cum_error=cum_error+cabs(Z1[ii+M*jj]-Z2[ii+M*jj]);
			cum_error = cum_error + abs((double)(Z1[ii+M*jj]-Z2[ii+M*jj]));
                }
        }
        return cum_error;
}



int copy_to_a_file(double complex *Z, int M, int N, int *matrix_m, int *matrix_n,int *vlength_m,int *vlength_n, int nels, int maxm, int maxn){
	//Copying Z
	write_complex_matrix_to_files(Z,M,N);
	write_complex_matrix_to_files(Z,M,N);

	//Copying matrix_m
	write_integer_matrix_to_files(matrix_m,maxm,nels,1);
	
	//Copying matrix_n
	write_integer_matrix_to_files(matrix_n,maxn,nels,2);

	//Copying vlength_m
	write_integer_matrix_to_files(vlength_m,nels,1,3);

	//Copying vlength_n
	write_integer_matrix_to_files(vlength_n,nels,1,4);	
	return 1;
}


int copy_to_a_file2(double complex *Z, int M, int N, int *matrix_m, int *matrix_n,int *vlength_m,int *vlength_n, int * v_compressed,int nels, int maxm, int maxn){
	//La diferencia entre copy_to_a_file y copy_to_a_file2 es que este 2 copia también el vector binario de compresión
	
	//Copying Z
	write_complex_matrix_to_files(Z,M,N);
	write_complex_matrix_to_files(Z,M,N);

	//Copying matrix_m
	write_integer_matrix_to_files(matrix_m,maxm,nels,1);
	
	//Copying matrix_n
	write_integer_matrix_to_files(matrix_n,maxn,nels,2);

	//Copying vlength_m
	write_integer_matrix_to_files(vlength_m,nels,1,3);

	//Copying vlength_n
	write_integer_matrix_to_files(vlength_n,nels,1,4);

	//Copying v_compressed
	write_integer_matrix_to_files(v_compressed,nels,1,5);	
	return 1;
}


int compare(double complex *Z, int M, int N, int *matrix_m, int *matrix_n,int *vlength_m,int *vlength_n, int nels, int maxm, int maxn){
	double complex *Zcopy;
	int *matrix_mcopy, *matrix_ncopy, *vlength_mcopy, *vlength_ncopy;
	Zcopy = (double complex *)malloc(M*N*sizeof(double complex));
	matrix_mcopy = (int *)malloc(maxm*nels*sizeof(int));
	matrix_ncopy = (int *)malloc(maxn*nels*sizeof(int));
	vlength_mcopy = (int *)malloc(nels*sizeof(int));
	vlength_ncopy = (int *)malloc(nels*sizeof(int));
	
	//Reading z
	read_complex_matrix_from_files(Zcopy,M,N);
          
	//Reading matrix_m
	read_integer_matrix_from_files(matrix_mcopy,maxm,nels,1);
	  
	//Reading matrix_n
	read_integer_matrix_from_files(matrix_ncopy,maxn,nels,2);
          
	//Reading vlength_m
	read_integer_matrix_from_files(vlength_mcopy,nels,1,3);

	//Reading vlength_n
	read_integer_matrix_from_files(vlength_ncopy,nels,1,4);	

	//Computing errors
	double errZ, errmm, errmn, errvm, errvn;

	errZ = compare_complex_matrices(Z,Zcopy,M,N);
	errmm = compare_integer_matrices(matrix_m,matrix_mcopy,maxm,nels);
	errmn = compare_integer_matrices(matrix_n,matrix_ncopy,maxn,nels);
	errvm = compare_integer_matrices(matrix_m,matrix_mcopy,nels,1);
	errvn = compare_integer_matrices(matrix_m,matrix_mcopy,nels,1);

	//Nota, Z no sale exactamente igual. El error es del orden 1e-7. Esto se debe al formato al copiar (se pierde coma flotante)
	printf("errZ = %f\n",errZ);
	printf("errmm = %f\n",errmm);
	printf("errmn = %f\n",errmn);
	printf("errvm = %f\n",errvm);
	printf("errvn = %f\n",errvn);
	
	return 1;
}











