

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cusolverDn.h>
#include <assert.h>
#include <string>
#include <cuda_runtime.h>


#include <complex.h> //OJO

extern "C" int multiplicar_por_dos(cuDoubleComplex *A, int M, int N){
	//A: matrix
	//M: number of rows
	//N: number of cols
	for (int ii=0; ii<N; ii++){
		for (int jj=0; jj<M; jj++){
			A[ii*M+jj] = cuCmul(make_cuDoubleComplex(2.0,0.0),A[ii*M+jj]);
		}
	}
	return 1;
}


extern "C" int cuda_copy(cuDoubleComplex *A, cuDoubleComplex *B, int M, int N){
	//Para probar a copiar desde la GPU
	
	cuDoubleComplex *d_A = NULL;
	cudaMalloc(&d_A,sizeof(cuDoubleComplex)*M*N);
	cudaMemcpy(d_A,A,sizeof(cuDoubleComplex)*M*N,cudaMemcpyHostToDevice);
	cudaMemcpy(B,d_A,sizeof(cuDoubleComplex)*M*N,cudaMemcpyDeviceToHost);
	
	return 1;	

}


double compute_norm(cuDoubleComplex *W, int lengthW){
	        double aux = 0.0;
		        for (int ii=0; ii<lengthW; ii++){
				                //aux = aux + pow(cabs(W[ii]),2);
				                aux = aux + pow(cuCabs(W[ii]),2);
						        }
			        return sqrt(aux);
}


void substract(cuDoubleComplex *vec, cuDoubleComplex *a, cuDoubleComplex *b, int N){
	        //Performs the operation v = a - b,
	        //vec: where the result is stored
	        //a: first vector
	        //b: second vector
	        //N: length of a and b
	        for (int ii=0; ii<N; ii++){
			                //vec[ii] = a[ii] - b[ii];
			                vec[ii] = cuCsub(a[ii],b[ii]);
					        }
		        return;
}



__global__ void kernel_substract(cuDoubleComplex *vec, cuDoubleComplex *a, cuDoubleComplex *b, int N){
        int ii = blockIdx.x*blockDim.x + threadIdx.x;
        vec[ii]=cuCsub(a[ii],b[ii]); //a[ii]-b[ii];
        return;
}

void d_substract(cuDoubleComplex *vec, cuDoubleComplex *a, cuDoubleComplex *b, int N){
        //Performs the operation v = a - b,
        //vec: where the result is stored
        //a: first vector
        //b: second vector
        //N: length of a and b

        kernel_substract<<<N,1>>>(vec,a,b,N);

	return;
}

//cuda_svd2 (basado en cuda_pruebas,que a pesar de su nombre es la versión correcta de SVD. En esta segunda función, vamos a procurar que nos devuelva las matrices en formato reducido)
extern "C" int cuda_svd2(cuDoubleComplex *U, double *S, cuDoubleComplex *Vt, cuDoubleComplex *A, int M, int N){
	//Este tal y como está FUNCIONA
	//Para probar a copiar desde la GPU

	printf("Starting cuda_svd2\n");	

	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
    	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;


	cuDoubleComplex *d_A = NULL;
	double *d_S = NULL;
	cuDoubleComplex *d_U = NULL;
	cuDoubleComplex *d_Vt = NULL;
	int *devInfo = NULL;
	cuDoubleComplex *d_work = NULL;
	double *d_rwork = NULL;

	int lwork = 0;
	int lda = M;

	//int *d_work = NULL;
	
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    	cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	int minMN = min(M,N);

	
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex)*M*N);
	cudaStat2 = cudaMalloc((void**)&d_S, sizeof(double)*minMN);
	//cudaStat3 = cudaMalloc((void**)&d_U, sizeof(cuDoubleComplex)*M*M);
	cudaStat3 = cudaMalloc((void**)&d_U, sizeof(cuDoubleComplex)*M*N);
	//cudaStat4 = cudaMalloc((void**)&d_Vt, sizeof(cuDoubleComplex)*N*N);
	cudaStat4 = cudaMalloc((void**)&d_Vt, sizeof(cuDoubleComplex)*N*N);

	cudaStat5 = cudaMalloc((void**)&devInfo,sizeof(int));


	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuDoubleComplex)*M*N, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

	cusolver_status = cusolverDnZgesvd_bufferSize(
			cusolverH,
			M,
			N,
			&lwork);
	
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	
	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);
	assert(cudaSuccess == cudaStat1);

	
	//Step 4: compute SVD
	signed char jobu = 'S'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	
	cusolver_status = cusolverDnZgesvd(
			cusolverH,
			jobu,
			jobvt,
			M,
			N,
			d_A,
			lda,
			d_S,
			d_U,
			lda, //ldu
			d_Vt,
			N, //ldvt (antes lda)
			d_work,
			lwork,
			d_rwork,
			devInfo);
		    
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

 
	cudaStat1 = cudaMemcpy(U, d_U, sizeof(cuDoubleComplex)*M*N,cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(Vt, d_Vt, sizeof(cuDoubleComplex)*N*N,cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(S, d_S, sizeof(double)*minMN,cudaMemcpyDeviceToHost);
   

	//free resources
	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_Vt);
	cudaFree(devInfo); //Posible error
	cudaFree(d_work);
	cudaFree(d_rwork);
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);

	return 1;	

}



__global__ void kernel_Sinv(double *d_vec, cuDoubleComplex *d_vec_complex, int N, double treshold){
	//Performs the filtering operation and transforms to cuDoubleComplex typoe
	int ii = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_vec[ii]<treshold){
		d_vec_complex[ii] = make_cuDoubleComplex(0.0,0.0);
	} else {
		d_vec_complex[ii] = make_cuDoubleComplex(1/d_vec[ii],0.0);
	}
	return;
}


extern "C" int cuda_pinv(cuDoubleComplex *pinvA, cuDoubleComplex *U, double *S, cuDoubleComplex *Vt, cuDoubleComplex *A, int M, int N, double treshold){
	//Este tal y como está FUNCIONA
	//Para probar a copiar desde la GPU
	//Based on cuda_svd2. Intended to compute pinv in cuda

	//A is MxN
	//pinvA is NxM

	printf("Starting cuda_pinv\n");	

	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;


	cuDoubleComplex *d_A = NULL;
	double *d_S = NULL;
	cuDoubleComplex *d_U = NULL;
	cuDoubleComplex *d_Vt = NULL;
	cuDoubleComplex *d_aux_V = NULL;  //Aux for V operations
	cuDoubleComplex *d_pinvA = NULL;
	cuDoubleComplex *d_Sinv = NULL;
	int *devInfo = NULL;
	cuDoubleComplex *d_work = NULL;
	double *d_rwork = NULL;

	int lwork = 0;
	const cuDoubleComplex h_alpha = make_cuDoubleComplex(1.0,0.0);
	const cuDoubleComplex h_beta  = make_cuDoubleComplex(0.0,0.0);

	int lda = M;

	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    	cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	int minMN = min(M,N);
	
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex)*M*N);
	cudaStat2 = cudaMalloc((void**)&d_S, sizeof(double)*minMN);
	cudaStat3 = cudaMalloc((void**)&d_U, sizeof(cuDoubleComplex)*M*N);
	cudaStat4 = cudaMalloc((void**)&d_Vt, sizeof(cuDoubleComplex)*N*N);
	cudaStat6 = cudaMalloc((void**)&d_aux_V, sizeof(cuDoubleComplex)*N*N); //disorder in cudaStat 6 and 5
	cudaStat6 = cudaMalloc((void**)&d_pinvA, sizeof(cuDoubleComplex)*N*M); //reusing cudaStat6
	cudaStat6 = cudaMalloc((void**)&d_Sinv, sizeof(cuDoubleComplex)*N);
	cudaStat5 = cudaMalloc((void**)&devInfo,sizeof(int));


	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuDoubleComplex)*M*N, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

	cusolver_status = cusolverDnZgesvd_bufferSize(
			cusolverH,
			M,
			N,
			&lwork);
	
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	
	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);
	assert(cudaSuccess == cudaStat1);

	
	//Step 4: compute SVD
	signed char jobu = 'S'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	
	cusolver_status = cusolverDnZgesvd(
			cusolverH,
			jobu,
			jobvt,
			M,
			N,
			d_A,
			lda,
			d_S,
			d_U,
			lda, //ldu
			d_Vt,
			N, //ldvt (antes lda)
			d_work,
			lwork,
			d_rwork,
			devInfo);
		    
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	kernel_Sinv<<<N,1>>>(d_S, d_Sinv, N, treshold);

	//Compute d_aux_V = d_Sinv*Vt
	cublas_status = cublasZdgmm(
			cublasH, CUBLAS_SIDE_LEFT,
			N,N,
			d_Vt, N,
			d_Sinv,1,
			d_aux_V,N);

	//Compute (d_Sinv*Vt)'*U'=V*d_Sinv*U'
	cublas_status = cublasZgemm(
			cublasH,
			CUBLAS_OP_C, CUBLAS_OP_C,
			N, M, N,
			&h_alpha,
			d_aux_V, N,
			d_U, M,
			&h_beta,
			d_pinvA,N);

	cudaStat1 = cudaMemcpy(U, d_U, sizeof(cuDoubleComplex)*M*N,cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(Vt, d_aux_V, sizeof(cuDoubleComplex)*N*N,cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(S, d_S, sizeof(double)*minMN,cudaMemcpyDeviceToHost); //>For debug. Originally d_SS
	cudaStat4 = cudaMemcpy(pinvA, d_pinvA, sizeof(cuDoubleComplex)*N*M, cudaMemcpyDeviceToHost);
   

	//free resources
	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_Vt);
	cudaFree(d_aux_V);
	cudaFree(d_pinvA);
	cudaFree(d_Sinv);
	cudaFree(devInfo); //Posible error
	cudaFree(d_work);
	cudaFree(d_rwork);
	
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);


	printf("CAFE CON LOU\n");

	return 1;	

}




__global__ void kernel_filter(double *d_V, int N, double treshold){
	int ii = blockIdx.x*blockDim.x + threadIdx.x;
	if (d_V[ii]<treshold){
		d_V[ii]=0.0;
	} else {
		d_V[ii]=1.0/d_V[ii];
	}
	return;
}

extern "C" int cuda_filter(double *V, double treshold, int N){
	//V: vector to perform filtering
	//treshold: cut off for filtering
	//N: length of the vector

	double *d_V = NULL;

	cudaMalloc(&d_V,N*sizeof(double));
	cudaMemcpy(d_V,V,N*sizeof(double),cudaMemcpyHostToDevice);
	
	kernel_filter<<<N,1>>>(d_V,N,treshold);

	cudaMemcpy(V,d_V,N*sizeof(double),cudaMemcpyDeviceToHost);

	return 1;

}

extern "C" int cuda_svd(cuDoubleComplex *U, double *S, cuDoubleComplex *Vt, cuDoubleComplex *A, int M, int N){
	//Se han hecho cambios erroneos. No confiar en este código

	//U: U matrix of the SVD
	//S: singular values of the SVD
	//V: V matrix of the SVD
	//A: matrix to be decomposed
	//M: number of rows of A
	//N: number of columns of A
	
	cusolverDnHandle_t cusolverH = NULL;
    	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cudaError_t cudaStat5 = cudaSuccess;
	cudaError_t cudaStat6 = cudaSuccess;

	cuDoubleComplex *d_A = NULL;
	double *d_S = NULL;
	cuDoubleComplex *d_U = NULL;
	cuDoubleComplex *d_Vt = NULL;
	int *devInfo = NULL;
	cuDoubleComplex *d_work = NULL;
	double *d_rwork = NULL;

	int lwork = 0;
	int lda = M;

	//int *d_work = NULL;
	
	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    	//cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	int minMN = min(M,N);

	printf("M = %d\n",M);
	printf("N = %d\n",N);

      cudaStat1 = cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex)*M*N);
      cudaStat2 = cudaMalloc((void**)&d_S, sizeof(double)*minMN);
      cudaStat3 = cudaMalloc((void**)&d_U, sizeof(cuDoubleComplex)*M*M);
      cudaStat4 = cudaMalloc((void**)&d_Vt, sizeof(cuDoubleComplex)*N*N);
      cudaStat5 = cudaMalloc((void**)&devInfo,sizeof(int));

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(cuDoubleComplex)*M*N, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);

	//step 3 query working space
	cusolver_status = cusolverDnZgesvd_bufferSize(
			cusolverH,
			M,
			N,
			&lwork);
	
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	
	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);
	assert(cudaSuccess == cudaStat1);

	
	//Step 4: compute SVD
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	
	cusolver_status = cusolverDnZgesvd(
			cusolverH,
			jobu,
			jobvt,
			M,
			N,
			d_A,
			lda,
			d_S,
			d_U,
			lda, //ldu
			d_Vt,
			N, //ldvt (antes lda)
			d_work,
			lwork,
			d_rwork,
			devInfo);
	    
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMemcpy(U, d_U, sizeof(cuDoubleComplex)*M*M,cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(Vt, d_Vt, sizeof(cuDoubleComplex)*N*N,cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(S, d_S, sizeof(double)*minMN,cudaMemcpyDeviceToHost);


	//free resources
	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_Vt);
	cudaFree(devInfo); //Posible error
	cudaFree(d_work);
	cudaFree(d_rwork);
	
	//cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);

	return 1;


}

int get_proper_cols(cuDoubleComplex *Asubset, cuDoubleComplex *A, int *col_vec, int M, int N, int length_col_vec){
        //Output
        //Asubset: selected cols of A

        //Input
        //A matrix to extract cols
        //col_vec: vector containing the rows to be extracted
        //M: number of rows of A
        //N: number of cols of A
        //length_col_vec: number of elements of col_vec

        for (int ii=0; ii<length_col_vec; ii++){
                for (int jj=0; jj<M; jj++){
                        Asubset[ii*M+jj] = A[(col_vec[ii]-1)*M+jj];
                }
        }
        return 1;
}


int receive_integer_vector(int *v, int N){
        //Let's assume v has length 4
        for (int ii=0;ii<4;ii++){
                printf("v[%d] = %d\n",ii,v[ii]);
        }
        return 1;
}


int get_proper_rows(cuDoubleComplex *Asubset, cuDoubleComplex *A, int *row_vec, int M, int N, int length_row_vec){
        //Output
        //Asubset: selected rows of A

        //Input
        //A matrix to rows
        //row vec: vector containing the rows to be extracted
        //M: number of rows of A
        //N: number of cols of A
        //length_row_vec: number of elements of row_vec

        for (int ii=0; ii<N; ii++){
                for (int jj=0; jj<length_row_vec; jj++){
                        Asubset[ii*length_row_vec+jj] = A[ii*M+(row_vec[jj]-1)];
                }
        }

        return 1;
}





extern "C" int aux_pinv_compression(cuDoubleComplex *C, cuDoubleComplex *U, cuDoubleComplex *R, cuDoubleComplex *A, int *row_samples, int *col_samples, int ns, int M, int N, double tolerance){
	
	//This function returns the CUR compression of A
	//C: Mxns matrix
	//U: nsxns matrix
	//R: nsxN matrix
	//A: MxN matrix to be comressed
	//row_samples: vector of length ns containing the indices of the rows to take (Matlab notation ie start at 1)
	//col_samples: idem for columns
	//ns: number of samples
	//M: number of rows of A
	//N: number of cols of A
	//tolerance: tolerance for performing pseudoinverse

	int aux = 0;
	aux = get_proper_cols(C,A,col_samples,M,N,ns);
	aux = get_proper_rows(R,A,row_samples,M,N,ns);

	

	cuDoubleComplex *Uintersection = NULL; //The interesection of C and R
	cuDoubleComplex *Usvd = NULL; //The U of the SVD
	double *Ssvd = NULL; //The S of the SVD
	cuDoubleComplex *Vtsvd = NULL; //The Vt of the SVD

	Uintersection = (cuDoubleComplex *)malloc(ns*ns*sizeof(cuDoubleComplex));
	Usvd  = (cuDoubleComplex *)malloc(ns*ns*sizeof(cuDoubleComplex));
	Ssvd  = (double *)malloc(ns*sizeof(cuDoubleComplex));
	Vtsvd =	(cuDoubleComplex *)malloc(ns*ns*sizeof(cuDoubleComplex));

	for (int ii=0; ii<ns; ii++){
		for (int jj=0; jj<ns; jj++){
			Uintersection[ii*ns+jj] = A[(col_samples[ii]-1)*M+(row_samples[jj]-1)];
		}
	}	

	cuda_pinv(U,Usvd,Ssvd,Vtsvd,Uintersection,ns,ns,tolerance);	
	return 1;
}

extern "C" double compute_error_compression(cuDoubleComplex *C, cuDoubleComplex *U, cuDoubleComplex *R, cuDoubleComplex *Avtest_old, cuDoubleComplex *Avtest_new, cuDoubleComplex *vtest, int M1, int N1, int M2){  
	
	//C is M1xN1
	//U is N1xN1
	//R is N1xM2
	//vtest is M2x1
       //Avtest_old and Avtest_new are M1x1

	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;


	cusolver_status = cusolverDnCreate(&cusolverH);
	cublas_status = cublasCreate(&cublasH);
 
 	const cuDoubleComplex h_alpha = make_cuDoubleComplex(1.0,0.0);
        const cuDoubleComplex h_beta  = make_cuDoubleComplex(0.0,0.0);

	double err_here = 0.0;
	
	double norm_Avtest_new = 0.0;
	double norm_Avtest_diff = 0.0;

	cuDoubleComplex *d_C = NULL;
	cuDoubleComplex *d_U = NULL;
	cuDoubleComplex *d_R = NULL;
	cuDoubleComplex *d_Avtest_old = NULL;
	cuDoubleComplex *d_Avtest_new = NULL;
	cuDoubleComplex *d_vtest = NULL;
	cuDoubleComplex *d_aux1 = NULL; //For storing the res of R*v
	cuDoubleComplex *d_aux2 = NULL; //For storing the res of U*(R*v)
	cuDoubleComplex *d_aux3 = NULL; //For storing the res of Avtest_old-Avtest_new

	cudaMalloc((void**)&d_C, sizeof(cuDoubleComplex)*M1*N1);
	cudaMalloc((void**)&d_U, sizeof(cuDoubleComplex)*N1*N1);
	cudaMalloc((void**)&d_R, sizeof(cuDoubleComplex)*N1*M2);
	cudaMalloc((void**)&d_Avtest_old, sizeof(cuDoubleComplex)*M1);
	cudaMalloc((void**)&d_Avtest_new, sizeof(cuDoubleComplex)*M1);
	cudaMalloc((void**)&d_vtest, sizeof(cuDoubleComplex)*M2);
	cudaMalloc((void**)&d_aux1, sizeof(cuDoubleComplex)*N1);
	cudaMalloc((void**)&d_aux2, sizeof(cuDoubleComplex)*N1);
	cudaMalloc((void**)&d_aux3, sizeof(cuDoubleComplex)*M1);


	cudaMemcpy(d_C,C, sizeof(cuDoubleComplex)*M1*N1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_U,U, sizeof(cuDoubleComplex)*N1*N1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_R,R, sizeof(cuDoubleComplex)*N1*M2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_Avtest_old,Avtest_old, sizeof(cuDoubleComplex)*M1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_Avtest_new,Avtest_new, sizeof(cuDoubleComplex)*M1,cudaMemcpyHostToDevice);
	cudaMemcpy(d_vtest,vtest, sizeof(cuDoubleComplex)*M2,cudaMemcpyHostToDevice);

	//R*vtest
	cublas_status = cublasZgemm(
			cublasH,
			CUBLAS_OP_N, CUBLAS_OP_N,
			N1, 1, M2,
			&h_alpha,
			d_R, N1,
			d_vtest, M2,
			&h_beta,
			d_aux1,N1);
	//U*d_aux1 = U*(R*vtest)
	cublas_status = cublasZgemm(
			cublasH,
			CUBLAS_OP_N, CUBLAS_OP_N,
			N1,1 , N1,
			&h_alpha,
			d_U, N1,
			d_aux1, N1,
			&h_beta,
			d_aux2,N1);
	//C*d_aux12 = C*(U*(R*vtest))
	cublas_status = cublasZgemm(
			cublasH,
			CUBLAS_OP_N, CUBLAS_OP_N,
			M1, 1 , N1,
			&h_alpha,
			d_C, M1,
			d_aux2, N1,
			&h_beta,
			d_Avtest_new,M1);
	cudaMemcpy(Avtest_new,d_Avtest_new, sizeof(cuDoubleComplex)*M1,cudaMemcpyDeviceToHost);
	
	cublas_status = cublasDznrm2(cublasH,M1,d_Avtest_new,1,&norm_Avtest_new);
	d_substract(d_aux3,d_Avtest_old,d_Avtest_new,M1);	
	cublas_status = cublasDznrm2(cublasH,M1,d_aux3,1,&norm_Avtest_diff);

	double norm_diff_dummy = 0.0;
	double norm_new_dummy  = 0.0;
	cuDoubleComplex *diff_dummy;
	diff_dummy = (cuDoubleComplex *)malloc(M1*sizeof(cuDoubleComplex));
	substract(diff_dummy,Avtest_old,Avtest_new,M1);
	norm_diff_dummy = compute_norm(diff_dummy,M1);
	norm_new_dummy  = compute_norm(Avtest_new,M1);

	err_here = norm_Avtest_diff/norm_Avtest_new;

	cudaFree(d_C);
	cudaFree(d_U);
	cudaFree(d_R);
	cudaFree(d_Avtest_old);
	cudaFree(d_Avtest_new);
	cudaFree(d_vtest);
	cudaFree(d_aux1);
	cudaFree(d_aux2);
	cudaFree(d_aux3);
	
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);

	return err_here;
}

