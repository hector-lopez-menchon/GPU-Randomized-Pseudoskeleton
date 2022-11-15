
#include <math.h>
#include <stdlib.h>

#include <stdio.h> //Incluido para cuda
#include <cuda_profiler_api.h> //Incluido para cuda profiling
#include <cuComplex.h>
#include <cusolverDn.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define _PI     3.14159265358979

/* Arguments */
#define IN_ARG   4

#define OUT_ARG  1

#define norm(V)            sqrt(V[0]*V[0]+V[1]*V[1]+V[2]*V[2]) 
#define rrdot(V1,V2)       (V1[0]*V2[0] + V1[1]*V2[1] + V1[2]*V2[2])
#define rcdot_r(V1,V2)     (V1[0]*V2[0].r + V1[1]*V2[1].r + V1[2]*V2[2].r)
#define rcdot_i(V1,V2)     (V1[0]*V2[0].i + V1[1]*V2[1].i + V1[2]*V2[2].i)
#define cross_rc(C,A,B)    { C[0].r = A[1]*B[2].r - A[2]*B[1].r; C[0].i = A[1]*B[2].i - A[2]*B[1].i;\
                             C[1].r = A[2]*B[0].r - A[0]*B[2].r; C[1].i = A[2]*B[0].i - A[0]*B[2].i;\
                             C[2].r = A[0]*B[1].r - A[1]*B[0].r; C[2].i = A[0]*B[1].i - A[1]*B[0].i;\
                           }
#define cross_rr(C,A,B)    { C[0] = A[1]*B[2] - A[2]*B[1];\
                             C[1] = A[2]*B[0] - A[0]*B[2];\
                             C[2] = A[0]*B[1] - A[1]*B[0];\
                           }


#define Gp 3

typedef double triad[3]; /* Vector or triangle information */
typedef double cuad[4];  /* Edge information */
typedef struct { double r,i; } dcomplex;

//void autoz(dcomplex *pI, dcomplex Ii[3][3], int Ts, triad *topol, triad *vertex, triad *trian, double rt[3], triad *un, double k);
//void anal_1divR(double *I_sc, double I_vc[3], int Ts, triad *topol, triad *vertex, triad *trian, double rt[3], triad *un, double *ds, double k, double signe );

__device__ void autoz(dcomplex *pI, dcomplex Ii[3][3], int Ts, triad *topol, triad *vertex, triad *trian, double rt[3], triad *un, double k);
__device__ void anal_1divR(double *I_sc, double I_vc[3], int Ts, triad *topol, triad *vertex, triad *trian, double rt[3], triad *un, double *ds, double k, double signe );

int check_password(void);
void decode_password(unsigned int key, unsigned int p1, unsigned int p2, unsigned int *psys, unsigned int *psn, unsigned int *pdia, unsigned int *pmes, unsigned int *pano);
int compara_date(unsigned int dia, unsigned int mes, unsigned int ano, unsigned int dia2, unsigned int mes2, unsigned int ano2);

double pruebilla(double a);

//Function declaration
__device__ int impedance_matrix(double *Zr,
		      double *Zi,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      double *r1_arg,	      //Parameters originally in obj
		      double *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg);


__device__ int impedance_matrix_modified(double *Zr,
		      double *Zi,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      double *r1_arg,	      //Parameters originally in obj
		      double *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg,
		      int *Tlist_f_arg,
		      char *Tcomp_f_arg,
		      int *Elist_f_arg,
		      int nTf_arg,
		      int *Tlist_s_arg,
		      char *Tcomp_s_arg,
		      int *Elist_s_arg);

__device__ int impedance_matrix_single_element(double *Zr,
		      double *Zi,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      double *r1_arg,	      //Parameters originally in obj
		      double *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg);

__device__ int impedance_matrix_single_element2(cuDoubleComplex *Ztotal,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      double *r1_arg,	      //Parameters originally in obj
		      double *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg);




static int checked = 0;

double meanvalue(double a, double b){
	return (a+b)/2;
}
__device__ void fillmatrix_aux(double *M, int ii, int nrows){
	for (int jj=0; jj<nrows; jj++){
		M[jj] = 1.0*(ii*nrows+jj);
	}
	return;
}

__device__ void fillmatrix_artificial(double *M, int nrows, int ncols){
	int ii = blockIdx.x*blockDim.x + threadIdx.x;
	if (ii <nrows){
		fillmatrix_aux(M+ii*nrows,ii,nrows);
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

__global__ void impedance_matrix_compute_elbyel2(double *Zr,
		      double *Zi,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      double *r1_arg,	      //Parameters originally in obj
		      double *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg,
		      int limx,
		      int limy)
{
	int num_chunks, chunk_size, local_index, col_index, iii, aa, jjj;
	num_chunks = 1024; //768; //1024; //3072; //Antes 1024
	chunk_size = n_r2_arg/num_chunks;
	local_index = 0;
	iii = blockIdx.x*blockDim.x + threadIdx.x;
	jjj = blockIdx.y*blockDim.y + threadIdx.y;
	if (iii<limx&& jjj<limy){ //OJO ALPHA
	      aa = impedance_matrix_single_element(Zr+iii+jjj*n_r1_arg,
	      Zi+iii+jjj*n_r1_arg,
 	      field_arg,		// Parameters originally in EM_data
	      k_arg,
	      eta_arg,
	      Rinteg_s_arg,
	      Ranal_s_arg,
	      Rinteg_f_arg,
	      cor_solid_arg,
	      flag_arg,
	      r1_arg+iii,	      //Parameters originally in obj
	      r2_arg+jjj,	      //r1_arg and r2_arg where actually int arrays, here converted 
	      vertex_arg,	
	      topol_arg,      //This should have been converted to double
	      trian_arg,      //This should have been converted to double
	      edges_arg,      //This should have been converted to double
	      un_arg,
	      ds_arg,
	      ln_arg,
	      cent_arg,
	      N_arg,
	      1,
	      1,
	      rows_vertex_arg,
	      cols_vertex_arg,
	      rows_topol_arg,
	      cols_topol_arg,
	      rows_trian_arg,
	      cols_trian_arg,
	      rows_edges_arg,
	      cols_edges_arg,
	      rows_un_arg,
	      cols_un_arg,
	      rows_ds_arg,
	      cols_ds_arg,
	      n_ln_arg,
	      rows_cent_arg,
	      cols_cent_arg);
	      //printf("I'm thread %d ending if \n",iii);
	} 
	return;
}


// end impedance_matrix_compute_elbyel2

//impedance_matrix_compute_elbyel3


__global__ void impedance_matrix_compute_elbyel3(cuDoubleComplex *Ztotal,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      double *r1_arg,	      //Parameters originally in obj
		      double *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg,
		      int limx,
		      int limy)
{
	int num_chunks, chunk_size, local_index, col_index, iii, aa, jjj;
	num_chunks = 1024; //768; //1024; //3072; //Antes 1024
	chunk_size = n_r2_arg/num_chunks;
	local_index = 0;
	iii = blockIdx.x*blockDim.x + threadIdx.x;
	jjj = blockIdx.y*blockDim.y + threadIdx.y;
	if (iii<limx && jjj<limy){ //OJO ALPHA
	      aa = impedance_matrix_single_element2(Ztotal+iii+jjj*n_r1_arg,
 	      field_arg,		// Parameters originally in EM_data
	      k_arg,
	      eta_arg,
	      Rinteg_s_arg,
	      Ranal_s_arg,
	      Rinteg_f_arg,
	      cor_solid_arg,
	      flag_arg,
	      r1_arg+iii,	      //Parameters originally in obj
	      r2_arg+jjj,	      //r1_arg and r2_arg where actually int arrays, here converted 
	      vertex_arg,	
	      topol_arg,      //This should have been converted to double
	      trian_arg,      //This should have been converted to double
	      edges_arg,      //This should have been converted to double
	      un_arg,
	      ds_arg,
	      ln_arg,
	      cent_arg,
	      N_arg,
	      1,
	      1,
	      rows_vertex_arg,
	      cols_vertex_arg,
	      rows_topol_arg,
	      cols_topol_arg,
	      rows_trian_arg,
	      cols_trian_arg,
	      rows_edges_arg,
	      cols_edges_arg,
	      rows_un_arg,
	      cols_un_arg,
	      rows_ds_arg,
	      cols_ds_arg,
	      n_ln_arg,
	      rows_cent_arg,
	      cols_cent_arg);
	} 
	return;
}

//end impedance_matrix_compute_elbyel3

extern "C" int impedance_matrix_cuda_elbyel(double *Zr,
		      double *Zi,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      double *r1_arg,	      //Parameters originally in obj
		      double *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg)
{
		//Define pointers for the structures that will be sent to device
//===============

cudaEvent_t start1, stop1;
cudaEventCreate(&start1);
cudaEventCreate(&stop1);
cudaEventRecord(start1);

//===============
		double *Zr_dev, *Zi_dev, *r1_arg_dev, *r2_arg_dev, *vertex_arg_dev, *topol_arg_dev, *trian_arg_dev, *edges_arg_dev, *un_arg_dev, *ds_arg_dev, *ln_arg_dev, *cent_arg_dev;
		//Allocate memory
		cudaMalloc(&Zr_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&Zi_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&r1_arg_dev,n_r1_arg*sizeof(double));
		cudaMalloc(&r2_arg_dev,n_r2_arg*sizeof(double));
		cudaMalloc(&vertex_arg_dev,rows_vertex_arg*cols_vertex_arg*sizeof(double));
		cudaMalloc(&topol_arg_dev,rows_topol_arg*cols_topol_arg*sizeof(double));
		cudaMalloc(&trian_arg_dev,rows_trian_arg*cols_trian_arg*sizeof(double));
		cudaMalloc(&edges_arg_dev,rows_edges_arg*cols_edges_arg*sizeof(double));
		cudaMalloc(&un_arg_dev,rows_un_arg*cols_un_arg*sizeof(double));
		cudaMalloc(&ds_arg_dev,rows_ds_arg*cols_ds_arg*sizeof(double));
		cudaMalloc(&ln_arg_dev,n_ln_arg*sizeof(double));
		cudaMalloc(&cent_arg_dev,rows_cent_arg*cols_cent_arg*sizeof(double));

		int num_chunks;
		num_chunks = 1024;  //1024; //3072; //Antes 1024

		//Copy arrays if necessary
		cudaMemcpy(r1_arg_dev,r1_arg,n_r1_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(r2_arg_dev,r2_arg,n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(vertex_arg_dev,vertex_arg,rows_vertex_arg*cols_vertex_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(topol_arg_dev,topol_arg,rows_topol_arg*cols_topol_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(trian_arg_dev,trian_arg,rows_trian_arg*cols_trian_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(edges_arg_dev,edges_arg,rows_edges_arg*cols_edges_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(un_arg_dev,un_arg,rows_un_arg*cols_un_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ds_arg_dev,ds_arg,rows_ds_arg*cols_ds_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ln_arg_dev,ln_arg,n_ln_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cent_arg_dev,cent_arg,rows_cent_arg*cols_cent_arg*sizeof(double),cudaMemcpyHostToDevice);

		cudaEventRecord(stop1);
		cudaEventSynchronize(stop1);
		float milliseconds1 = 0;
		cudaEventElapsedTime(&milliseconds1,start1,stop1);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		cudaProfilerStart();
		dim3 blocksPerGrid(100,100,1);  //96,32
		dim3 threadsPerBlock(1,1,1); //Antes 3,3,3

		impedance_matrix_compute_elbyel<<<blocksPerGrid,threadsPerBlock>>>(Zr_dev,
		      Zi_dev,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      r1_arg_dev,	      //Parameters originally in obj
		      r2_arg_dev,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      n_r1_arg,
		      n_r2_arg,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg);
		cudaProfilerStop();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds,start,stop);


		cudaEvent_t start2, stop2;
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2);
		
		cudaMemcpy(Zr,Zr_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(Zi,Zi_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		cudaEventRecord(stop2);
		cudaEventSynchronize(stop2);
		float milliseconds2 = 0;
		cudaEventElapsedTime(&milliseconds2,start2,stop2);

		return 1;
}


//end impedance_matrix_cuda_elbyel


//impedance_matrix_cuda_elbyel2

//This version is for computing arbitrary elements

extern "C" int impedance_matrix_cuda_elbyel2(double *Zr,
		      double *Zi,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      int *r1_arg,	      //Parameters originally in obj
		      int *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted. Converted to int again 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg,
		      int blocks_x,
		      int blocks_y,
		      int threads_x,
		      int threads_y,
		      int limx,
		      int limy)
{
		//Define pointers for the structures that will be sent to device

		cudaEvent_t start1, stop1;
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventRecord(start1);

		double *Zr_dev, *Zi_dev, *r1_arg_dev, *r2_arg_dev, *vertex_arg_dev, *topol_arg_dev, *trian_arg_dev, *edges_arg_dev, *un_arg_dev, *ds_arg_dev, *ln_arg_dev, *cent_arg_dev;
		//Allocate memory
		cudaMalloc(&Zr_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&Zi_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&r1_arg_dev,n_r1_arg*sizeof(double));
		cudaMalloc(&r2_arg_dev,n_r2_arg*sizeof(double));
		cudaMalloc(&vertex_arg_dev,rows_vertex_arg*cols_vertex_arg*sizeof(double));
		cudaMalloc(&topol_arg_dev,rows_topol_arg*cols_topol_arg*sizeof(double));
		cudaMalloc(&trian_arg_dev,rows_trian_arg*cols_trian_arg*sizeof(double));
		cudaMalloc(&edges_arg_dev,rows_edges_arg*cols_edges_arg*sizeof(double));
		cudaMalloc(&un_arg_dev,rows_un_arg*cols_un_arg*sizeof(double));
		cudaMalloc(&ds_arg_dev,rows_ds_arg*cols_ds_arg*sizeof(double));
		cudaMalloc(&ln_arg_dev,n_ln_arg*sizeof(double));
		cudaMalloc(&cent_arg_dev,rows_cent_arg*cols_cent_arg*sizeof(double));

		int num_chunks;
		num_chunks = 1024;  //1024; //3072; //Antes 1024

		//Copy arrays if necessary
		cudaMemcpy(r1_arg_dev,r1_arg,n_r1_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(r2_arg_dev,r2_arg,n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(vertex_arg_dev,vertex_arg,rows_vertex_arg*cols_vertex_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(topol_arg_dev,topol_arg,rows_topol_arg*cols_topol_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(trian_arg_dev,trian_arg,rows_trian_arg*cols_trian_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(edges_arg_dev,edges_arg,rows_edges_arg*cols_edges_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(un_arg_dev,un_arg,rows_un_arg*cols_un_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ds_arg_dev,ds_arg,rows_ds_arg*cols_ds_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ln_arg_dev,ln_arg,n_ln_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cent_arg_dev,cent_arg,rows_cent_arg*cols_cent_arg*sizeof(double),cudaMemcpyHostToDevice);

		cudaMemcpy(Zr_dev,Zr,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(Zi_dev,Zi,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);

		cudaEventRecord(stop1);
		cudaEventSynchronize(stop1);
		float milliseconds1 = 0;
		cudaEventElapsedTime(&milliseconds1,start1,stop1);
		
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		cudaProfilerStart();
	
		dim3 blocksPerGrid(blocks_x,blocks_y,1);  //96,32
		dim3 threadsPerBlock(threads_x,threads_y,1); //Antes 3,3,3


		impedance_matrix_compute_elbyel2<<<blocksPerGrid,threadsPerBlock>>>(Zr_dev,
		      Zi_dev,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      r1_arg_dev,	      //Parameters originally in obj
		      r2_arg_dev,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      n_r1_arg,
		      n_r2_arg,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg,
		      limx,
		      limy);
		cudaProfilerStop();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds,start,stop);
		printf("Time in cuda function = %f. ",milliseconds);
		cudaEvent_t start2, stop2;
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2);

		cudaMemcpy(Zr,Zr_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(Zi,Zi_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		printf("Right before impedance_matrix_cuda ends Zr[0] = %f\n",Zr[0]);

		cudaEventRecord(stop2);
		cudaEventSynchronize(stop2);
		float milliseconds2 = 0;
		cudaEventElapsedTime(&milliseconds2,start2,stop2);

		return 1;
}


//end impedance_matrix_cuda_elbyel2

//impedance_matrix_cuda_elbyel3

extern "C" int impedance_matrix_cuda_elbyel3(cuDoubleComplex *Ztotal,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      int *r1_arg,	      //Parameters originally in obj
		      int *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted. Converted to int again 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg,
		      int blocks_x,
		      int blocks_y,
		      int threads_x,
		      int threads_y,
		      int limx,
		      int limy)
{
		//Define pointers for the structures that will be sent to device
//===============

cudaEvent_t start1, stop1;
cudaEventCreate(&start1);
cudaEventCreate(&stop1);
cudaEventRecord(start1);

//===============
		printf("blocks_x: %d\n",blocks_x);
		printf("blocks_y: %d\n",blocks_y);

		double *r1_arg_dev, *r2_arg_dev, *vertex_arg_dev, *topol_arg_dev, *trian_arg_dev, *edges_arg_dev, *un_arg_dev, *ds_arg_dev, *ln_arg_dev, *cent_arg_dev;
		
		cuDoubleComplex *Ztotal_dev;
		//Allocate memory
		//cudaMalloc(&Zr_dev,n_r1_arg*n_r2_arg*sizeof(double));
		//cudaMalloc(&Zi_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&Ztotal_dev,n_r1_arg*n_r2_arg*sizeof(cuDoubleComplex));
		cudaMalloc(&r1_arg_dev,n_r1_arg*sizeof(double));
		cudaMalloc(&r2_arg_dev,n_r2_arg*sizeof(double));
		cudaMalloc(&vertex_arg_dev,rows_vertex_arg*cols_vertex_arg*sizeof(double));
		cudaMalloc(&topol_arg_dev,rows_topol_arg*cols_topol_arg*sizeof(double));
		cudaMalloc(&trian_arg_dev,rows_trian_arg*cols_trian_arg*sizeof(double));
		cudaMalloc(&edges_arg_dev,rows_edges_arg*cols_edges_arg*sizeof(double));
		cudaMalloc(&un_arg_dev,rows_un_arg*cols_un_arg*sizeof(double));
		cudaMalloc(&ds_arg_dev,rows_ds_arg*cols_ds_arg*sizeof(double));
		cudaMalloc(&ln_arg_dev,n_ln_arg*sizeof(double));
		cudaMalloc(&cent_arg_dev,rows_cent_arg*cols_cent_arg*sizeof(double));

		int num_chunks;
		num_chunks = 1024;  //1024; //3072; //Antes 1024

		//Copy arrays if necessary
		cudaMemcpy(r1_arg_dev,r1_arg,n_r1_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(r2_arg_dev,r2_arg,n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(vertex_arg_dev,vertex_arg,rows_vertex_arg*cols_vertex_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(topol_arg_dev,topol_arg,rows_topol_arg*cols_topol_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(trian_arg_dev,trian_arg,rows_trian_arg*cols_trian_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(edges_arg_dev,edges_arg,rows_edges_arg*cols_edges_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(un_arg_dev,un_arg,rows_un_arg*cols_un_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ds_arg_dev,ds_arg,rows_ds_arg*cols_ds_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ln_arg_dev,ln_arg,n_ln_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cent_arg_dev,cent_arg,rows_cent_arg*cols_cent_arg*sizeof(double),cudaMemcpyHostToDevice);

		//Ojo: aquí sólo inicializamos a zero
		//cudaMemcpy(Zr_dev,Zr,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		//cudaMemcpy(Zi_dev,Zi,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(Ztotal_dev,Ztotal,n_r1_arg*n_r2_arg*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);

	


//===============

cudaEventRecord(stop1);
cudaEventSynchronize(stop1);
float milliseconds1 = 0;
cudaEventElapsedTime(&milliseconds1,start1,stop1);
printf("Time in cuda part 1 = %f. ",milliseconds1);


//===============



		//Execute the cuda function
		//Antes <<<1024,3>>>
		//Luego <<<3072,1>>>
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		cudaProfilerStart();
		//dim3 blocksPerGrid(3072,3072,1);
		//dim3 threadsPerBlock(1,1,1);
		//192,12, funciona. 192,16 funciona. 615,5 funciona. 154,20 no funciona. 205,15 funciona. 171,18 no funciona. 181, 17 no funciona	
		//dim3 blocksPerGrid(181,181,1);  //96,32
		//dim3 threadsPerBlock(17,17,1); //Antes 3,3,3

		//Lo que estaba hasta ahora
		//dim3 blocksPerGrid(3072,3072,1);  //96,32
		//dim3 threadsPerBlock(1,1,1); //Antes 3,3,3

		//dim3 blocksPerGrid(16384,16384,1);  //96,32 //OJO ALPHA
		
		//dim3 blocksPerGrid(50,50,1);  //96,32
		//dim3 threadsPerBlock(1,1,1); //Antes 3,3,3
		dim3 blocksPerGrid(blocks_x,blocks_y,1);  //96,32
		dim3 threadsPerBlock(threads_x,threads_y,1); //Antes 3,3,3

		//printf("Check input of impedance_matrix_compute_elbyel3 in impedance_matrix_cuda_elbyel3\n");
		//printf("field arg: %d\n",field_arg);
		//printf("k_arg: %f\n",k_arg);
		//printf("eta_arg %f\n",eta_arg);
		//printf("Rinteg_s_arg: %f\n",Rinteg_s_arg);
		//printf("Ranal_s_arg: %f\n",Ranal_s_arg);
		//printf("Rinteg_f_arg: %f\n",Rinteg_f_arg);
		//printf("cor_solid_arg: %d\n",cor_solid_arg);
		//printf("flag_arg: %d\n",flag_arg);
		//printf("r1_arg_dev[0]: %d\n",r1_arg[0]);
		//printf("r2_arg_dev[0]: %d\n",r2_arg[0]);
		//printf("vertex_arg_dev[0]:");	



		impedance_matrix_compute_elbyel3<<<blocksPerGrid,threadsPerBlock>>>(Ztotal_dev,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      r1_arg_dev,	      //Parameters originally in obj
		      r2_arg_dev,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      n_r1_arg,
		      n_r2_arg,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg,
		      limx,
		      limy);
//
//		   cudaError_t err = cudaGetLastError();        // Get error code
//   if ( err != cudaSuccess )
// {
//      printf("CUDA Error: %s\n", cudaGetErrorString(err));
//      exit(-1);
//   }
//cudaDeviceSynchronize();
		cudaProfilerStop();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds,start,stop);
		printf("Time in cuda function = %f. ",milliseconds);
		//Copy arrays back to Host
		//fillmatrix_artificial<<<1024,3>>>(Zr_dev,n_r1_arg,n_r2_arg);
		//fillmatrix_artificial<<<1024,3>>>(Zi_dev,n_r1_arg,n_r2_arg);
		//cudaDeviceSynchronize();	
		//printf("Finnished impedance_matrix_compute\n");
	
		//printf("Lo imposible\n");

		//printf("n_r1_arg*n_r2_arg = %d \n",n_r1_arg*n_r2_arg);

//===============


cudaEvent_t start2, stop2;
cudaEventCreate(&start2);
cudaEventCreate(&stop2);
cudaEventRecord(start2);

//===============



		//cudaMemcpy(Zr,Zr_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		//cudaMemcpy(Zi,Zi_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(Ztotal,Ztotal_dev,n_r1_arg*n_r2_arg*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);

		//printf("Right before impedance_matrix_cuda ends Zr[0] = %f\n",Zr[0]);

//===============

cudaEventRecord(stop2);
cudaEventSynchronize(stop2);
float milliseconds2 = 0;
cudaEventElapsedTime(&milliseconds2,start2,stop2);
printf("Time in cuda part 2 = %f. ",milliseconds2);


//===============



	return 1;
}


//end impedance_matrix_cuda_elbyel3

//impedance_matrix_cuda_elbyel4

extern "C" double impedance_matrix_cuda_elbyel4(cuDoubleComplex *Ztotal,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      int *r1_arg,	      //Parameters originally in obj
		      int *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted. Converted to int again 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg,
		      int blocks_x,
		      int blocks_y,
		      int threads_x,
		      int threads_y,
		      int limx,
		      int limy)
{
		//Define pointers for the structures that will be sent to device
//===============

//cudaEvent_t start1, stop1;
//cudaEventCreate(&start1);
//cudaEventCreate(&stop1);
//cudaEventRecord(start1);

//===============
		printf("blocks_x: %d\n",blocks_x);
		printf("blocks_y: %d\n",blocks_y);

		double *r1_arg_dev, *r2_arg_dev, *vertex_arg_dev, *topol_arg_dev, *trian_arg_dev, *edges_arg_dev, *un_arg_dev, *ds_arg_dev, *ln_arg_dev, *cent_arg_dev;
		
		cuDoubleComplex *Ztotal_dev;
		//Allocate memory
		//cudaMalloc(&Zr_dev,n_r1_arg*n_r2_arg*sizeof(double));
		//cudaMalloc(&Zi_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&Ztotal_dev,n_r1_arg*n_r2_arg*sizeof(cuDoubleComplex));
		cudaMalloc(&r1_arg_dev,n_r1_arg*sizeof(double));
		cudaMalloc(&r2_arg_dev,n_r2_arg*sizeof(double));
		cudaMalloc(&vertex_arg_dev,rows_vertex_arg*cols_vertex_arg*sizeof(double));
		cudaMalloc(&topol_arg_dev,rows_topol_arg*cols_topol_arg*sizeof(double));
		cudaMalloc(&trian_arg_dev,rows_trian_arg*cols_trian_arg*sizeof(double));
		cudaMalloc(&edges_arg_dev,rows_edges_arg*cols_edges_arg*sizeof(double));
		cudaMalloc(&un_arg_dev,rows_un_arg*cols_un_arg*sizeof(double));
		cudaMalloc(&ds_arg_dev,rows_ds_arg*cols_ds_arg*sizeof(double));
		cudaMalloc(&ln_arg_dev,n_ln_arg*sizeof(double));
		cudaMalloc(&cent_arg_dev,rows_cent_arg*cols_cent_arg*sizeof(double));

		int num_chunks;
		num_chunks = 1024;  //1024; //3072; //Antes 1024

		//Copy arrays if necessary
		cudaMemcpy(r1_arg_dev,r1_arg,n_r1_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(r2_arg_dev,r2_arg,n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(vertex_arg_dev,vertex_arg,rows_vertex_arg*cols_vertex_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(topol_arg_dev,topol_arg,rows_topol_arg*cols_topol_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(trian_arg_dev,trian_arg,rows_trian_arg*cols_trian_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(edges_arg_dev,edges_arg,rows_edges_arg*cols_edges_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(un_arg_dev,un_arg,rows_un_arg*cols_un_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ds_arg_dev,ds_arg,rows_ds_arg*cols_ds_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ln_arg_dev,ln_arg,n_ln_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cent_arg_dev,cent_arg,rows_cent_arg*cols_cent_arg*sizeof(double),cudaMemcpyHostToDevice);

		//Ojo: aquí sólo inicializamos a zero
		//cudaMemcpy(Zr_dev,Zr,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		//cudaMemcpy(Zi_dev,Zi,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(Ztotal_dev,Ztotal,n_r1_arg*n_r2_arg*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);

	


//===============

//cudaEventRecord(stop1);
//cudaEventSynchronize(stop1);
//float milliseconds1 = 0;
//cudaEventElapsedTime(&milliseconds1,start1,stop1);
//printf("Time in cuda part 1 = %f. ",milliseconds1);


//===============



		//Execute the cuda function
		//Antes <<<1024,3>>>
		//Luego <<<3072,1>>>
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		cudaProfilerStart();
		//dim3 blocksPerGrid(3072,3072,1);
		//dim3 threadsPerBlock(1,1,1);
		//192,12, funciona. 192,16 funciona. 615,5 funciona. 154,20 no funciona. 205,15 funciona. 171,18 no funciona. 181, 17 no funciona	
		//dim3 blocksPerGrid(181,181,1);  //96,32
		//dim3 threadsPerBlock(17,17,1); //Antes 3,3,3

		//Lo que estaba hasta ahora
		//dim3 blocksPerGrid(3072,3072,1);  //96,32
		//dim3 threadsPerBlock(1,1,1); //Antes 3,3,3

		//dim3 blocksPerGrid(16384,16384,1);  //96,32 //OJO ALPHA
		
		//dim3 blocksPerGrid(50,50,1);  //96,32
		//dim3 threadsPerBlock(1,1,1); //Antes 3,3,3
		dim3 blocksPerGrid(blocks_x,blocks_y,1);  //96,32
		dim3 threadsPerBlock(threads_x,threads_y,1); //Antes 3,3,3

		//printf("Check input of impedance_matrix_compute_elbyel3 in impedance_matrix_cuda_elbyel3\n");
		//printf("field arg: %d\n",field_arg);
		//printf("k_arg: %f\n",k_arg);
		//printf("eta_arg %f\n",eta_arg);
		//printf("Rinteg_s_arg: %f\n",Rinteg_s_arg);
		//printf("Ranal_s_arg: %f\n",Ranal_s_arg);
		//printf("Rinteg_f_arg: %f\n",Rinteg_f_arg);
		//printf("cor_solid_arg: %d\n",cor_solid_arg);
		//printf("flag_arg: %d\n",flag_arg);
		//printf("r1_arg_dev[0]: %d\n",r1_arg[0]);
		//printf("r2_arg_dev[0]: %d\n",r2_arg[0]);
		//printf("vertex_arg_dev[0]:");	



		impedance_matrix_compute_elbyel3<<<blocksPerGrid,threadsPerBlock>>>(Ztotal_dev,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      r1_arg_dev,	      //Parameters originally in obj
		      r2_arg_dev,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      n_r1_arg,
		      n_r2_arg,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg,
		      limx,
		      limy);
//
//		   cudaError_t err = cudaGetLastError();        // Get error code
//   if ( err != cudaSuccess )
// {
//      printf("CUDA Error: %s\n", cudaGetErrorString(err));
//      exit(-1);
//   }
//cudaDeviceSynchronize();
		cudaProfilerStop();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds,start,stop);
		printf("Time in cuda function = %f. ",milliseconds);
		//Copy arrays back to Host
		//fillmatrix_artificial<<<1024,3>>>(Zr_dev,n_r1_arg,n_r2_arg);
		//fillmatrix_artificial<<<1024,3>>>(Zi_dev,n_r1_arg,n_r2_arg);
		//cudaDeviceSynchronize();	
		//printf("Finnished impedance_matrix_compute\n");
	
		//printf("Lo imposible\n");

		//printf("n_r1_arg*n_r2_arg = %d \n",n_r1_arg*n_r2_arg);

//===============


cudaEvent_t start2, stop2;
cudaEventCreate(&start2);
cudaEventCreate(&stop2);
cudaEventRecord(start2);

//===============



		//cudaMemcpy(Zr,Zr_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		//cudaMemcpy(Zi,Zi_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(Ztotal,Ztotal_dev,n_r1_arg*n_r2_arg*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);

		//printf("Right before impedance_matrix_cuda ends Zr[0] = %f\n",Zr[0]);

//===============

cudaEventRecord(stop2);
cudaEventSynchronize(stop2);
float milliseconds2 = 0;
cudaEventElapsedTime(&milliseconds2,start2,stop2);
printf("Time in cuda part 2 = %f. ",milliseconds2);


//===============



	return milliseconds;
}


//end impedance_matrix_cuda_elbyel4


extern "C" int cuda_cur_pure(double *Zr,
		      double *Zi,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      int *r1_arg,	      //Parameters originally in obj
		      int *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted. Converted to int again 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      double *Cmat_r,
		      double *Cmat_i,
		      double *Umat_r,
		      double *Umat_i,
		      double *Rmat_r,
		      double *Rmat_i,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg,
		      int blocks_x,
		      int blocks_y,
		      int threads_x,
		      int threads_y,
		      int limx,
		      int limy,
		      double tol_compress)
{
		//Define pointers for the structures that will be sent to device
//===============

//===============
		double *Zr_dev, *Zi_dev, *r1_arg_dev, *r2_arg_dev, *vertex_arg_dev, *topol_arg_dev, *trian_arg_dev, *edges_arg_dev, *un_arg_dev, *ds_arg_dev, *ln_arg_dev, *cent_arg_dev;
		//Allocate memory
		cudaMalloc(&Zr_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&Zi_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&r1_arg_dev,n_r1_arg*sizeof(double));
		cudaMalloc(&r2_arg_dev,n_r2_arg*sizeof(double));
		cudaMalloc(&vertex_arg_dev,rows_vertex_arg*cols_vertex_arg*sizeof(double));
		cudaMalloc(&topol_arg_dev,rows_topol_arg*cols_topol_arg*sizeof(double));
		cudaMalloc(&trian_arg_dev,rows_trian_arg*cols_trian_arg*sizeof(double));
		cudaMalloc(&edges_arg_dev,rows_edges_arg*cols_edges_arg*sizeof(double));
		cudaMalloc(&un_arg_dev,rows_un_arg*cols_un_arg*sizeof(double));
		cudaMalloc(&ds_arg_dev,rows_ds_arg*cols_ds_arg*sizeof(double));
		cudaMalloc(&ln_arg_dev,n_ln_arg*sizeof(double));
		cudaMalloc(&cent_arg_dev,rows_cent_arg*cols_cent_arg*sizeof(double));

		int num_chunks;
		num_chunks = 1024;  //1024; //3072; //Antes 1024

		//Copy arrays if necessary
		cudaMemcpy(r1_arg_dev,r1_arg,n_r1_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(r2_arg_dev,r2_arg,n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(vertex_arg_dev,vertex_arg,rows_vertex_arg*cols_vertex_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(topol_arg_dev,topol_arg,rows_topol_arg*cols_topol_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(trian_arg_dev,trian_arg,rows_trian_arg*cols_trian_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(edges_arg_dev,edges_arg,rows_edges_arg*cols_edges_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(un_arg_dev,un_arg,rows_un_arg*cols_un_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ds_arg_dev,ds_arg,rows_ds_arg*cols_ds_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ln_arg_dev,ln_arg,n_ln_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cent_arg_dev,cent_arg,rows_cent_arg*cols_cent_arg*sizeof(double),cudaMemcpyHostToDevice);

		//Ojo: aquí sólo inicializamos a zero
		cudaMemcpy(Zr_dev,Zr,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(Zi_dev,Zi,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
	


//===============


//===============

		dim3 blocksPerGrid(blocks_x,blocks_y,1);  //96,32
		dim3 threadsPerBlock(threads_x,threads_y,1); //Antes 3,3,3


		impedance_matrix_compute_elbyel2<<<blocksPerGrid,threadsPerBlock>>>(Zr_dev,
		      Zi_dev,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      r1_arg_dev,	      //Parameters originally in obj
		      r2_arg_dev,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      n_r1_arg,
		      n_r2_arg,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg,
		      limx,
		      limy);
//===============

//===============



		cudaMemcpy(Zr,Zr_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(Zi,Zi_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);

//===============

//===============



	return 1;


}

//kernel_Sinv
__global__ void kernel_Sinv(double *d_vec, cuDoubleComplex *d_vec_complex, int N, double treshold){ 
	        //Ferorms the filtering operation and transforms to cuDoubleComplex typoe    
	        int ii = blockIdx.x*blockDim.x + threadIdx.x; 
		if (d_vec[ii]<treshold){      
			d_vec_complex[ii] = make_cuDoubleComplex(0.0,0.0); 
		} else {
			d_vec_complex[ii] = make_cuDoubleComplex(1/d_vec[ii],0.0); 
		}
 		return;
}		
//end kernel_Sinv

///////////////////////////////for debug prueba

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





//cuda_complete_compression

extern "C" double cuda_complete_compression(cuDoubleComplex *Ccompressed,
		      cuDoubleComplex *Ucompressed,
		      cuDoubleComplex *Rcompressed,
		      int cur_order,
		      double *vec_time_cuda,
		      double pinv_tol,
		      cuDoubleComplex *Avtest_old,
		      cuDoubleComplex *Avtest_new,
		      cuDoubleComplex *vtest,
		      int *col_samples, //Cambiado a int
		      int *row_samples, //Cambiadno a int
		      int C_blocks_x,
		      int C_blocks_y,
		      int C_threads_x,
		      int C_threads_y,
		      int U_blocks_x,
		      int U_blocks_y,
		      int U_threads_x,
		      int U_threads_y,
		      int R_blocks_x,
		      int R_blocks_y,
		      int R_threads_x,
		      int R_threads_y,
		      cuDoubleComplex *Ztotal,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      int *r1_arg,	      //Parameters originally in obj
		      int *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted. Converted to int again 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg,
		      int blocks_x,
		      int blocks_y,
		      int threads_x,
		      int threads_y,
		      int limx,
		      int limy)
{
		double *r1_arg_dev, *r2_arg_dev, *vertex_arg_dev, *topol_arg_dev, *trian_arg_dev, *edges_arg_dev, *un_arg_dev, *ds_arg_dev, *ln_arg_dev, *cent_arg_dev;
	
		double *d_col_samples, *d_row_samples;
		double err_here;

		double norm_Avtest_new = 0.0;
		double norm_Avtest_diff = 0.0;

		cuDoubleComplex *Ztotal_dev, *d_C, *d_U, *d_R, *d_Uintersection, *d_vtest, *d_Avtest_old, *d_Avtest_new, *d_aux1, *d_aux2, *d_aux3;


		//For pinv_operations, 
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

		double *d_S = NULL;  
       		cuDoubleComplex *d_Vt = NULL;  
		cuDoubleComplex *d_aux_V = NULL;
		//cuDoubleComplex d_pinvA = NULL;
		cuDoubleComplex *d_Sinv = NULL;
		cuDoubleComplex *d_Usvd = NULL; //Es la U de la descomposición SVD
		int *devInfo = NULL;
		cuDoubleComplex *d_work = NULL;   
		double *d_rwork = NULL;  
        	int lwork = 0;   
		const cuDoubleComplex h_alpha = make_cuDoubleComplex(1.0,0.0);
		const cuDoubleComplex h_beta  = make_cuDoubleComplex(0.0,0.0);
		int lda = cur_order;
	        cusolver_status = cusolverDnCreate(&cusolverH);  
		cublas_status = cublasCreate(&cublasH);   

		cudaMalloc(&r1_arg_dev,n_r1_arg*sizeof(double));
		cudaMalloc(&r2_arg_dev,n_r2_arg*sizeof(double));
		cudaMalloc(&vertex_arg_dev,rows_vertex_arg*cols_vertex_arg*sizeof(double));
		cudaMalloc(&topol_arg_dev,rows_topol_arg*cols_topol_arg*sizeof(double));
		cudaMalloc(&trian_arg_dev,rows_trian_arg*cols_trian_arg*sizeof(double));
		cudaMalloc(&edges_arg_dev,rows_edges_arg*cols_edges_arg*sizeof(double));
		cudaMalloc(&un_arg_dev,rows_un_arg*cols_un_arg*sizeof(double));
		cudaMalloc(&ds_arg_dev,rows_ds_arg*cols_ds_arg*sizeof(double));
		cudaMalloc(&ln_arg_dev,n_ln_arg*sizeof(double));
		cudaMalloc(&cent_arg_dev,rows_cent_arg*cols_cent_arg*sizeof(double));

		cudaEvent_t start1, stop1;
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventRecord(start1);

		cudaMalloc(&d_C,n_r1_arg*cur_order*sizeof(cuDoubleComplex));
		cudaMalloc(&d_U,cur_order*cur_order*sizeof(cuDoubleComplex));
		cudaMalloc(&d_Uintersection,cur_order*cur_order*sizeof(cuDoubleComplex));
		cudaMalloc(&d_R,cur_order*n_r2_arg*sizeof(cuDoubleComplex));
		cudaMalloc(&d_vtest,n_r2_arg*sizeof(cuDoubleComplex));
		cudaMalloc(&d_Avtest_old,n_r1_arg*sizeof(cuDoubleComplex));
		cudaMalloc(&d_Avtest_new,n_r1_arg*sizeof(cuDoubleComplex));
		cudaMalloc(&d_aux1,cur_order*sizeof(cuDoubleComplex));
		cudaMalloc(&d_aux2,cur_order*sizeof(cuDoubleComplex));
		cudaMalloc(&d_aux3,n_r1_arg*sizeof(cuDoubleComplex));
		cudaMalloc(&d_col_samples,cur_order*sizeof(double));
		cudaMalloc(&d_row_samples,cur_order*sizeof(double));
		
		cudaMalloc((void**)&d_S, sizeof(double)*cur_order); 
		cudaMalloc((void**)&d_Usvd, sizeof(cuDoubleComplex)*cur_order*cur_order);	
		cudaMalloc((void**)&d_Vt, sizeof(cuDoubleComplex)*cur_order*cur_order);
		cudaMalloc((void**)&d_aux_V, sizeof(cuDoubleComplex)*cur_order*cur_order);
        	cudaMalloc((void**)&d_Sinv, sizeof(cuDoubleComplex)*cur_order); 
		cudaMalloc((void**)&devInfo,sizeof(int));  

		cudaEventRecord(stop1);
		cudaEventSynchronize(stop1);

		float milliseconds1 = 0.0;
		cudaEventElapsedTime(&milliseconds1,start1,stop1);
		cudaMemcpy(r1_arg_dev,r1_arg,n_r1_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(r2_arg_dev,r2_arg,n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(vertex_arg_dev,vertex_arg,rows_vertex_arg*cols_vertex_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(topol_arg_dev,topol_arg,rows_topol_arg*cols_topol_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(trian_arg_dev,trian_arg,rows_trian_arg*cols_trian_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(edges_arg_dev,edges_arg,rows_edges_arg*cols_edges_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(un_arg_dev,un_arg,rows_un_arg*cols_un_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ds_arg_dev,ds_arg,rows_ds_arg*cols_ds_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ln_arg_dev,ln_arg,n_ln_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cent_arg_dev,cent_arg,rows_cent_arg*cols_cent_arg*sizeof(double),cudaMemcpyHostToDevice);

		cudaEvent_t start2, stop2;
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2);

		cudaMemset(d_C,0,n_r1_arg*cur_order*sizeof(cuDoubleComplex));
		cudaMemset(d_U,0,cur_order*cur_order*sizeof(cuDoubleComplex));
		cudaMemset(d_Uintersection,0,cur_order*cur_order*sizeof(cuDoubleComplex));
		cudaMemset(d_R,0,cur_order*n_r2_arg*sizeof(cuDoubleComplex));

		cudaMemcpy(d_Avtest_old,Avtest_old,n_r1_arg*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice); 
		
		cudaMemcpy(d_vtest,vtest,n_r2_arg*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
		cudaMemcpy(d_row_samples,row_samples,cur_order*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_col_samples,col_samples,cur_order*sizeof(double),cudaMemcpyHostToDevice);

		dim3 C_blocksPerGrid(C_blocks_x,C_blocks_y,1);  //96,32
		dim3 C_threadsPerBlock(C_threads_x,C_threads_y,1); //Antes 3,3,3
		

		impedance_matrix_compute_elbyel3<<<C_blocksPerGrid,C_threadsPerBlock>>>(d_C,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      r1_arg_dev,	      //Parameters originally in obj
		      d_col_samples,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      n_r1_arg,
		      cur_order,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg,
		      n_r1_arg,
		      cur_order);

////////////Computing U //cuando esta sólo el segundo, segfault
		dim3 U_blocksPerGrid(U_blocks_x,U_blocks_y,1);  //96,32
		dim3 U_threadsPerBlock(U_threads_x,U_threads_y,1); //Antes 3,3,3
		
		//He cambiado d_Uintersection por d_U
		impedance_matrix_compute_elbyel3<<<U_blocksPerGrid,U_threadsPerBlock>>>(d_Uintersection,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      d_row_samples,	      //Parameters originally in obj
		      d_col_samples,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      cur_order,
		      cur_order,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg,
		      cur_order,
		      cur_order);
//////Computing R	


		dim3 R_blocksPerGrid(R_blocks_x,R_blocks_y,1);  //96,32
		dim3 R_threadsPerBlock(R_threads_x,R_threads_y,1); //Antes 3,3,3


		impedance_matrix_compute_elbyel3<<<R_blocksPerGrid,R_threadsPerBlock>>>(d_R,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      d_row_samples,	      //Parameters originally in obj
		      r2_arg_dev,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      cur_order,
		      n_r2_arg,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg,
		      cur_order,
		      n_r2_arg);

        cusolver_status = cusolverDnZgesvd_bufferSize(
			cusolverH,
			cur_order,
			cur_order,
			&lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);   
        cudaStat1 = cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex)*lwork);  
      	cudaMalloc((void**)&d_rwork, sizeof(double)*(cur_order-1));
        assert(cudaSuccess == cudaStat1);          
        signed char jobu = 'S'; // all m columns of U  
        signed char jobvt = 'A'; // all n columns of VT 

	cusolver_status = cusolverDnZgesvd(
                        cusolverH,
                        jobu,
                        jobvt,
                        cur_order,
                        cur_order,
                        d_Uintersection,
                        lda,
                        d_S,
                        d_Usvd,
                        lda, //ldu
                        d_Vt,
                        cur_order, //ldvt (antes lda)
                        d_work,
                        lwork,
                        d_rwork,
                        devInfo);

        cudaStat1 = cudaDeviceSynchronize();
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
        assert(cudaSuccess == cudaStat1);


        kernel_Sinv<<<cur_order,1>>>(d_S, d_Sinv, cur_order, pinv_tol);

        cublas_status = cublasZdgmm(
                        cublasH, CUBLAS_SIDE_LEFT,
                        cur_order,cur_order,
                        d_Vt, cur_order,
                        d_Sinv,1,
                        d_aux_V,cur_order);
        cublas_status = cublasZgemm(
                        cublasH,
                        CUBLAS_OP_C, CUBLAS_OP_C,
                        cur_order, cur_order, cur_order,
                        &h_alpha,
                        d_aux_V, cur_order,
                        d_Usvd, cur_order,
                        &h_beta,
                        d_U,cur_order);



        //R*vtest
        cublas_status = cublasZgemm(
                        cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        cur_order, 1, n_r2_arg,
                        &h_alpha,
                        d_R, cur_order,
                        d_vtest,n_r2_arg,
                        &h_beta,
                        d_aux1,cur_order);
        //U*d_aux1 = U*(R*vtest)
        cublas_status = cublasZgemm(
                        cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        cur_order,1 , cur_order,
                        &h_alpha,
                        d_U, cur_order,
                        d_aux1, cur_order,
                        &h_beta,
                        d_aux2,cur_order);
        //C*d_aux12 = C*(U*(R*vtest))
        cublas_status = cublasZgemm(
                        cublasH,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n_r1_arg, 1 , cur_order,
                        &h_alpha,
                        d_C, n_r1_arg,
                        d_aux2, cur_order,
                        &h_beta,
                        d_Avtest_new,n_r1_arg);
        cublas_status = cublasDznrm2(cublasH,n_r1_arg,d_Avtest_new,1,&norm_Avtest_new);
        d_substract(d_aux3,d_Avtest_old,d_Avtest_new,n_r1_arg);
        cublas_status = cublasDznrm2(cublasH,n_r1_arg,d_aux3,1,&norm_Avtest_diff);

	
        err_here = norm_Avtest_diff/norm_Avtest_new;
	
	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);
	float milliseconds2 = 0.0;
	cudaEventElapsedTime(&milliseconds2,start2,stop2);

	vec_time_cuda[0] = milliseconds1 + milliseconds2;
	
	cudaMemcpy(Ccompressed,d_C,n_r1_arg*cur_order*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	cudaMemcpy(Ucompressed,d_U,cur_order*cur_order*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	cudaMemcpy(Rcompressed,d_R,cur_order*n_r2_arg*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	cudaMemcpy(Avtest_new,d_Avtest_new, sizeof(cuDoubleComplex)*n_r1_arg,cudaMemcpyDeviceToHost);
	cudaFree(r1_arg_dev);
	cudaFree(r2_arg_dev);
	cudaFree(vertex_arg_dev);
	cudaFree(topol_arg_dev);
	cudaFree(trian_arg_dev);
	cudaFree(edges_arg_dev);
	cudaFree(un_arg_dev);
	cudaFree(ds_arg_dev);
	cudaFree(ln_arg_dev);
	cudaFree(cent_arg_dev);
	cudaFree(d_col_samples);
	cudaFree(d_row_samples);
//	cudaFree(Ztotal_dev);
	cudaFree(d_C);
	cudaFree(d_U);
	cudaFree(d_R);
	cudaFree(d_Uintersection);
	cudaFree(d_vtest);
	cudaFree(d_Avtest_old);
	cudaFree(d_Avtest_new);
	cudaFree(d_aux1);
	cudaFree(d_aux2);
	cudaFree(d_aux3);
	cudaFree(d_S);
	cudaFree(d_Vt);
	cudaFree(d_aux_V);
	cudaFree(d_Sinv);
	cudaFree(d_Usvd);
	cudaFree(devInfo);
	cudaFree(d_work);
	cudaFree(d_rwork);
//
	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);

	return err_here;
}

//end cuda_complete_compression

extern "C" int impedance_matrix_cuda(double *Zr,
		      double *Zi,
	 	      int field_arg,		// Parameters originally in EM_data
		      double k_arg,
		      double eta_arg,
		      double Rinteg_s_arg,
		      double Ranal_s_arg,
		      double Rinteg_f_arg,
		      int cor_solid_arg,
		      int flag_arg,
		      double *r1_arg,	      //Parameters originally in obj
		      double *r2_arg,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      double *vertex_arg,	
		      double *topol_arg,      //This should have been converted to double
		      double *trian_arg,      //This should have been converted to double
		      double *edges_arg,      //This should have been converted to double
		      double *un_arg,
		      double *ds_arg,
		      double *ln_arg,
		      double *cent_arg,
		      int N_arg,
		      int n_r1_arg,
		      int n_r2_arg,
		      int rows_vertex_arg,
		      int cols_vertex_arg,
		      int rows_topol_arg,
		      int cols_topol_arg,
		      int rows_trian_arg,
		      int cols_trian_arg,
		      int rows_edges_arg,
		      int cols_edges_arg,
		      int rows_un_arg,
		      int cols_un_arg,
		      int rows_ds_arg,
		      int cols_ds_arg,
		      int n_ln_arg,
		      int rows_cent_arg,
		      int cols_cent_arg)
{
		cudaEvent_t start1, stop1;
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventRecord(start1);

		double *Zr_dev, *Zi_dev, *r1_arg_dev, *r2_arg_dev, *vertex_arg_dev, *topol_arg_dev, *trian_arg_dev, *edges_arg_dev, *un_arg_dev, *ds_arg_dev, *ln_arg_dev, *cent_arg_dev;
		//Allocate memory
		cudaMalloc(&Zr_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&Zi_dev,n_r1_arg*n_r2_arg*sizeof(double));
		cudaMalloc(&r1_arg_dev,n_r1_arg*sizeof(double));
		cudaMalloc(&r2_arg_dev,n_r2_arg*sizeof(double));
		cudaMalloc(&vertex_arg_dev,rows_vertex_arg*cols_vertex_arg*sizeof(double));
		cudaMalloc(&topol_arg_dev,rows_topol_arg*cols_topol_arg*sizeof(double));
		cudaMalloc(&trian_arg_dev,rows_trian_arg*cols_trian_arg*sizeof(double));
		cudaMalloc(&edges_arg_dev,rows_edges_arg*cols_edges_arg*sizeof(double));
		cudaMalloc(&un_arg_dev,rows_un_arg*cols_un_arg*sizeof(double));
		cudaMalloc(&ds_arg_dev,rows_ds_arg*cols_ds_arg*sizeof(double));
		cudaMalloc(&ln_arg_dev,n_ln_arg*sizeof(double));
		cudaMalloc(&cent_arg_dev,rows_cent_arg*cols_cent_arg*sizeof(double));

		int num_chunks;
		num_chunks = 1024;  //1024; //3072; //Antes 1024

		//Copy arrays if necessary
		cudaMemcpy(r1_arg_dev,r1_arg,n_r1_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(r2_arg_dev,r2_arg,n_r2_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(vertex_arg_dev,vertex_arg,rows_vertex_arg*cols_vertex_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(topol_arg_dev,topol_arg,rows_topol_arg*cols_topol_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(trian_arg_dev,trian_arg,rows_trian_arg*cols_trian_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(edges_arg_dev,edges_arg,rows_edges_arg*cols_edges_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(un_arg_dev,un_arg,rows_un_arg*cols_un_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ds_arg_dev,ds_arg,rows_ds_arg*cols_ds_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ln_arg_dev,ln_arg,n_ln_arg*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cent_arg_dev,cent_arg,rows_cent_arg*cols_cent_arg*sizeof(double),cudaMemcpyHostToDevice);

		char *Tcomp_s, *Tcomp_f;
		int *Tlist_s, *Tlist_f, *Elist_s, *Elist_f;
		Tlist_f = (int*)calloc(2*n_r1_arg,sizeof(int));
		Tlist_s = (int*)calloc(2*n_r2_arg,sizeof(int)); //commented for memset
		Tcomp_f = (char*)calloc(cols_trian_arg,sizeof(char));
		Tcomp_s = (char*)calloc(cols_trian_arg,sizeof(char));
		Tcomp_s = (char*)calloc(num_chunks*cols_trian_arg,sizeof(char)); //commented for memset
		
		Elist_f = (int*)calloc(n_ln_arg,sizeof(int));
		Elist_s = (int*)calloc(num_chunks*n_ln_arg,sizeof(int)); //commented for memset

		int nTf;
		fill_row_arrays(field_arg,              // Parameters originally in EM_data
                      k_arg,
                      eta_arg,
                      Rinteg_s_arg,
                      Ranal_s_arg,
                      Rinteg_f_arg,
                      cor_solid_arg,
                      flag_arg,
                      r1_arg,         //Parameters originally in obj
                      r2_arg,         //r1_arg and r2_arg where actually int arrays, here converted
                      vertex_arg,
                      topol_arg,      //This should have been converted to double
                      trian_arg,      //This should have been converted to double
                      edges_arg,      //This should have been converted to double
                      un_arg,
                      ds_arg,
                      ln_arg,
                      cent_arg,
                      N_arg,
                      n_r1_arg,
                      n_r2_arg,
                      rows_vertex_arg,
                      cols_vertex_arg,
                      rows_topol_arg,
                      cols_topol_arg,
                      rows_trian_arg,
                      cols_trian_arg,
                      rows_edges_arg,
                      cols_edges_arg,
                      rows_un_arg,
                      cols_un_arg,
                      rows_ds_arg,
                      cols_ds_arg,
                      n_ln_arg,
                      rows_cent_arg,
                      cols_cent_arg,
                      Tlist_f,
                      Tcomp_f,
                      Elist_f,
                      &nTf);

		//Bring lists to device (important: initilizing the _s list in host and copying them to device is inifficiente. Better use memset)
		char *Tcomp_s_dev, *Tcomp_f_dev;
		int *Tlist_s_dev, *Tlist_f_dev, *Elist_s_dev, *Elist_f_dev;
		
		cudaMalloc(&Tlist_f_dev,2*n_r1_arg*sizeof(int));
		cudaMalloc(&Tlist_s_dev,2*n_r2_arg*sizeof(int));
		cudaMalloc(&Tcomp_f_dev,cols_trian_arg*sizeof(char));
		cudaMalloc(&Tcomp_s_dev,num_chunks*cols_trian_arg*sizeof(char));
		cudaMalloc(&Elist_f_dev,n_ln_arg*sizeof(int));
		cudaMalloc(&Elist_s_dev,num_chunks*n_ln_arg*sizeof(int));

		cudaMemcpy(Tlist_f_dev,Tlist_f,2*n_r1_arg*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(Tlist_s_dev,Tlist_s,2*n_r2_arg*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(Tcomp_f_dev,Tcomp_f,cols_trian_arg*sizeof(char),cudaMemcpyHostToDevice);
		cudaMemcpy(Tcomp_s_dev,Tcomp_s,num_chunks*cols_trian_arg*sizeof(char),cudaMemcpyHostToDevice);
		cudaMemcpy(Elist_f_dev,Elist_f,n_ln_arg*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(Elist_s_dev,Elist_s,num_chunks*n_ln_arg*sizeof(int),cudaMemcpyHostToDevice);
		cudaEventRecord(stop1);
		cudaEventSynchronize(stop1);
		float milliseconds1 = 0;
		cudaEventElapsedTime(&milliseconds1,start1,stop1);


		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		impedance_matrix_compute<<<1024,3>>>(Zr_dev,
		      Zi_dev,
	 	      field_arg,		// Parameters originally in EM_data
		      k_arg,
		      eta_arg,
		      Rinteg_s_arg,
		      Ranal_s_arg,
		      Rinteg_f_arg,
		      cor_solid_arg,
		      flag_arg,
		      r1_arg_dev,	      //Parameters originally in obj
		      r2_arg_dev,	      //r1_arg and r2_arg where actually int arrays, here converted 
		      vertex_arg_dev,	
		      topol_arg_dev,      //This should have been converted to double
		      trian_arg_dev,      //This should have been converted to double
		      edges_arg_dev,      //This should have been converted to double
		      un_arg_dev,
		      ds_arg_dev,
		      ln_arg_dev,
		      cent_arg_dev,
		      N_arg,
		      n_r1_arg,
		      n_r2_arg,
		      rows_vertex_arg,
		      cols_vertex_arg,
		      rows_topol_arg,
		      cols_topol_arg,
		      rows_trian_arg,
		      cols_trian_arg,
		      rows_edges_arg,
		      cols_edges_arg,
		      rows_un_arg,
		      cols_un_arg,
		      rows_ds_arg,
		      cols_ds_arg,
		      n_ln_arg,
		      rows_cent_arg,
		      cols_cent_arg,
		      Tlist_f_dev,
		      Tcomp_f_dev,
		      Elist_f_dev,
		      nTf,
		      Tlist_s_dev,
		      Tcomp_s_dev,
		      Elist_s_dev);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds,start,stop);

		cudaEvent_t start2, stop2;
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2);


		cudaMemcpy(Zr,Zr_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		cudaMemcpy(Zi,Zi_dev,n_r1_arg*n_r2_arg*sizeof(double),cudaMemcpyDeviceToHost);
		cudaEventRecord(stop2);
		cudaEventSynchronize(stop2);
		float milliseconds2 = 0;
		cudaEventElapsedTime(&milliseconds2,start2,stop2);





	return 1;
}


