#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include<math.h>


#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////////
void invert(float** src, float** dst, int n, int batchSize)
{
    	cublasHandle_t handle;
    	cublascall(cublasCreate_v2(&handle));

    	int *P, *INFO;

    	cudacall(cudaMalloc(&P, n * batchSize * sizeof(int)));
    	cudacall(cudaMalloc(&INFO,  batchSize * sizeof(int)));

    	int lda = n;

    	float **A = (float **)malloc(batchSize*sizeof(float *));
    	float **A_d, *A_dflat;

    	cudacall(cudaMalloc(&A_d,batchSize*sizeof(float *)));
    	cudacall(cudaMalloc(&A_dflat, n*n*batchSize*sizeof(float)));

	A[0] = A_dflat;
    	for (int i = 1; i < batchSize; i++)
      		A[i] = A[i-1]+(n*n);

    	cudacall(cudaMemcpy(A_d,A,batchSize*sizeof(float *),cudaMemcpyHostToDevice));
   
 	for (int i = 0; i < batchSize; i++)
      		cudacall(cudaMemcpy(A_dflat+(i*n*n), src[i], n*n*sizeof(float), cudaMemcpyHostToDevice));


    	cublascall(cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize));


    	int INFOh[batchSize];
    	cudacall(cudaMemcpy(INFOh,INFO,batchSize*sizeof(int),cudaMemcpyDeviceToHost));

    	for (int i = 0; i < batchSize; i++)
      		if(INFOh[i]  != 0)
      		{
        		fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
        		cudaDeviceReset();
        		exit(EXIT_FAILURE);
      		}

    	float **C = (float **)malloc(batchSize*sizeof(float *));
    	float **C_d, *C_dflat;

    	cudacall(cudaMalloc(&C_d,batchSize*sizeof(float *)));
    	cudacall(cudaMalloc(&C_dflat, n*n*batchSize*sizeof(float)));
    	C[0] = C_dflat;
    	for (int i = 1; i < batchSize; i++)
      		C[i] = C[i-1] + (n*n);
    	cudacall(cudaMemcpy(C_d,C,batchSize*sizeof(float *),cudaMemcpyHostToDevice));
    	cublascall(cublasSgetriBatched(handle,n,(const float **)A_d,lda,P,C_d,lda,INFO,batchSize));

    	cudacall(cudaMemcpy(INFOh,INFO,batchSize*sizeof(int),cudaMemcpyDeviceToHost));

    	for (int i = 0; i < batchSize; i++)
	      	if(INFOh[i] != 0)
      		{
        		fprintf(stderr, "Inversion of matrix %d Failed: Matrix may be singular\n", i);
        		cudaDeviceReset();
      			exit(EXIT_FAILURE);
      		}
    	for (int i = 0; i < batchSize; i++)
      	cudacall(cudaMemcpy(dst[i], C_dflat + (i*n*n), n*n*sizeof(float), cudaMemcpyDeviceToHost));
    	
	cudaFree(A_d); cudaFree(A_dflat); free(A);
	cudaFree(C_d); cudaFree(C_dflat); free(C);
    	cudaFree(P); cudaFree(INFO); cublasDestroy_v2(handle);
}


////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////////


__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width) 
{
    //2D Thread ID
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;

    //Pvalue stores the Pd element that is computed by the thread
    float Pvalue = 0;
	if(col<Width && row < Width)
	{
		    for(int k = 0; k < Width ; ++k) 
		    {
		        float Mdelement = Md[row*Width + k];
		        float Ndelement = Nd[k*Width + col];
		        Pvalue += (Mdelement*Ndelement);
		
		    }
		    Pd[row*Width + col] = Pvalue;
	}
}



void mul(float* M,float* N,int Width)
{
		
	float * P = (float *) malloc(Width*Width*sizeof(float));
	float *Md, *Nd, *Pd;



	unsigned long int size = Width*Width*sizeof(float);
  

    //Transfer M and N to device memory
    	cudaMalloc((void**)&Md, size);
    	cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);

    	cudaMalloc((void**)&Nd, size);
    	cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);

    	//Allocate P on the device
    	cudaMalloc((void**)&Pd,size);

    	//Setup the execution configuration
    	dim3 dimBlock(Width,Width);
    	dim3 dimGrid(1,1);


	if (Width*Width > 1024)
	{
		//printf("\n\n enter inside if condi\n\n");
		
		dimGrid.x = (Width-1)/32+1;
        	dimGrid.y = (Width-1)/32+1;
	
		dimBlock.x = 32;
	        dimBlock.y = 32;



	}

     
    //Launch the device computation threads!
	MatrixMulKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd,Width);

    //Transfer P from device to host
    	cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);

    //Free device matrices
    	cudaFree(Md);
    	cudaFree(Nd);
    	cudaFree(Pd);

	int i;

	fprintf(stdout,"\n\n");

	if(Width<11)
	{


		fprintf(stdout,"\n\nMatrix Multiplication, M x Inv(M) :\n\n");
		for(i = 0; i < Width*Width; i++)
   		{
			if(P[i])
				fprintf(stdout,"%10f ",P[i]) ;
			else
				fprintf(stdout,"%9f ",P[i]) ;
	        	



			if((i+1)%Width==0)
				fprintf(stdout,"\n");
		}
  

	}
	else
	{
		FILE *fp;	
	
		fp = fopen("Mat_Inv_out", "a");

 		if (!fp) 
		{
	    		fprintf(stderr, "Failed to open matAdata.\n");
	    		exit(1);
	  	}
		fprintf(fp,"\n\nMatrix Multiplication, M x Inv(M) :\n\n");
	 	for(i = 0; i < Width*Width; i++)
   		{	if(P[i])
				fprintf(fp,"%10f ",P[i]) ;
			else
				fprintf(fp,"%9f ",P[i]) ;
			
		        if((i+1)%Width==0)
				fprintf(fp,"\n");
		}
    		fclose(fp);
	}

	
	//printf("\n Matrix multiplication completed !!\n\n"); 
	free(M);
	free(N);
	free(P);

}


////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////////


void fill(float* h, int w)
{ 
	
	unsigned int i, num;
	int divide;
	FILE *f;

	f=fopen("/dev/urandom", "r");
	if (!f) {
        	fprintf(stderr, "Failed open file\n");
        	exit(1);
    	}
	for(i=0; i< w*w; i++)
	{
		fread(&num, sizeof(unsigned int), 1, f);
		fread(&divide, sizeof(int), 1, f);
		h[i] = ((float)num)/((float)divide);
		//scanf("%f",&h[i]);
	}
	fclose(f);
/*
	unsigned int i;
	srand((unsigned int)time(NULL));
	for(i=0; i< w*w; i++)
	{
		h[i] = ((float)rand()/(float)(RAND_MAX)) * 99;
		//scanf("%f",&h[i]);
	}
	
*/

} 

////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////////

void test_invert(int n )
{
    	
	//printf("Enter the order of the square matrix :");
	//scanf("%d",&n);
    	const int mybatch = 1;


	//float* mat1[n * n];
	float mat1_size = sizeof(float) * n * n;
    	float* mat1 = (float*) malloc(mat1_size);

	fill(mat1, n);

    	float *result_flat = (float *)malloc(mybatch*n*n*sizeof(float));
    	float **results = (float **)malloc(mybatch*sizeof(float *));

    	for (int i = 0; i < mybatch; i++)
      		results[i] = result_flat + (i*n*n);

    	float **inputs = (float **)malloc(mybatch*sizeof(float *));
    	
	//inputs[0]  = zero_pivot;

	inputs[0]  = mat1;


	invert(inputs, results, n, mybatch);

	if(n<11)
	{

		for (int qq = 0; qq < mybatch; qq++)
		{
	      		if(mybatch==1)
				fprintf(stdout, "Input Matrix, M :\n\n");
			else
				fprintf(stdout, "Input Matrix %d:\n\n", qq);
			
	      		for(int i=0; i<n; i++)
	      		{
	        		for(int j=0; j<n; j++)
				{	
					if(inputs[qq][i*n+j])
		            			fprintf(stdout,"%12f ",inputs[qq][i*n+j]);
					else
						fprintf(stdout,"%11f ",inputs[qq][i*n+j]);
				}
	        			fprintf(stdout,"\n");
	      		}
	    	}
	    	fprintf(stdout,"\n\n");




	    	for (int qq = 0; qq < mybatch; qq++)
		{

			if(mybatch==1)
				fprintf(stdout, "Inverse of the Input Matrix, Inv(M):\n\n");
			else
				fprintf(stdout, "Inverse Matrix %d:\n\n", qq);
	      		for(int i=0; i<n; i++)
	      		{
	        		for(int j=0; j<n; j++)
				{
					if(results[qq][i*n+j])
		            			fprintf(stdout,"%10f ",results[qq][i*n+j]);
					else
		            			fprintf(stdout,"%9f ",results[qq][i*n+j]);
	        		
				}
				fprintf(stdout,"\n");
	      		}
	    	}
	}


	else // order of the matrix is more than 10 x 10 then output the results in the file
	{
		printf("\nThe order of matrix is too large to display in terminal\n, Please open the file : Mat_Inv_out.txt located in the current folder. To see the output.\n\n");
		
		FILE *fp;


 		fp = fopen("Mat_Inv_out", "w");

 		if (!fp) 
		{
    			fprintf(stderr, "Failed to open Mat_Inv_out.\n");
		    	exit(1);
  		}



		for (int qq = 0; qq < mybatch; qq++)
		{

			if(mybatch==1)
				fprintf(fp, "Input Matrix , M:\n\n");
			else
				fprintf(fp, "Input Matrix %d:\n\n", qq);


	      	
			
	      		for(int i=0; i<n; i++)
      			{
        			for(int j=0; j<n; j++)
				{
					if(inputs[qq][i*n+j])
		            			fprintf(fp,"%12f ",inputs[qq][i*n+j]);
					else
						fprintf(fp,"%11f ",inputs[qq][i*n+j]);
				}
		            		
        			fprintf(fp,"\n");
	      		}
    		}
	    	fprintf(fp,"\n\n");

		for (int qq = 0; qq < mybatch; qq++)
		{
			if(mybatch==1)
				fprintf(fp, "Inverse of the Input Matrix, Inv(M):\n\n");
	      		else
				fprintf(fp, "Inverse %d:\n\n", qq);
	      		for(int i=0; i<n; i++)
	      		{
	        		for(int j=0; j<n; j++)	
				{
					if(results[qq][i*n+j])
		            			fprintf(fp,"%10f ",results[qq][i*n+j]);
					else
		            			fprintf(fp,"%9f ",results[qq][i*n+j]);
	        		
				}

	        		fprintf(fp,"\n");
	      		}
	    	}

		fclose(fp);
			
	}// end of if else condition for output

	float *A, *B;

	A=inputs[0];
	B=results[0];
	mul(A, B, n );

	//mul(inputs[0][], results[0][], n );

}

////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	if(argc!=2)
	{
		printf("Usage: %s <matrix_width>\n", argv[0]);
		return 0;
	}

	int w;
	w = atoi( argv[1] );
	
   	test_invert(w);
    	return 0;
}

/*
$ nvcc -arch=sm_20 -o t540 t540.cu -lcublas
$ ./t540
Input 0:

0.000000        3.000000        4.000000
1.000000        3.000000        10.000000
4.000000        9.000000        16.000000
Input 1:

0.500000        3.000000        4.000000
1.000000        3.000000        10.000000
4.000000        9.000000        16.000000
Input 2:

0.000000        3.000000        4.000000
1.000000        5.000000        6.000000
9.000000        8.000000        2.000000
Input 3:

22.000000       3.000000        4.000000
1.000000        5.000000        6.000000
9.000000        8.000000        2.000000


Inverse 0:

-0.700000       -0.200000       0.300000
0.400000        -0.266667       0.066667
-0.050000       0.200000        -0.050000
Inverse 1:

-1.076923       -0.307692       0.461538
0.615385        -0.205128       -0.025641
-0.076923       0.192308        -0.038462
Inverse 2:

-4.750000       3.250000        -0.250000
6.500000        -4.500000       0.500000
-4.625000       3.375000        -0.375000
Inverse 3:

0.045894        -0.031401       0.002415
-0.062802       -0.009662       0.154589
0.044686        0.179952        -0.129227
$


$ nvcc -arch=sm_20 -o t540 t540.cu -lcublas
$ ./t540 
Enter the order of the aquare matrix :4
Input 0:

-0.100222 -2.553872 -69.072723 0.016120 
-2.752346 -1.230871 1.997387 0.606710 
-0.029929 -0.583444 2.733107 0.254404 
-1.844285 -0.070541 1.906255 10.758991 


Inverse 0:

0.017501	-0.374555	0.713068	0.004234	
-0.056876	-0.005437	-1.457745	0.034861	
-0.012399	0.000729	0.052888	-0.001273	
0.004824	-0.064370	0.103305	0.094125	

*/
