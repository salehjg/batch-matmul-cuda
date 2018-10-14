/*
 ============================================================================
 Name        : batched-matmul-cuda.cu
 Author      : salehjg
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>


#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// C = AB
template <int BLOCK_SIZE>
__global__ void kernel_batched_matmul(
		const float * matA,
		const float * matB,
		float * matC,
		int dim0,
		int dim1A, int dim2A,
		int dim1B, int dim2B,
		int dim1C, int dim2C){
	extern __shared__ float smem[];

	const unsigned int len_subA = BLOCK_SIZE * dim2A, len_subB = BLOCK_SIZE * dim1B; //len of sub matrices of A and B.
	const unsigned long len_A = dim0*dim1A*dim2A, len_B = dim0*dim1B*dim2B;
	const unsigned int BLOCKSIZE_P2 = BLOCK_SIZE*BLOCK_SIZE;

	//smemA = smem + 0;
	//smemB = smem + len_subA;


    // Block index
    unsigned int bx = blockIdx.x;
    unsigned int  by = blockIdx.y;

    // Thread index
    unsigned int  tx = threadIdx.x;
    unsigned int  ty = threadIdx.y;

    unsigned int  c_pos_x, c_pos_y;
    c_pos_x = bx*BLOCK_SIZE + tx;
    c_pos_y = by*BLOCK_SIZE + ty;

    unsigned long gidx1,gidx2;
    unsigned int _d1,_d2;

    //printf("## bx:%u, by:%u, tx:%u, ty:%u, c_pos_x:%u, c_pos_y:%u\n",bx,by,tx,ty,c_pos_x,c_pos_y);


	unsigned long offsetA = (by * BLOCK_SIZE) * dim2A;
	unsigned long offsetB = (bx * BLOCK_SIZE); //first row (d1=0)

	// Load sub matrices from global memory into shared memory

	unsigned long idxA, idxB;
	idxA = ty* BLOCK_SIZE + tx;
	idxB = ty* BLOCK_SIZE + tx;

	//printf("*** bx:%u, by:%u, tx:%u, ty:%u ,idxA:%ld, idxB:%ld\n",bx,by,tx,ty,idxA,idxB);

	while(idxA < len_subA){//Block-stride loop
		gidx1 = offsetA + idxA;
		if(idxA < len_subA && gidx1 < len_A) {
			smem[idxA] = matA[gidx1];
			printf("bx:%u, by:%u, tx:%u, ty:%u ,idxA:%ld, gidx1:%ld\n",bx,by,tx,ty,idxA,gidx1);
		}else{
			smem[idxA] = 0;
		}
		idxA += BLOCKSIZE_P2;
	}

	///TODO: It might be better to store transposed subMatB in shared memory to avoid shared memory read conflict.
	///      But then we might get shared memory write conflict. (?)
	while(idxB < len_subB ){//Block-stride loop
		//gidx2 = offsetB + (bx*BLOCK_SIZE)*dim2B + (idxB % dim2B);
		_d2 = idxB%BLOCK_SIZE;
		_d1 = (idxB/BLOCK_SIZE);
		gidx2 = offsetB + _d1*dim2B + _d2;
		if(idxB < len_subB && gidx2 < len_B && _d1<dim1B && _d2<dim2B){
			smem[len_subA+idxB] = matB[gidx2];
			printf("* bx:%u, by:%u ,tx:%u, ty:%u ,idxB:%ld, _d1:%d, _d2:%d, gidx2:%ld\n",bx,by,tx,ty,idxB,_d1,_d2,gidx2);
		}else{
			smem[len_subA+idxB] = 0;
		}
		idxB += BLOCKSIZE_P2;
	}





	__syncthreads();




    	// Multiply and add each result to produce output element of current thread in the thread block.
    if(c_pos_x<dim2C && c_pos_y<dim1C){
    	unsigned long idx = ty* BLOCK_SIZE + tx;
    	float output_element = 0.0f;

    	//dim2A=dim1B is common equal dimension of 2 matrices  --- block-stride loop
    	for (int k = 0; k < dim2A; k++) {
    		output_element += smem[ty*dim2A+k] * smem[len_subA+ k*BLOCK_SIZE+tx];
    		printf("### c_pos_x:%d, c_pos_y:%d, smem[%d]=%f, smem[%d]=%f\n",
    				c_pos_x,c_pos_y,
    				ty*dim2A+k,smem[ty*dim2A+k],
    				len_subA+ k*BLOCK_SIZE+tx,smem[len_subA+ k*BLOCK_SIZE+tx]);
    	}

    	///TODO: Check matC index to not to exceed the len of matC!
    	matC[c_pos_y*dim2C + c_pos_x] = output_element;

    }


	
}

void batched_matmul(
		const float * matA, //row-major device ptr (batch, hA, wA) == (dim0A,  dim1A  , *dim2A* )
		const float * matB, //row-major device ptr (batch, hB, wB) == (dim0B, *dim1B* ,  dim2B  )
		float * matC,		//row-major device ptr (batch, hB, wB) == (dim0B,  dim1A  ,  dim2B  )
		int dim0A, int dim1A, int dim2A,
		int dim0B, int dim1B, int dim2B){
	if(dim2A != dim1B){printf("ERR@batched_matmul: BAD SHAPE.\n"); return;}
	if(dim0B != dim0A){printf("ERR@batched_matmul: BAD BATCH SIZES.\n"); return;}

	const int BLOCK_DIM = 8;
	dim3 blocksize(BLOCK_DIM,BLOCK_DIM,1);
	dim3 gridsize(0,0,1);
	gridsize.x = (dim2B + BLOCK_DIM-1)/BLOCK_DIM;
	gridsize.y = (dim1A + BLOCK_DIM-1)/BLOCK_DIM;
	unsigned long sharedmemsize = (BLOCK_DIM*dim2A + BLOCK_DIM* dim1B)*sizeof(float);
	printf("@batched_matmul:\n");
	printf("\tBLOCK:(%d, %d)\n",blocksize.x,blocksize.y);
	printf("\t GRID:(%d, %d)\n",gridsize.x,gridsize.y);

	if(BLOCK_DIM==8){
		kernel_batched_matmul<8> <<<gridsize, blocksize, sharedmemsize>>>(
				matA,
				matB,
				matC,
				dim0A,

				dim1A, //hA
				dim2A, //wA

				dim1B, //hA
				dim2B, //wA

				dim1A,
				dim2B);
		CudaCheckError();
	}else{
		printf("ERR@batched_matmul: UNDEFINED BLOCK_DIM.\n"); return;
	}

}

void printTensorContent(float * tn,int dim0,int dim1,int dim2){
    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = dim0;
    _dim1 = dim1;
    _dim2 = dim2;

    if(dim0*dim1*dim2 > 1000){
        _dim0 = 1;
        _dim1 = 1;
    }

    float val;
    unsigned long indx;
    for(int d0=0;d0<_dim0;d0++){
        for(int d1=0;d1<_dim1;d1++){
            for(int d2=0;d2<_dim2;d2++){

                indx = d0*dim1*dim2+
                       d1*dim2+
                       d2;
                val = tn[indx];
                printf("(%d, %d, %d): %f\n",d0,d1,d2,val);// "("<<d0<<", "<<d1<<", "<<d2<<")" << ": "<< val<<endl;
            }}}
}

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}


// rslt = MAT1 * MAT2
// everything is row-major, so matrixH means dim1 and matrixW means dim2
float* LA_MatMul(float* mat1,float* mat2,
                 int batchsize, int matrix_rank,
                 int matrixH1,int matrixW1,
                 int matrixH2,int matrixW2){
    if(matrix_rank!=3){printf("LA_MATMUL: invalid matrix rank\n");return nullptr;}
    if(matrixW1!=matrixH2){printf("LA_MATMUL: bad shapes\n");return nullptr;}
    float* rslt = (float*)malloc(batchsize*matrixH1*matrixW2*sizeof(float));
    int indxS1=0;
    int indxS2=0;
    int indxD=0;

    for(int b=0;b<batchsize;b++) {
        // for element of output of matrixH1 x matrixW2
        for(int j=0;j<matrixH1;j++){
            for(int i=0;i<matrixW2;i++){
                //mat1: select row j
                //mat2: select col i
                float sum=0;
                for(int mat1_x=0;mat1_x<matrixW1;mat1_x++)
                {
                    indxS1 = b*matrixH1*matrixW1 +
                             j*matrixW1 + mat1_x;
                    /*indxS2 = b*matrixH2*matrixW2 +
                             mat1_x*matrixW1 + j;*/
                    indxS2 = b*matrixH2*matrixW2 +
                             mat1_x*matrixW2 + i;

                    sum += mat1[indxS1] * mat2[indxS2];
                }
                // for element of output of matrixH1 x matrixW2
                indxD = b*matrixH1*matrixW2 +
                        j*matrixW2 + i;
                rslt[indxD] = sum;

            }
        }
    }
    return rslt;
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int dim0, int dim1A, int dim2A, int dim1B, int dim2B) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dim0 * dim1A * dim2A;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = reinterpret_cast<float *>(malloc(mem_size_A));

    unsigned int size_B = dim0 * dim1B * dim2B;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = reinterpret_cast<float *>(malloc(mem_size_B));

    // Initialize host memory
    const float valB = 0.5f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    unsigned int size_C = dim1A * dim2B;
    unsigned int mem_size_C = size_C * sizeof(float);
    float *h_C = reinterpret_cast<float *>(malloc(mem_size_C));
    float *host_results = LA_MatMul(h_A, h_B,dim0,3, dim1A,dim2A, dim1B,dim2B);

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    CudaSafeCall(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));

    CudaSafeCall(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));

    CudaSafeCall(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // copy host memory to device
    CudaSafeCall(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));


    printf("\n** LEN_A: %ld\n",size_A);
	printf("** LEN_B: %ld\n",size_B);
	printf("** LEN_C: %ld\n\n",size_C);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
    //batched_matmul(d_A, d_B, d_C,   dim0,dim1A,dim2A,   dim0,dim1B,dim2B);

    printf("done\n");

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    CudaSafeCall(cudaEventCreate(&start));

    cudaEvent_t stop;
    CudaSafeCall(cudaEventCreate(&stop));

    // Record the start event
    CudaSafeCall(cudaEventRecord(start, NULL));

    // Execute the kernel
    int nIter = 1;

    for (int j = 0; j < nIter; j++) {
    	batched_matmul(d_A, d_B, d_C,   dim0,dim1A,dim2A,   dim0,dim1B,dim2B);
    }

    // Record the stop event
    CudaSafeCall(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    CudaSafeCall(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CudaSafeCall(cudaEventElapsedTime(&msecTotal, start, stop));

    printf("Time= %.3f msec\n", msecTotal / nIter);

    // Copy result from device to host
    CudaSafeCall(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    printf("\n\nMatA:\n");
    printTensorContent(h_A,dim0,dim1A,dim2A);
    printf("\n\nMatB:\n");
    printTensorContent(h_B,dim0,dim1B,dim2B);
    printf("\n\nMatC:\n");
    printTensorContent(host_results,dim0,dim1A,dim2B);

    printf("\n\nChecking computed result for correctness: \n");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero

    for (int i = 0; i < static_cast<int>(size_C); i++) {
        double abs_err = fabs(h_C[i] - (host_results[i]));
        double dot_length = dim2A;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f,\t\tref=%.8f error term is > %E\n",
                   i, h_C[i], host_results[i], eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    CudaSafeCall(cudaFree(d_A));
    CudaSafeCall(cudaFree(d_B));
    CudaSafeCall(cudaFree(d_C));

    if (correct) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    // MatC = MatA * MatB
    // Everything is row-major, so dim2 is width of matrix and dim1 is height of it.
    int batchsize = 1;
    int dim1A = 8;       	int dim2A = 2;
    int dim1B = 2;       	int dim2B = 8;

    printf("MatrixA(dim0:%d, dim1: %d, dim2:%d)\nMatrixB(dim0:%d, dim1: %d, dim2:%d)\n", batchsize,dim1A,dim2A,batchsize,dim1B,dim2B);

    int matrix_result = MatrixMultiply(batchsize, dim1A, dim2A, dim1B, dim2B);

    exit(matrix_result);
}

