#include "cuda.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

texture<float, 2, cudaReadModeElementType> texA;



#ifndef min

 #define min(a,b) (((a) < (b)) ? (a) : (b))

#endif


#define MAX_PITCH_VALUE_IN_BYTES       262144
#define MAX_TEXTURE_WIDTH_IN_BYTES     65536
#define MAX_TEXTURE_HEIGHT_IN_BYTES    32768
#define MAX_PART_OF_FREE_MEMORY_USED   0.9

__global__ void cuParallelSqrt(float *dist, int width, int pitch, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}

__global__ void cuComputeDistanceGlobal( float* A, int wA, int pA, float* B, int wB, int pB, int dim,  float* AB){

	// Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
	__shared__ float shared_A[16][16];
	__shared__ float shared_B[16][16];
    

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	
	// Other variables
	float tmp;
    float ssd = 0;
	
    // Looping parameters
    begin_A = 16 * blockIdx.y;
    begin_B = 16 * blockIdx.x;
    step_A  = 16 * pA;
    step_B  = 16 * pB;
    end_A   = begin_A + (dim-1) * pA;

    // Conditions
	int cond0 = (begin_A + tx < wA); // used to write in shared memory
    int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
    int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix


    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
        
        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/pA + ty < dim){
            shared_A[ty][tx] = (cond0)? A[a + pA * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? B[b + pB * ty + tx] : 0;
        }
        else{
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1){
            for (int k = 0; k < BLOCK_DIM; ++k){
				tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
			}
        }
        
        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1)
        AB[ (begin_A + ty) * pB + begin_B + tx ] = ssd;

}


void knn(float* ref_host, int ref_width, float* query_host, int query_width, int height, int k, float* dist_host, int* ind_host){ 

    unsigned int size_of_float = sizeof(float);
    unsigned int size_of_int   = sizeof(int);

    //Device variables
    float        *query_dev;
    float        *ref_dev;
    float        *dist_dev;
    int          *ind_dev;
    cudaArray    *ref_array;
    cudaError_t  result;

    //Sizes
    size_t query_pitch;
    size_t query_pitch_in_bytes;
    size_t ref_pitch;
    size_t ref_pitch_in_bytes;
    size_t ind_pitch;
    size_t ind_pitch_in_bytes;
    size_t max_nb_query_traited;
    size_t actual_nb_query_width;
    size_t memory_total;
    size_t memory_free;

    unsigned int use_texture = ( ref_width*size_of_float<=MAX_TEXTURE_WIDTH_IN_BYTES && height*size_of_float<=MAX_TEXTURE_HEIGHT_IN_BYTES );

    // CUDA Initialisation
    cuInit(0);
    
    // Check free memory using driver API ; only (MAX_PART_OF_FREE_MEMORY_USED*100)% of memory will be used
    CUcontext cuContext;
    CUdevice  cuDevice=0;
    cuCtxCreate(&cuContext, 0, cuDevice);
    cuMemGetInfo(&memory_free, &memory_total);
    printf("Memory :: Free : %d , Total : %d",memory_free,memory_total);
    cuCtxDetach (cuContext);

    max_nb_query_traited = ( memory_free * MAX_PART_OF_FREE_MEMORY_USED - size_of_float * ref_width*height ) / ( size_of_float * (height + ref_width) + size_of_int * k);
    max_nb_query_traited = min( query_width, (max_nb_query_traited / 16) * 16 );

	// Allocation of global memory for query points and for distances
    result = cudaMallocPitch( (void **) &query_dev, &query_pitch_in_bytes, max_nb_query_traited * size_of_float, height + ref_width);
    if (result){
        printf("\n%d\n",max_nb_query_traited*size_of_float*(height+ref_width));
        return;
    }

    printf("\nQuery pitch : %d\n", query_pitch_in_bytes/sizeof(float));
	query_pitch = query_pitch_in_bytes/size_of_float;
    dist_dev    = query_dev + height * query_pitch;

    result = cudaMallocPitch( (void **) &ind_dev, &ind_pitch_in_bytes, max_nb_query_traited * size_of_int, k);
	if (result){
        cudaFree(query_dev);
        printErrorMessage(result, max_nb_query_traited*size_of_int*k);
        return;
    }

    ind_pitch = ind_pitch_in_bytes/size_of_int;
    if (use_texture){
	
        // Allocation of texture memory
        cudaChannelFormatDesc channelDescA = cudaCreateChannelDesc<float>();
        result = cudaMallocArray( &ref_array, &channelDescA, ref_width, height );
        if (result){
            printErrorMessage(result, ref_width*height*size_of_float);
            cudaFree(ind_dev);
            cudaFree(query_dev);
            return;
        }
        cudaMemcpyToArray( ref_array, 0, 0, ref_host, ref_width * height * size_of_float, cudaMemcpyHostToDevice );
        
        // Set texture parameters and bind texture to array
        texA.addressMode[0] = cudaAddressModeClamp;
        texA.addressMode[1] = cudaAddressModeClamp;
        texA.filterMode     = cudaFilterModePoint;
        texA.normalized     = 0;
        cudaBindTextureToArray(texA, ref_array);
		
    }
    else{
	
		// Allocation of global memory
        result = cudaMallocPitch( (void **) &ref_dev, &ref_pitch_in_bytes, ref_width * size_of_float, height);
        if (result){
            printErrorMessage(result,  ref_width*size_of_float*height);
            cudaFree(ind_dev);
            cudaFree(query_dev);
            return;
        }
        ref_pitch = ref_pitch_in_bytes/size_of_float;
        cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_host, ref_width*size_of_float,  ref_width*size_of_float, height, cudaMemcpyHostToDevice);
    }

    // Split queries to fit in GPU memory
    for (int i=0; i<query_width; i+=max_nb_query_traited){
        
		// Number of query points considered
        actual_nb_query_width = min( max_nb_query_traited, query_width-i );
        
        // Copy of part of query actually being treated
        cudaMemcpy2D(query_dev, query_pitch_in_bytes, &query_host[i], query_width*size_of_float, query_width*size_of_float, height, cudaMemcpyHostToDevice);

        //Block size for computing distances
        dim3 g_16x16(actual_nb_query_width/16, ref_width/16, 1);
        dim3 t_16x16(16, 16, 1);

        if (actual_nb_query_width%16 != 0) g_16x16.x += 1;
        if (ref_width  %16 != 0) g_16x16.y += 1;

        //Block size for sorting distances
        dim3 g_256x1(actual_nb_query_width/256, 1, 1);
        dim3 t_256x1(256, 1, 1);
        if (actual_nb_query_width%256 != 0) g_256x1.x += 1;

        //Computing square root of k elements
        dim3 g_k_16x16(actual_nb_query_width/16, k/16, 1);
        dim3 t_k_16x16(16, 16, 1);
        if (actual_nb_query_width%16 != 0) g_k_16x16.x += 1;
        if (k  %16 != 0) g_k_16x16.y += 1;

    }
}

int main(void){
	
    // Variables and parameters
    float* ref;                 // Pointer to reference point array
    float* query;               // Pointer to query point array
    float* dist;                // Pointer to distance array
	int*   ind;                 // Pointer to index array
	int    ref_nb     = 65535;   // Reference point number, max=65535
	int    query_nb   = 65535;   // Query point number,     max=65535
	int    dim        = 32;     // Dimension of points
	int    k          = 20;     // Nearest neighbors to consider
	int    iterations = 100;
	int    i;

	ref    = (float *) malloc(ref_nb   * dim * sizeof(float));
	query  = (float *) malloc(query_nb * dim * sizeof(float));
	dist   = (float *) malloc(query_nb * k * sizeof(float));
	ind    = (int *)   malloc(query_nb * k * sizeof(float));

	// Fill random values 
	srand(time(NULL));
	for (i=0 ; i<ref_nb   * dim ; i++) ref[i]    = (float)rand() / (float)RAND_MAX;
	for (i=0 ; i<query_nb * dim ; i++) query[i]  = (float)rand() / (float)RAND_MAX;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;

	cudaEventRecord(start, 0);	

	for (i=0; i<iterations; i++)
		knn(ref, ref_nb, query, query_nb, dim, k, dist, ind);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	//printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time/1000, iterations, elapsed_time/(iterations*1000));
	
	// Destroy cuda event object and free memory
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(ind);
	free(dist);
	free(query);
	free(ref);

}
