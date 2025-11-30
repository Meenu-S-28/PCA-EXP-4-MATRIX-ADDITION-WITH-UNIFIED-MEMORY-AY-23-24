# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24

<h3>Name: Meenu S</h3>
<h3>Register Number: 212223230124</h3>

<h1> <align=center> MATRIX ADDITION WITH UNIFIED MEMORY </h3>
  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:
To perform Matrix addition with unified memory and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Setup Device and Properties
Initialize the CUDA device and get device properties.
2.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory
3.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
4.	Initialize Data on Host
5.	Generate random floating-point data for matrices A and B using the initialData function.
6.	Measure the time taken for initialization.
7.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.
8.	Measure the time taken for matrix addition on the host.
9.	Invoke Kernel
10.	Define grid and block dimensions for the CUDA kernel launch.
11.	Warm-up the kernel with a dummy launch for unified memory page migration.
12.	Measure GPU Execution Time
13.	Launch the CUDA kernel to compute the matrix sum on the GPU.
14.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.
15.	Check for Kernel Errors
16.	Check for any errors that occurred during the kernel launch.
17.	Verify Results
18.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.
19.	Free Allocated Memory
20.	Free memory allocated on the device using cudaFree.
21.	Reset Device and Exit
22.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:
```c
%%cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

__global__ void addMatrix(float *A, float *B, float *C, int N) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * N + ix;
    
    if (ix < N && iy < N)
        C[idx] = A[idx] + B[idx];
}

int main() {
    int N = 1 << 10;  // 1024x1024
    size_t bytes = N * N * sizeof(float);
    
    float *A, *B, *C1, *C2;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C1, bytes);
    cudaMallocManaged(&C2, bytes);
    
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    
    // WITH memset - zeros the memory first
    memset(C1, 0, bytes);
    
    // WITHOUT memset - uses uninitialized memory
    // C2 is left uninitialized
    
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    
    printf("=== WITH memset() ===\n");
    addMatrix<<<grid, block>>>(A, B, C1, N);
    cudaDeviceSynchronize();
    printf("C1[0] = %.0f (expected 3)\n", C1[0]);
    printf("C1[100] = %.0f (expected 3)\n\n", C1[100]);
    
    printf("=== WITHOUT memset() ===\n");
    addMatrix<<<grid, block>>>(A, B, C2, N);
    cudaDeviceSynchronize();
    printf("C2[0] = %.0f (expected 3)\n", C2[0]);
    printf("C2[100] = %.0f (expected 3)\n", C2[100]);
    
    printf("\nBoth work! memset() just ensures clean initial state.\n");
    
    cudaFree(A); cudaFree(B); cudaFree(C1); cudaFree(C2);
    
    return 0;
}
```

## OUTPUT:
#### with memset()
<img width="1584" height="148" alt="image" src="https://github.com/user-attachments/assets/3abcfb13-3868-450f-adf6-55672ffd1c23" />

#### without memset()
<img width="1627" height="151" alt="image" src="https://github.com/user-attachments/assets/c685aa9f-c507-4113-85cd-2eb1b13701df" />

## RESULT:
Thus the program has been executed by using unified memory. It is observed that removing memset function has given less/more_______________time.
