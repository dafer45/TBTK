/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file Diagonalizer.cu
 *
 *  @author Andreas Theiler
 */

#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/GPUResourceManager.h"

#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "cusolver_utils.h"

using namespace std;

namespace TBTK{
namespace Solver{

void Diagonalizer::solveGPU(){
    //Initialize device
    int device = GPUResourceManager::getInstance().allocateDevice();
	TBTKAssert(
		cudaSetDevice(device) == cudaSuccess,
		"Diagonalizer::solveGPU()",
		"CUDA set device error for device " << device << ".",
		""
	);

    cudaStream_t stream = NULL;
    TBTKAssert(
        cudaStreamCreateWithFlags(
            &stream, 
            cudaStreamNonBlocking
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "Failed to set up stream on device.",
        ""
    )
    
    //Create handle for cuSolver and set stream
    cusolverDnHandle_t cusolverHandle = NULL;
    TBTKAssert(
        cusolverDnCreate(
            &cusolverHandle
        ) == CUSOLVER_STATUS_SUCCESS,
        "Diagonalizer::solveGPU()",
        "CUDA error in cusolverDnXsyevd_bufferSize.",
        ""
    )
    TBTKAssert(
        cusolverDnSetStream(
            cusolverHandle,
             stream
            ) == CUSOLVER_STATUS_SUCCESS,
        "Diagonalizer::solveGPU()",
        "CUDA error setting up stream for cusolver.",
        ""
    ) 

    //Allocate memory on device for hamiltonian and corresponding output
    int n = getModel().getBasisSize();	//...nxn-matrix.
    complex<double> *hamiltonian_device;
    double *eigenValues_device;
    int *info_device = nullptr;

    TBTKAssert(
        cudaMalloc(
            reinterpret_cast<void **>(&hamiltonian_device), 
            sizeof(complex<double>) * hamiltonian.getSize()
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error allocating memory on device.",
        ""
    )
    TBTKAssert(
        cudaMalloc(
            reinterpret_cast<void **>(&eigenValues_device),
             sizeof(double) * n
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error allocating memory on device.",
        ""
    )
    TBTKAssert(
        cudaMalloc(
            reinterpret_cast<void **>(&info_device),
             sizeof(int)
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error allocating memory on device.",
        ""
    )

    //Copy hamiltonian to device
    TBTKAssert(
        cudaMemcpyAsync(
            hamiltonian_device, 
            hamiltonian.getData(), 
            sizeof(complex<double>) * hamiltonian.getSize(), 
            cudaMemcpyHostToDevice,
            stream) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error copying to memory on device.",
        ""
    )

    //Set up the cusolver routine
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;		//...eigenvalues and eigenvectors...
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;		//...for an upper triangular Matrix...

    void *buffer_device = nullptr; // Device buffer memory
    void *buffer_host = nullptr;  //Host buffer memory
    size_t sizeBuffer_device = 0; //Size of buffer memory needed on device
    size_t sizeBuffer_host = 0; //Size of buffer memory needed on host

    int info;
    
    //Check if buffer is needed and allocate accordingly
    TBTKAssert(
        cusolverDnXsyevd_bufferSize(
            cusolverHandle, 
            NULL, 
            jobz, 
            uplo, 
            n, 
            CUDA_C_64F, //Complex double input matrix
            hamiltonian_device,
            n,
            CUDA_R_64F,
            eigenValues_device, 
            CUDA_C_64F,
            &sizeBuffer_device,
            &sizeBuffer_host
        ) == CUSOLVER_STATUS_SUCCESS,
        "Diagonalizer::solveGPU()",
        "CUDA error in cusolverDnXsyevd_bufferSize.",
        ""
    )

    
    TBTKAssert(
        cudaMalloc(reinterpret_cast<void **>(&buffer_device), 
        sizeof(complex<double>) * sizeBuffer_device
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "Failed to allocate buffer memory on device.",
        "" 
    )
    buffer_host = malloc(sizeof(complex<double>) * sizeBuffer_host);

    //Run the diagonalization routine
    

    // TBTKAssert(
    //     cusolverDnXsyevd(
    //         cusolverHandle, 
    //         NULL, 
    //         jobz, 
    //         uplo,
    //         n, 
    //         CUDA_C_64F,
    //         hamiltonian_device,
    //         n,
    //         CUDA_R_64F,
    //         eigenValues_device,
    //         CUDA_C_64F,
    //         buffer_device,
    //         sizeBuffer_device,
    //         buffer_host,
    //         sizeBuffer_host,
    //         info_device
    //     ) == CUSOLVER_STATUS_SUCCESS,
    //     "Diagonalizer::solveGPU()",
    //     "CUDA error in cusolverDnXsyevd.",
    //     "" 
    // )

    int nrEigenValues = 0;
    CUSOLVER_CHECK( cusolverDnXsyevd(
        cusolverHandle, 
        NULL, 
        jobz, 
        uplo,
        n, 
        CUDA_C_64F,
        hamiltonian_device,
        n,
        CUDA_R_64F,
        eigenValues_device,
        CUDA_C_64F,
        buffer_device,
        sizeBuffer_device,
        buffer_host,
        sizeBuffer_host,
        info_device
    ));

    TBTKAssert(
        cudaMemcpyAsync(
            &info,
            info_device,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error copying to memory from device.",
        ""
    )
    TBTKAssert(
        info == 0,
        "Diagonalizer:solve()",
        "Diagonalization routine cusolverDnXsyevd exited with INFO=" + to_string(info) + ".",
        "See CUDA documentation for cusolverDnXsyevd for further information."
    );

    TBTKAssert(
        cudaMemcpyAsync(
            getEigenVectorsRW().getData(),
            hamiltonian_device,
            sizeof(complex<double>)*hamiltonian.getSize(),
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error copying to memory from device.",
        ""
    )

    TBTKAssert(
        cudaMemcpyAsync(
            getEigenValuesRW().getData(),
            eigenValues_device,
            sizeof(double)*n,
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error copying to memory from device.",
        ""
    )

    TBTKAssert(
        cudaStreamSynchronize(
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error while synchronizing stream.",
        ""
    )

    // Free device resources
    TBTKAssert(
        cudaFree(
            hamiltonian_device
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error freeing device memory.",
        ""
    )
    TBTKAssert(
        cudaFree(
            eigenValues_device
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error freeing device memory.",
        ""
    )
    TBTKAssert(
        cudaFree(
            info_device
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error freeing device memory.",
        ""
    )
    TBTKAssert(
        cudaFree(
            buffer_device
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error freeing device memory.",
        ""
    )
    TBTKAssert(
        cusolverDnDestroy(
            cusolverHandle
        ) == CUSOLVER_STATUS_SUCCESS,
        "Diagonalizer::solveGPU()",
        "CUDA error destroying cusolver handle.",
        ""
    )
    TBTKAssert(
        cudaStreamDestroy(
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error destroying cuda stream.",
        ""
    )
	GPUResourceManager::getInstance().freeDevice(device);

    free(buffer_host);
    buffer_host = nullptr;
}

};	//End of namespace Solver
};	//End of namespace TBTK