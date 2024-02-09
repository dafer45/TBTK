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
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
 */

#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/GPUResourceManager.h"

#include <vector>
#include <type_traits>
#include <string>

#include <cusolverDn.h>
#include <cusolverMg.h>
#include <cuda_runtime.h>
#include "cusolverMg_utils.h"

using namespace std;

namespace TBTK{
namespace Solver{

void Diagonalizer::solveMultiGPU(complex<double>* matrix, double* eigenValues, const int &n){
    const cudaDataType_t computeType = CUDA_C_64F; // At the moment only complex double is supported
    typedef complex<double> data_type; // Could also be double
    int numDevices = 0;
    TBTKAssert(
		cudaGetDeviceCount(&numDevices) == cudaSuccess,
		"Diagonalizer::solveMultiGPU()",
		"Error in cudaGetDeviceCount().",
		""
	);
    
    if(numDevices <= 1){
        Streams::out << "Detect 1 gpu or less, falling back to single GPU operation." << endl;
        useMultiGPUAcceleration = false;
        solveGPU(matrix, eigenValues,n);
    }
    // set up various calculation parameters and resources
    const int IA = 1;
    const int JA = 1;
    const int T_A = 256; // tile size recommended value is 256 or 512
    const int lda = n;

    //Allocate available devices
    static std::vector<int> deviceList(numDevices);
    const int deviceCount = numDevices;
    
    for(int i = 0; i < deviceCount; i++) {
        if(!GPUResourceManager::getInstance().getDeviceBusy(i)){
            deviceList[i] = GPUResourceManager::getInstance().allocateDevice();
        }
        else{
            //TODO 
            Streams::err << "Not all GPUs are available" << endl;
            numDevices--;
        }
        
    }

    //TODO used for workaround of memory leak in cuSolverMg
    //TODO All static variables can be removed once bug fix from nvidia comes back, same for all  if(!isInitialized)
    static bool isInitialized = false;
    if(!isInitialized){
        GPUResourceManager::getInstance().enableP2PAccess();
    }
    
    
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    if(calculationMode == CalculationMode::EigenValues){
        jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    }

    cudaLibMgMatrixDesc_t descrMatrix;
    static cudaLibMgGrid_t gridA;
    static cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    static cusolverMgHandle_t cusolverHandle = NULL;
    if(!isInitialized){
        TBTKAssert(
            cusolverMgCreate(&cusolverHandle) == CUSOLVER_STATUS_SUCCESS,
            "Diagonalizer::solveMultiGPU()",
            "Error in cusolverMgCreate().",
            ""
        );
        TBTKAssert(
            cusolverMgDeviceSelect(cusolverHandle, numDevices, deviceList.data())
                        == CUSOLVER_STATUS_SUCCESS,
            "Diagonalizer::solveMultiGPU()",
            "Error in cusolverMgDeviceSelect().",
            ""
        );
        TBTKAssert(
            cusolverMgCreateDeviceGrid(&gridA, 1, numDevices, deviceList.data(), mapping)
                        == CUSOLVER_STATUS_SUCCESS,
            "Diagonalizer::solveMultiGPU()",
            "Error in cusolverMgCreateDeviceGrid().",
            ""
        );
    }
    TBTKAssert(
        cusolverMgCreateMatrixDesc(&descrMatrix, n, // nubmer of rows of matrix
            n,          // number of columns of matrix
            n,          // number or rows in a tile
            T_A,        // number of columns in a tile
            computeType, gridA)
        == CUSOLVER_STATUS_SUCCESS,
        "Diagonalizer::solveMultiGPU()",
        "Error in cusolverMgCreateMatrixDesc().",
        ""
    );
    isInitialized = true;

    vector<data_type *> array_d_A(numDevices, nullptr);

    // Allocate resource for local matrices on the devices
    // createEmptyMatrix(numDevices, 
    //                     deviceList.data(),
    //                     n,                  // number of columns of global matrix
    //                     T_A,                // number of columns per column tile 
    //                     lda,                // leading dimension of local matrix
    //                     array_d_A.data());
    createMat<data_type>(numDevices, deviceList.data(), n, /* number of columns of global A */
    T_A,                          /* number of columns per column tile */
    lda,                          /* leading dimension of local A */
    array_d_A.data());
    memcpyH2D<data_type>(numDevices, deviceList.data(), n, n,
                         /* input */
                         matrix,
                         lda,
                         /* output */
                         n,                /* number of columns of global A */
                         T_A,              /* number of columns per column tile */
                         lda,              /* leading dimension of local A */
                         array_d_A.data(), /* host pointer array of dimension nbGpus */
                         IA, JA);
    // Allocate buffer memory
    int64_t lwork = 0;
    TBTKAssert(
        cusolverMgSyevd_bufferSize(
            cusolverHandle, jobz, CUBLAS_FILL_MODE_LOWER, // Only lower supported according to documentation
            n, reinterpret_cast<void **>(array_d_A.data()), IA,
            JA,                                                     
            descrMatrix, reinterpret_cast<void *>(eigenValues), CUDA_R_64F,
            computeType, &lwork)
         == CUSOLVER_STATUS_SUCCESS,
		"Diagonalizer::solveMultiGPU()",
		"Error in cusolverMgSyevd_bufferSize().",
		""
	);
    std::vector<data_type *> array_d_work(numDevices, nullptr);

    // array_d_work[i] points to device workspace of device i
    workspaceAlloc(numDevices, deviceList.data(),
                    sizeof(data_type) * lwork,
                    reinterpret_cast<void **>(array_d_work.data()));
    // sync all devices before calculation
    TBTKAssert(
        cudaDeviceSynchronize() == cudaSuccess,
		"Diagonalizer::solveMultiGPU()",
		"Error in cudaDeviceSynchronize().",
		""
	);
    // Run the eigenvalue solver
    int info = 0;
    cusolverStatus_t error;
    error =
    cusolverMgSyevd(
        cusolverHandle, jobz, CUBLAS_FILL_MODE_LOWER,
        n, reinterpret_cast<void **>(array_d_A.data()),
        IA, JA, descrMatrix, reinterpret_cast<void **>(eigenValues),
        CUDA_R_64F, computeType,
        reinterpret_cast<void **>(array_d_work.data()), lwork, &info
        );
    string errorString;
    // More fine grained error handling for this
    if(error != CUSOLVER_STATUS_SUCCESS){
        switch(error){
            case CUSOLVER_STATUS_INVALID_VALUE:
                errorString = "CUSOLVER_STATUS_INVALID_VALUE";
                break;
            case CUSOLVER_STATUS_INTERNAL_ERROR:
                errorString = "CUSOLVER_STATUS_INTERNAL_ERROR";
                break;
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                errorString = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                break;
            case CUSOLVER_STATUS_NOT_INITIALIZED:
                errorString = "CUSOLVER_STATUS_NOT_INITIALIZED";
                break;
            case CUSOLVER_STATUS_ALLOC_FAILED:
                errorString = "CUSOLVER_STATUS_ALLOC_FAILED";
                break;
            case CUSOLVER_STATUS_ARCH_MISMATCH:
                errorString = "CUSOLVER_STATUS_ARCH_MISMATCH";
                break;
            case CUSOLVER_STATUS_EXECUTION_FAILED:
                errorString = "CUSOLVER_STATUS_EXECUTION_FAILED";
                break;
            default:
                errorString = "Unknown cusolverStatus_t error: " + error;            
        }

    }
    TBTKAssert(
        error
          == CUSOLVER_STATUS_SUCCESS,
		"Diagonalizer::solveMultiGPU()",
		"Error in cusolverMgSyevd().",
		"cusolverStatus_t error: " << errorString << ". Error info value: " << info
	);
    TBTKAssert(
        info == 0,
		"Diagonalizer::solveMultiGPU()",
		"Error in cusolverMgSyevd().",
		"For more information see Nvidia documentation regarding info error of: " << info
	);
    // sync all devices after calculation
    TBTKAssert(
        cudaDeviceSynchronize() == cudaSuccess,
		"Diagonalizer::solveMultiGPU()",
		"Error in cudaDeviceSynchronize().",
		""
	);
    // Copy data to host from devices
    memcpyD2H<data_type>(numDevices, deviceList.data(), n, n,
                         n, 
                         T_A,
                         lda,
                         array_d_A.data(), IA, JA,
                         matrix,
                         lda);

    // sync all devices before clean up
    TBTKAssert(
        cudaDeviceSynchronize() == cudaSuccess,
		"Diagonalizer::solveMultiGPU()",
		"Error in cudaDeviceSynchronize().",
		""
	);

    // Clean up
    destroyMat(numDevices, deviceList.data(), n,
                T_A,
                reinterpret_cast<void **>(array_d_A.data()));
          
    workspaceFree(numDevices, deviceList.data(), reinterpret_cast<void **>(array_d_work.data()));


    //TODO uncomment after bug fix in CUDA library
    // TBTKAssert(
    //     cusolverMgDestroyGrid(gridA)
    //      == CUSOLVER_STATUS_SUCCESS,
	// 	"Diagonalizer::cusolverMgDestroyGrid()",
	// 	"Error in cusolverMgDestroyGrid().",
	// 	""
	// );
    // gridA = NULL;
    TBTKAssert(
        cusolverMgDestroyMatrixDesc(descrMatrix)
         == CUSOLVER_STATUS_SUCCESS,
		"Diagonalizer::solveMultiGPU()",
		"Error in cusolverMgDestroyMatrixDesc().",
		""
	);
    descrMatrix = NULL;
    // // TODO this crashes the function if executed twice???
    // TBTKAssert(
    //     cusolverMgDestroy(cusolverHandle)
    //      == CUSOLVER_STATUS_SUCCESS,
	// 	"Diagonalizer::solveMultiGPU()",
	// 	"Error in cusolverMgDestroy().",
	// 	""
	// );
    // cusolverHandle = NULL;
	TBTKAssert(
		cudaDeviceSynchronize() == cudaSuccess,
		"Diagonalizer::solveMultiGPU()",
		"Error in cudaDeviceSynchronize().",
		""
	);
    //Workaround to avoid memory leak caused by the cuSolverMg library
    // TODO remove if memory leak is fixed
	// int currentDevice = 0;
	// TBTKAssert(
	// 	cudaGetDevice(&currentDevice) == cudaSuccess,
	// 	"Diagonalizer::solveMultiGPU()",
	// 	"Error in cudaGetDevice().",
	// 	""
	// );
    // for(int i = 0; i < deviceCount; i++){
    //     if(GPUResourceManager::getInstance().getDeviceBusy(i)){
	// 		TBTKAssert(
	// 			cudaSetDevice(deviceList[i]) == cudaSuccess,
	// 			"Diagonalizer::solveMultiGPU()",
	// 			"Error in cudaSetDevice().",
	// 			""
	// 		);
	// 		TBTKAssert(
	// 			cudaDeviceReset() == cudaSuccess,
	// 			"Diagonalizer::solveMultiGPU()",
	// 			"Error in cudaResetDevice().",
	// 			""
	// 		);
	// 		TBTKAssert(
	// 			cudaSetDevice(currentDevice) == cudaSuccess,
	// 			"Diagonalizer::solveMultiGPU()",
	// 			"Error in cudaSetDevice().",
	// 			""
	// 		);
    //     }
    // }
    // // -------- End of workaround

    for(int i = 0; i < deviceCount; i++){
        if(GPUResourceManager::getInstance().getDeviceBusy(i)){
            GPUResourceManager::getInstance().freeDevice(deviceList[i]);
        }
    }
	if(calculationMode == CalculationMode::EigenValuesAndEigenVectors){
		eigenVectorsAvailable = true;
	}
}

void Diagonalizer::solveGPU(complex<double>* matrix, double* eigenValues, const int &n){
    GPUResourceManager::getInstance().setVerbose(getVerbose());
    if(useMultiGPUAcceleration){
        solveMultiGPU(matrix, eigenValues, n);
        return;
    }
    cudaDataType_t computeType = CUDA_C_64F; // At the moment only complex double is supported
    typedef complex<double> data_type;
    //Initialize device
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    int device;
    if(numDevices > 1){
        device = GPUResourceManager::getInstance().allocateDevice();
    }
    else{
        device = 0;
    }
    //Run in parallel on one gpu instead
    
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

    //Print device memory use
    if(getGlobalVerbose() && getVerbose()){
            size_t deviceMemorySize = n*n*sizeof(
                data_type //Matrix size
            );
            deviceMemorySize += sizeof(double) * n; //Eigenvalues
            deviceMemorySize += sizeof(int); //Info device

            Streams::out << "\tDevice memory use for input: ";
            if(deviceMemorySize < 1024){
                Streams::out << deviceMemorySize
                    << "B\n";
            }
            else if(deviceMemorySize < 1024*1024){
                Streams::out << deviceMemorySize/1024
                << "KB\n";
            }
            else if(deviceMemorySize < 1024*1024*1024){
                Streams::out << deviceMemorySize/1024/1024
                 << "MB\n";
            }
            else{
                Streams::out << deviceMemorySize/1024/1024/1024
                    << "GB\n";
            }
        }

    //Allocate memory on device for hamiltonian and corresponding output
    data_type *hamiltonian_device;
    double *eigenValues_device;
    int *info_device = nullptr;

    TBTKAssert(
        cudaMallocAsync(
            reinterpret_cast<void **>(&hamiltonian_device), 
            sizeof(data_type) * n*n,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error allocating device memory.",
        ""
    ) 
    TBTKAssert(
        cudaMallocAsync(
            reinterpret_cast<void **>(&eigenValues_device),
            sizeof(double) * n,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error allocating memory on device.",
        ""
    ) 

    TBTKAssert(
        cudaMallocAsync(
            reinterpret_cast<void **>(&info_device),
             sizeof(int),
             stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error allocating memory on device.",
        ""
    )
    //Copy hamiltonian to device
    cudaStreamSynchronize(stream);
    TBTKAssert(
        cudaMemcpyAsync(
            hamiltonian_device, 
            matrix, 
            sizeof(data_type) * n*n, 
            cudaMemcpyHostToDevice,
            stream) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error copying to memory on device.",
        ""
    )


    //Set up the cusolver routine
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; //...eigenvalues and eigenvectors...
    if(calculationMode == CalculationMode::EigenValues){
        jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    }		
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;		//...for an lower triangular Matrix...

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
            computeType, //Complex double input matrix
            hamiltonian_device,
            n,
            CUDA_R_64F,
            eigenValues_device, 
            computeType,
            &sizeBuffer_device,
            &sizeBuffer_host
        ) == CUSOLVER_STATUS_SUCCESS,
        "Diagonalizer::solveGPU()",
        "CUDA error in cusolverDnXsyevd_bufferSize.",
        ""
    )

    //Print device memory use
    if(getGlobalVerbose() && getVerbose()){
        size_t deviceMemorySize = sizeBuffer_device;

        Streams::out << "\tDevice memory use for buffer: ";
        if(deviceMemorySize < 1024){
            Streams::out << deviceMemorySize
                << "B\n";
        }
        else if(deviceMemorySize < 1024*1024){
            Streams::out << deviceMemorySize/1024
            << "KB\n";
        }
        else if(deviceMemorySize < 1024*1024*1024){
            Streams::out << deviceMemorySize/1024/1024
             << "MB\n";
        }
        else{
            Streams::out << deviceMemorySize/1024/1024/1024
                << "GB\n";
        }
    }

    // Cuda managed memory is used, instead of device memory, as this allocation
    // can become substancial for bigger hamiltonians
    TBTKAssert(
        cudaMallocAsync(reinterpret_cast<void **>(&buffer_device),
            sizeBuffer_device,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "Failed to allocate buffer memory on device.",
        "" 
    )

    buffer_host = malloc(sizeof(data_type) * sizeBuffer_host);
    cudaStreamSynchronize(stream);
    //Run the diagonalization routine
    TBTKAssert(
        cusolverDnXsyevd(
        cusolverHandle, 
        NULL, 
        jobz, 
        uplo,
        n, 
        computeType,
        hamiltonian_device,
        n,
        CUDA_R_64F,
        eigenValues_device,
        computeType,
        buffer_device,
        sizeBuffer_device,
        buffer_host,
        sizeBuffer_host,
        info_device
        ) == CUSOLVER_STATUS_SUCCESS,
        "Diagonalizer::solveGPU()",
        "CUDA error in cusolverDnXsyevd.",
        "" 
    )

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
            matrix,
            hamiltonian_device,
            sizeof(data_type)*n*n,
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error copying to memory from device.",
        ""
    )

    TBTKAssert(
        cudaMemcpyAsync(
            eigenValues,
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
        "CUDA error while synchronizing device stream.",
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
	if(calculationMode == CalculationMode::EigenValuesAndEigenVectors){
		eigenVectorsAvailable = true;
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK