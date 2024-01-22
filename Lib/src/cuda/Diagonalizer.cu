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

#include <cusolverDn.h>
#include <cusolverMg.h>
#include <cuda_runtime.h>
//TODO put his .h in a different directory (3rd party?)
#include "TBTK/cusolverMg_utils.h"

using namespace std;

namespace TBTK{
namespace Solver{

void Diagonalizer::solveMultiGPU(complex<double>* matrix, double* eigenValues, const int &n){
    
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if(numDevices <= 1){
        Streams::out << "Detect 1 gpu or less, falling back to single GPU operation." << endl;
        useMultiGPUAcceleration = false;
        solveGPU(matrix, eigenValues,n);
    }
    //Allocate available devices
    std::vector<int> deviceList(numDevices);
    for(int i = 0; i < numDevices; i++) {
        deviceList[i] = GPUResourceManager::getInstance().allocateDevice();
    }
    GPUResourceManager::getInstance().enableP2PAccess();

    // set up various calculation parameters and resources
    const int IA = 1;
    const int JA = 1;
    const int T_A = 256; // tile size recommended size is 256 or 512
    const int lda = n;

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    cudaLibMgMatrixDesc_t descrMatrix;
    cudaLibMgGrid_t gridA;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    cusolverMgHandle_t cusolverHandle = NULL;
    //TODO check for CUSOLVER_STATUS_SUCCESS
    cusolverMgCreate(&cusolverHandle);
    cusolverMgDeviceSelect(cusolverHandle, numDevices, deviceList.data());
    cusolverMgCreateDeviceGrid(&gridA, 1, numDevices, deviceList.data(), mapping);
    cusolverMgCreateMatrixDesc(&descrMatrix, n, // nubmer of rows of matrix
                                n,          // number of columns of matrix
                                n,          // number or rows in a tile
                                T_A,        // number of columns in a tile
                                CUDA_C_64F, gridA);

    vector<complex<double> *> array_d_A(numDevices, nullptr);

    // Allocate resource for local matrices on the devices
    // createEmptyMatrix(numDevices, 
    //                     deviceList.data(),
    //                     n,                  // number of columns of global matrix
    //                     T_A,                // number of columns per column tile 
    //                     lda,                // leading dimension of local matrix
    //                     array_d_A.data());
    createMat<complex<double>>(numDevices, deviceList.data(), n, /* number of columns of global A */
    T_A,                          /* number of columns per column tile */
    lda,                          /* leading dimension of local A */
    array_d_A.data());
    memcpyH2D<complex<double>>(numDevices, deviceList.data(), n, n,
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
    //TODO check for CUSOLVER_STATUS_SUCCESS
    cusolverMgSyevd_bufferSize(
        cusolverHandle, (cusolverEigMode_t)jobz, CUBLAS_FILL_MODE_LOWER,
        n, reinterpret_cast<void **>(array_d_A.data()), IA,
        JA,                                                     
        descrMatrix, reinterpret_cast<void *>(eigenValues), CUDA_R_64F,
        CUDA_C_64F, &lwork);

    std::vector<complex<double> *> array_d_work(numDevices, nullptr);

    // array_d_work[i] points to device workspace of device i
    workspaceAlloc(numDevices, deviceList.data(),
                    sizeof(complex<double>) * lwork,
                    reinterpret_cast<void **>(array_d_work.data()));
    // sync all devices before calculation
    //TODO check for cudaSuccess
    cudaDeviceSynchronize();
    // Run the eigenvalue solver
    int info = 0;
     //TODO check for CUSOLVER_STATUS_SUCCESS
    cusolverMgSyevd(
        cusolverHandle, (cusolverEigMode_t)jobz, CUBLAS_FILL_MODE_LOWER,
        n, reinterpret_cast<void **>(array_d_A.data()),
        IA, JA, descrMatrix, reinterpret_cast<void **>(eigenValues),
        CUDA_R_64F, CUDA_C_64F,
        reinterpret_cast<void **>(array_d_work.data()), lwork, &info
        );
    // sync all devices after calculation
    //TODO check for cudaSuccess
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data to host from devices
    memcpyD2H<complex<double>>(numDevices, deviceList.data(), n, n,
                         n, 
                         T_A,
                         lda,
                         array_d_A.data(), IA, JA,
                         matrix,
                         lda);

    // Clean up
    destroyMat(numDevices, deviceList.data(), n,
                T_A,
                reinterpret_cast<void **>(array_d_A.data()));
          
    workspaceFree(numDevices, deviceList.data(), reinterpret_cast<void **>(array_d_work.data()));

}

// TODO delete if not needed anymore
// void Diagonalizer::createEmptyMatrix(int numDevices, 
//                                 const int *deviceIdA,
//                                 int N_A,
//                                 int T_A,
//                                 int llda,
//                                 complex<double> **array_d_A){
//     int currentDev = 0; // Get current device id
//     // TODO check against cudaSuccess
//     cudaGetDevice(&currentDev);
//     cudaDeviceSynchronize();
//     const int matrixNumblks = (N_A + T_A - 1) / T_A;
//     const int numBlocksDevice = (matrixNumblks + numDevices - 1) / numDevices;
//     // Allocate base pointers
//     for(int p = 0; p < numDevices; p++){
//         // TODO check against cudaSuccess
//         cudaSetDevice(deviceIdA[p]);
//         // Allocate numBlocksDevice blocks per device
//         // TODO check against cudaSuccess
//         cudaMalloc(&(array_d_A[p]), sizeof(complex<double>) * llda * T_A * numBlocksDevice);
//         // Set local matrix to zero
//         // TODO check against cudaSuccess
//         cudaMemset(array_d_A[p], 0, sizeof(complex<double>) * llda * T_A * numBlocksDevice);
//     }
//     // TODO check against cudaSuccess
//     cudaDeviceSynchronize();
//     cudaSetDevice(currentDev);
// }

void Diagonalizer::solveGPU(complex<double>* matrix, double* eigenValues, const int &n){
    if(useMultiGPUAcceleration){
        solveMultiGPU(matrix, eigenValues, n);
        return;
    }
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
                complex<double> //Matrix size
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
    complex<double> *hamiltonian_device;
    double *eigenValues_device;
    int *info_device = nullptr;

    TBTKAssert(
        cudaMallocAsync(
            reinterpret_cast<void **>(&hamiltonian_device), 
            sizeof(complex<double>) * n*n,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error allocating unified memory.",
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
            sizeof(complex<double>) * n*n, 
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

    buffer_host = malloc(sizeof(complex<double>) * sizeBuffer_host);
    cudaStreamSynchronize(stream);
    //Run the diagonalization routine
    TBTKAssert(
        cusolverDnXsyevd(
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
            sizeof(complex<double>)*n*n,
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
    if(numDevices > 1){
        GPUResourceManager::getInstance().freeDevice(device);
    }
    free(buffer_host);
    buffer_host = nullptr;
}

void Diagonalizer::setupBasisTransformationGPU(){
	//Get the OverlapAmplitudeSet.
	const OverlapAmplitudeSet &overlapAmplitudeSet
		= getModel().getOverlapAmplitudeSet();

	//Skip if the basis is assumed to be orthonormal.
	if(overlapAmplitudeSet.getAssumeOrthonormalBasis()){
		return;
    }

	//Fill the overlap matrix.
	int basisSize = getModel().getBasisSize();
	CArray<complex<double>> overlapMatrix(basisSize*basisSize);
	for(int n = 0; n < basisSize*basisSize; n++)
		overlapMatrix[n] = 0.;

	for(
		OverlapAmplitudeSet::ConstIterator iterator
			= overlapAmplitudeSet.cbegin();
		iterator != overlapAmplitudeSet.cend();
		++iterator
	){
		int row = getModel().getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getBraIndex()
		);
		int col = getModel().getHoppingAmplitudeSet().getBasisIndex(
			(*iterator).getKetIndex()
		);
		if(col >= row){
			overlapMatrix[row + col*basisSize]
				+= (*iterator).getAmplitude();
		}
	}

	//Diagonalize the overlap matrix.
	CArray<double> overlapMatrixEigenValues(basisSize);
    solveGPU( overlapMatrix.getData(), 
                overlapMatrixEigenValues.getData(), basisSize);

	//Setup basisTransformation storage.
	basisTransformation = CArray<complex<double>>(basisSize*basisSize);

	//Calculate the basis transformation using canonical orthogonalization.
	//See for example section 3.4.5 in Moder Quantum Chemistry, Attila
	//Szabo and Neil S. Ostlund.
	for(int row = 0; row < basisSize; row++){
		for(int col = 0; col < basisSize; col++){
			basisTransformation[row + basisSize*col]
				= overlapMatrix[
					row + basisSize*col
				]/sqrt(
					overlapMatrixEigenValues[col]
				);
		}
	}
}

void Diagonalizer::transformToOrthonormalBasisGPU(){
	//Skip if no basis transformation has been set up (the original basis
	//is assumed to be orthonormal).
	if(basisTransformation.getData() == nullptr)
		return;

	int basisSize = getModel().getBasisSize();

	//Perform the transformation H' = U^{\dagger}HU, where U is the
	//transform to the orthonormal basis.
	Matrix<complex<double>> h(basisSize, basisSize);
	Matrix<complex<double>> U(basisSize, basisSize);
	Matrix<complex<double>> Udagger(basisSize, basisSize);
	for(int row = 0; row < basisSize; row++){
		for(int col = 0; col < basisSize; col++){
			if(col >= row){
				h.at(row, col)
					= hamiltonian[row + col*basisSize];
			}
			else{
				h.at(row, col) = conj(
					hamiltonian[col + row*basisSize]
				);
			}

			U.at(row, col)
				= basisTransformation[row + basisSize*col];

			Udagger.at(row, col) = conj(
				basisTransformation[col + basisSize*row]
			);
		}
	}

	Matrix<complex<double>> hp = Udagger*h*U;

	for(int row = 0; row < basisSize; row++){
		for(int col = 0; col < basisSize; col++){
			if(col >= row){
				hamiltonian[row + col*basisSize]
					= hp.at(row, col);
			}
		}
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK