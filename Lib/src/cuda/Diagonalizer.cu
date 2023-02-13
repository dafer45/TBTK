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

using namespace std;

namespace TBTK{
namespace Solver{

void Diagonalizer::solveGPU(CArray<complex<double>>& matrix, CArray<double>& eigenValues){
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
        cudaMallocManaged(
            reinterpret_cast<void **>(&hamiltonian_device), 
            sizeof(complex<double>) * matrix.getSize()
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error allocating unified memory.",
        ""
    ) 
    TBTKAssert(
        cudaMallocManaged(
            reinterpret_cast<void **>(&eigenValues_device),
            sizeof(double) * eigenValues.getSize()
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
            matrix.getData(), 
            sizeof(complex<double>) * matrix.getSize(), 
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

    // Cuda managed memory is used, instead of device memory, as this allocation
    // can become substancial for bigger hamiltonians
    TBTKAssert(
        cudaMallocManaged(reinterpret_cast<void **>(&buffer_device),
            sizeof(complex<double>) * sizeBuffer_device
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "Failed to allocate buffer memory on device.",
        "" 
    )

    buffer_host = malloc(sizeof(complex<double>) * sizeBuffer_host);

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
            matrix.getData(),
            hamiltonian_device,
            sizeof(complex<double>)*matrix.getSize(),
            cudaMemcpyDeviceToHost,
            stream
        ) == cudaSuccess,
        "Diagonalizer::solveGPU()",
        "CUDA error copying to memory from device.",
        ""
    )

    TBTKAssert(
        cudaMemcpyAsync(
            eigenValues.getData(),
            eigenValues_device,
            sizeof(double)*eigenValues.getSize(),
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
    solveGPU( overlapMatrix, 
                overlapMatrixEigenValues);

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