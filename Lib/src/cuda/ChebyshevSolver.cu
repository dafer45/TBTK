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

/** @file ChebyshevExpander.cu
 *
 *  @author Kristofer Björnson
 */

//Flag used to work around incompatibilities between nlohmann::json and CUDA.
//This disables code in header files that depends on nlohmann::json.
#define TBTK_DISABLE_NLOHMANN_JSON

#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/GPUResourceManager.h"
#include "TBTK/HALinkedList.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>

#include <cmath>

using namespace std;

namespace TBTK{
namespace Solver{

complex<double> minus_one(-1., 0.);
complex<double> one(1., 0.);
complex<double> two(2., 0.);
complex<double> zero(0., 0.);
complex<double> i(0., 1.);

vector<complex<double>> ChebyshevExpander::calculateCoefficientsGPU(
	Index to,
	Index from
){
	vector<Index> toVector;
	toVector.push_back(to);

	return calculateCoefficientsGPU(toVector, from)[0];
}

#ifdef TBTK_CUDA_V_12
__global__
void extractCoefficients(
	complex<double>* jResult,
	complex<double>* coefficients,
	int currentCoefficient,
	int* coefficientMap,
	int numParallelCoefficients,
	int numCoefficients
){
	for (int to = blockIdx.x * blockDim.x + threadIdx.x; 
		to < numParallelCoefficients; 
		to += blockDim.x * gridDim.x)
	{
		int iterator = to*numCoefficients + currentCoefficient;
		coefficients[iterator] = jResult[coefficientMap[to]];
	}
}

vector<
	vector<std::complex<double>>
> ChebyshevExpander::calculateCoefficientsGPU(
	vector<Index> &to,
	Index from
){
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevExpander::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevExpander::calculateCoefficients()",
		"numCoefficients has to be larger than zero.",
		""
	);

	int device = GPUResourceManager::getInstance().allocateDevice();

	TBTKAssert(
		cudaSetDevice(device) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA set device error for device " << device << ".",
		""
	);

	vector<vector<complex<double>>> coefficients;
	for(unsigned int n = 0; n < to.size(); n++){
		coefficients.push_back(vector<complex<double>>(numCoefficients,0));
	}

	const HoppingAmplitudeSet &hoppingAmplitudeSet
		= getModel().getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet.getBasisIndex(from);
	thrust::host_vector<int> coefficientMap(to.size());

	for(int n = 0; n < (int)to.size(); n++){
		coefficientMap[n] =
			hoppingAmplitudeSet.getBasisIndex(to.at(n));
	}

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "ChebyshevExpander::calculateCoefficientsGPU\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tBasis size: "
			<< hoppingAmplitudeSet.getBasisSize() << "\n";
		Streams::out << "\tUsing damping: ";
		if(damping != NULL)
			Streams::out << "Yes\n";
		else
			Streams::out << "No\n";
	}

	int basisSize = hoppingAmplitudeSet.getBasisSize();
	thrust::host_vector<complex<double>> jIn1(basisSize, 0.);
	thrust::host_vector<complex<double>> jIn2(basisSize, 0.);

	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;


	for(int n = 0; n < (int)to.size(); n++)
			coefficients[0][n] = jIn1[coefficientMap[n]];

	SparseMatrix<complex<double>> sparseMatrix = hoppingAmplitudeSet.getSparseMatrix();
	sparseMatrix.setStorageFormat(SparseMatrix<complex<double>>::StorageFormat::CSR);
	const int numHoppingAmplitudes = sparseMatrix.getCSRNumMatrixElements();
	const int numRows = sparseMatrix.getNumRows();
	const unsigned int *csrRowPointers = sparseMatrix.getCSRRowPointers();
	const unsigned int *csrColumns = sparseMatrix.getCSRColumns();
	const complex<double> *csrValues = sparseMatrix.getCSRValues();

	// calculate total memory requirement for the device
	int totalMemoryRequirement
		= hoppingAmplitudeSet.getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += hoppingAmplitudeSet.getBasisSize()*sizeof(
		complex<double>
	);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(int);
	totalMemoryRequirement += hoppingAmplitudeSet.getBasisSize()*sizeof(
		int
	);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(int);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(
		complex<double>
	);
	totalMemoryRequirement += to.size()*numCoefficients*sizeof(
		complex<double>
	);
	totalMemoryRequirement += hoppingAmplitudeSet.getBasisSize()*sizeof(
		int
	);
	if(damping != NULL){
		totalMemoryRequirement += hoppingAmplitudeSet.getBasisSize(
		)*sizeof(complex<double>);
	}
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA memory requirement: ";
		if(totalMemoryRequirement < 1024){
			Streams::out << totalMemoryRequirement << "B\n";
		}
		else if(totalMemoryRequirement < 1024*1024){
			Streams::out << totalMemoryRequirement/1024 << "KB\n";
		}
		else{
			Streams::out << totalMemoryRequirement/1024/1024
				<< "MB\n";
		}
	}

	//Initialize GPU
	thrust::device_vector<complex<double>> jIn1_device = jIn1; //Already allocates and copies into the device memory
	thrust::device_vector<complex<double>> jIn2_device = jIn2;
	// int *cooHARowIndices_device;
	thrust::device_vector<int> csrHARowIndices_device(csrRowPointers,
													csrRowPointers + (numRows + 1));
	thrust::device_vector<int> csrColumns_device(csrColumns, csrColumns + numHoppingAmplitudes);
	const thrust::device_vector<complex<double>> csrValues_device(csrValues, csrValues + numHoppingAmplitudes);

	//The device coefficients are stored in a flattened array as complare to the host coefficients
	thrust::device_vector<complex<double>> coefficients_device(to.size()*numCoefficients);
	thrust::device_vector<int> coefficientMap_device = coefficientMap;
	complex<double> *damping_device = NULL;

	if(damping != NULL){
		TBTKAssert(
			cudaMalloc(
				(void**)&damping_device,
				hoppingAmplitudeSet.getBasisSize()*sizeof(
					complex<double>
				)
			) == cudaSuccess,
			"ChebyshevExpander::calculateCoefficientsGPU()",
			"CUDA malloc error while allocating damping_device.",
			""
		);
	}
	//Only the zeroth coefficients have been calculated, so they are copied to the device
	for(int n = 0; n < (int)to.size(); n++){
		thrust::copy(coefficients[n].begin(),
		coefficients[n].begin() + 1,
		coefficients_device.begin() + n*numCoefficients
		);
	}

	if(damping != NULL){
		TBTKAssert(
			cudaMemcpy(
				damping_device,
				damping,
				hoppingAmplitudeSet.getBasisSize()*sizeof(
					complex<double>
				),
				cudaMemcpyHostToDevice
			) == cudaSuccess,
			"ChebyshevExpander::calculateCoefficientsGPU()",
			"CUDA memcpy error while copying damping.",
			""
		);
	}

	cusparseHandle_t handle = NULL;
	TBTKAssert(
		cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"cuSPARSE create error.",
		""
	);

	//Create a sparse matrix on the device
    cusparseSpMatDescr_t descr;
    TBTKAssert( cusparseCreateCsr(&descr, numRows, numRows,
		numHoppingAmplitudes,
		csrHARowIndices_device.data().get(), 
		csrColumns_device.data().get(), 
		(void*) csrValues_device.data().get(),
					  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
					  CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F)
				== CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"Error in cusparseCreateCsr.",
		""
		);

	//Create the dense vector objects on the device
	cusparseDnVecDescr_t vecJIn1, vecJIn2;
	TBTKAssert(
		cusparseCreateDnVec(&vecJIn1, basisSize, jIn1_device.data().get(), CUDA_C_64F)
		== CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"Error in cusparseCreateDnVec.",
		""
	);
	TBTKAssert(
		cusparseCreateDnVec(&vecJIn2, basisSize, jIn2_device.data().get(), CUDA_C_64F)
		== CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"Error in cusparseCreateDnVec.",
		""
	);



	//Calculate |j1>
	int block_size = to.size();
	int num_blocks = 1;
	int maxNumThreads = 1024;
	if(block_size > maxNumThreads){
		block_size = maxNumThreads;
		num_blocks = 1 + ((to.size() - 1) / maxNumThreads); // ceil(to.size()/maxNumThreads)
	}
	
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA Block size: " << block_size << "\n";
		Streams::out << "\tCUDA Num blocks: " << num_blocks << "\n";
	}


	complex<double> multiplier = one/scaleFactor;
	//Allocate buffer memory for the cusparseSpMV routine
    void                *buffer_device = NULL;
    size_t               bufferSize = 0;
    TBTKAssert( 
		cusparseSpMV_bufferSize(
			handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&multiplier, descr, vecJIn1, &zero, vecJIn2, CUDA_C_64F,
			CUSPARSE_SPMV_CSR_ALG2, &bufferSize) 
		== CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"Error in cusparseSpMV_bufferSize.",
		"Error occured while allocating extra buffer in device memory"
	);
    TBTKAssert(
		cudaMalloc(&buffer_device, bufferSize)
		== cudaSuccess,
	   "ChebyshevExpander::calculateCoefficientsGPU()",
	   "Error in cudaMalloc.",
	   "Error occured while allocating extra buffer in device memory"
   	);
	TBTKAssert(
		cusparseSpMV(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			&multiplier, descr, vecJIn1, 
			&zero, vecJIn2,
			CUDA_C_64F,
			CUSPARSE_SPMV_CSR_ALG2, 
			buffer_device
		) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficentsGPU()",
		"Matrix-vector multiplication error.",
		""
	);

	size_t bufferSizeSecondOperation = 0;
    TBTKAssert( 
		cusparseSpMV_bufferSize(
			handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&multiplier, descr, vecJIn1, &minus_one, vecJIn2, CUDA_C_64F,
			CUSPARSE_SPMV_CSR_ALG2, &bufferSizeSecondOperation) 
		== CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"Error in cusparseSpMV_bufferSize.",
		"Error occured while allocating extra buffer in device memory"
	);
	TBTKAssert(
		bufferSizeSecondOperation <= bufferSize,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"Error in Allocating buffer for SPMV.",
		"Buffer memory requirements changed."
	);
	extractCoefficients <<< num_blocks, block_size >>> (
		jIn2_device.data().get(),
		coefficients_device.data().get(),
		1,
		coefficientMap_device.data().get(),
		to.size(),
		numCoefficients
	);
	//Switch the order of the vectors jIn1 <-> jIn2
	cusparseDnVecDescr_t *vecJIn1_ptr = &vecJIn1;
	cusparseDnVecDescr_t *vecJIn2_ptr = &vecJIn2;
	cusparseDnVecDescr_t *vecJTemp_ptr = NULL;
	thrust::device_ptr<complex<double>> jIn1_device_ptr = jIn1_device.data();
	thrust::device_ptr<complex<double>> jIn2_device_ptr = jIn2_device.data();
	thrust::device_ptr<complex<double>> jTemp_device_ptr = NULL;
	
	vecJTemp_ptr = vecJIn2_ptr;
	vecJIn2_ptr = vecJIn1_ptr;
	vecJIn1_ptr = vecJTemp_ptr;

	jTemp_device_ptr = jIn2_device_ptr;
	jIn2_device_ptr = jIn1_device_ptr;
	jIn1_device_ptr = jTemp_device_ptr;

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\tProgress (100 coefficients per dot): ";

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		multiplier = two/scaleFactor;
		TBTKAssert(
			cusparseSpMV(
				handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				&multiplier, descr, *vecJIn1_ptr, 
				&minus_one, *vecJIn2_ptr,
				CUDA_C_64F,
				CUSPARSE_SPMV_CSR_ALG2, 
				buffer_device
			) == CUSPARSE_STATUS_SUCCESS,
			"ChebyshevExpander::calculateCoefficentsGPU()",
			"Matrix-vector multiplication error.",
			""
		);
		extractCoefficients <<< num_blocks, block_size >>> (
			jIn2_device_ptr.get(),
			coefficients_device.data().get(),
			n,
			coefficientMap_device.data().get(),
			to.size(),
			numCoefficients
		);
		vecJTemp_ptr = vecJIn2_ptr;
		vecJIn2_ptr = vecJIn1_ptr;
		vecJIn1_ptr = vecJTemp_ptr;

		jTemp_device_ptr = jIn2_device_ptr;
		jIn2_device_ptr = jIn1_device_ptr;
		jIn1_device_ptr = jTemp_device_ptr;

		if(getGlobalVerbose() && getVerbose()){
			if(n%100 == 0)
				Streams::out << "." << flush;
			if(n%1000 == 0)
				Streams::out << " " << flush;
		}
	}
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\n";
	for(unsigned int n = 0; n < to.size(); n++){
		thrust::copy(
			coefficients_device.begin() + n*numCoefficients,
			coefficients_device.begin() + numCoefficients*(n+1),
			coefficients[n].begin()
		);
	}

    TBTKAssert(
		cusparseDestroySpMat(descr
	   ) == CUSPARSE_STATUS_SUCCESS,
	   "ChebyshevExpander::calculateCoefficientsGPU()",
	   "Error in cusparseDestroySpMat.",
	   "Error while destroying the handle descr stored on the device."	
   	)
	descr = NULL;

    TBTKAssert(
		cusparseDestroyDnVec(vecJIn1
	   ) == CUSPARSE_STATUS_SUCCESS,
	   "ChebyshevExpander::calculateCoefficientsGPU()",
	   "Error in cusparseDestroyDnVec.",
	   "Error while destroying the dense vector vecJIn1 stored on the device."	
    )
	vecJIn1 = NULL;
    TBTKAssert(
		cusparseDestroyDnVec(vecJIn2
	   ) == CUSPARSE_STATUS_SUCCESS,
	   "ChebyshevExpander::calculateCoefficientsGPU()",
	   "Error in cusparseDestroyDnVec.",
	   "Error while destroying the dense vector vecJIn2 stored on the device."	
    )
	vecJIn2 = NULL;
    TBTKAssert(
		cudaFree(buffer_device
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA free error while deallocating buffer_device.",
		""
	)
	buffer_device = NULL;
    TBTKAssert(
		cusparseDestroy(handle
	   ) == CUSPARSE_STATUS_SUCCESS,
	   "ChebyshevExpander::calculateCoefficientsGPU()",
	   "Error in cusparseDestroy.",
	   "Error while destroying the handle for the cuSparse calculation."	
    )
	handle = NULL;

	if(damping != NULL)
		cudaFree(damping_device);

	GPUResourceManager::getInstance().freeDevice(device);

	//Lorentzian convolution
	if(broadening != 0){
		double lambda = broadening*numCoefficients;
		for(int n = 0; n < numCoefficients; n++){
			for(int c = 0; c < (int)to.size(); c++){
				coefficients[c][n] = coefficients[c][n]*sinh(
					lambda*(
						1 - n/(double)numCoefficients
					)
				)/sinh(lambda);
//				coefficients[n + c*numCoefficients] = coefficients[n + c*numCoefficients]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
			}
		}
	}

	return coefficients;
}
#else // TBTK_CUDA_V_12
__global__
void extractCoefficients(
	cuDoubleComplex *jResult,
	int basisSize,
	cuDoubleComplex *coefficients,
	int currentCoefficient,
	int *coefficientMap,
	int numCoefficients
){
	int to = blockIdx.x*blockDim.x + threadIdx.x;
	if(to < basisSize && coefficientMap[to] != -1){
		coefficients[
			coefficientMap[to]*numCoefficients + currentCoefficient
		] = jResult[to];
	}
}

vector<
	vector<std::complex<double>>
> ChebyshevExpander::calculateCoefficientsGPU(
	vector<Index> &to,
	Index from
){
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevExpander::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevExpander::calculateCoefficients()",
		"numCoefficients has to be larger than zero.",
		""
	);

	int device = GPUResourceManager::getInstance().allocateDevice();

	TBTKAssert(
		cudaSetDevice(device) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA set device error for device " << device << ".",
		""
	);

	vector<vector<complex<double>>> coefficients;
	for(unsigned int n = 0; n < to.size(); n++){
		coefficients.push_back(vector<complex<double>>());
		coefficients[n].reserve(numCoefficients);
		for(int c = 0; c < numCoefficients; c++)
			coefficients[n].push_back(0);
	}

	const HoppingAmplitudeSet &hoppingAmplitudeSet
		= getModel().getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet.getBasisIndex(from);
	int *coefficientMap = new int[hoppingAmplitudeSet.getBasisSize()];
	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++)
		coefficientMap[n] = -1;
	for(int n = 0; n < (int)to.size(); n++){
		coefficientMap[
			hoppingAmplitudeSet.getBasisIndex(to.at(n))
		] = n;
	}

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "ChebyshevExpander::calculateCoefficientsGPU\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tBasis size: "
			<< hoppingAmplitudeSet.getBasisSize() << "\n";
		Streams::out << "\tUsing damping: ";
		if(damping != NULL)
			Streams::out << "Yes\n";
		else
			Streams::out << "No\n";
	}

	complex<double> *jIn1
		= new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jIn2
		= new complex<double>[hoppingAmplitudeSet.getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
	}

	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	for(int n = 0; n < hoppingAmplitudeSet.getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]][0] = jIn1[n];
//			coefficients[coefficientMap[n]*numCoefficients] = jIn1[n];

	SparseMatrix<complex<double>> sparseMatrix = hoppingAmplitudeSet.getSparseMatrix();
	sparseMatrix.setStorageFormat(SparseMatrix<complex<double>>::StorageFormat::CSR);

	const int numHoppingAmplitudes = sparseMatrix.getCSRNumMatrixElements();
	const unsigned int *csrRowPointers = sparseMatrix.getCSRRowPointers();
	const unsigned int *csrColumns = sparseMatrix.getCSRColumns();
	const complex<double> *csrValues = sparseMatrix.getCSRValues();
	int *cooHARowIndices_host = new int[numHoppingAmplitudes];
	int *cooHAColIndices_host = new int[numHoppingAmplitudes];
	complex<double> *cooHAValues_host = new complex<double>[
		numHoppingAmplitudes
	];
	for(
		unsigned int row = 0;
		row < sparseMatrix.getNumRows();
		row++
	){
		for(
			unsigned int n = csrRowPointers[row];
			n < csrRowPointers[row+1];
			n++
		){
			cooHARowIndices_host[n] = row;
			cooHAColIndices_host[n] = csrColumns[n];
			cooHAValues_host[n] = csrValues[n];
		}
	}

	//Initialize GPU
	complex<double> *jIn1_device;
	complex<double> *jIn2_device;
	int *cooHARowIndices_device;
	int *csrHARowIndices_device;
	int *cooHAColIndices_device;
	complex<double> *cooHAValues_device;
	complex<double> *coefficients_device;
	int *coefficientMap_device;
	complex<double> *damping_device = NULL;

	int totalMemoryRequirement
		= hoppingAmplitudeSet.getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += hoppingAmplitudeSet.getBasisSize()*sizeof(
		complex<double>
	);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(int);
	totalMemoryRequirement += hoppingAmplitudeSet.getBasisSize()*sizeof(
		int
	);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(int);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(
		complex<double>
	);
	totalMemoryRequirement += to.size()*numCoefficients*sizeof(
		complex<double>
	);
	totalMemoryRequirement += hoppingAmplitudeSet.getBasisSize()*sizeof(
		int
	);
	if(damping != NULL){
		totalMemoryRequirement += hoppingAmplitudeSet.getBasisSize(
		)*sizeof(complex<double>);
	}
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA memory requirement: ";
		if(totalMemoryRequirement < 1024){
			Streams::out << totalMemoryRequirement/1024 << "B\n";
		}
		else if(totalMemoryRequirement < 1024*1024){
			Streams::out << totalMemoryRequirement/1024 << "KB\n";
		}
		else{
			Streams::out << totalMemoryRequirement/1024/1024
				<< "MB\n";
		}
	}

	TBTKAssert(
		cudaMalloc(
			(void**)&jIn1_device,
			hoppingAmplitudeSet.getBasisSize()*sizeof(
				complex<double>
			)
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating jIn1_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&jIn2_device,
			hoppingAmplitudeSet.getBasisSize()*sizeof(
				complex<double>
			)
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating jIn2_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&cooHARowIndices_device,
			numHoppingAmplitudes*sizeof(int)
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating cooHARowIndices_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&csrHARowIndices_device,
			(hoppingAmplitudeSet.getBasisSize()+1)*sizeof(int)
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating csrHARowIndices_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&cooHAColIndices_device,
			numHoppingAmplitudes*sizeof(int)
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating cooHAColIndices_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&cooHAValues_device,
			numHoppingAmplitudes*sizeof(complex<double>)
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating cooHAValues_device.",
		""
	)
	TBTKAssert(
		cudaMalloc(
			(void**)&coefficients_device,
			to.size()*numCoefficients*sizeof(complex<double>)
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating coefficients_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&coefficientMap_device,
			hoppingAmplitudeSet.getBasisSize()*sizeof(int)
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating coefficientMap_device.",
		""
	);
	if(damping != NULL){
		TBTKAssert(
			cudaMalloc(
				(void**)&damping_device,
				hoppingAmplitudeSet.getBasisSize()*sizeof(
					complex<double>
				)
			) == cudaSuccess,
			"ChebyshevExpander::calculateCoefficientsGPU()",
			"CUDA malloc error while allocating damping_device.",
			""
		);
	}

	TBTKAssert(
		cudaMemcpy(
			jIn1_device,
			jIn1,
			hoppingAmplitudeSet.getBasisSize()*sizeof(
				complex<double>
			),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying jIn1.",
		""
	);
	TBTKAssert(
		cudaMemcpy(
			jIn2_device,
			jIn2,
			hoppingAmplitudeSet.getBasisSize()*sizeof(
				complex<double>
			),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying jIn2.",
		""
	);
	TBTKAssert(
		cudaMemcpy(
			cooHARowIndices_device,
			cooHARowIndices_host,
			numHoppingAmplitudes*sizeof(int),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying cooHARowIndices.",
		""
	);
	TBTKAssert(
		cudaMemcpy(
			cooHAColIndices_device,
			cooHAColIndices_host,
			numHoppingAmplitudes*sizeof(int),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficients()",
		"CUDA memcpy error while copying cooHAColIndices.",
		""
	)
	TBTKAssert(
		cudaMemcpy(
			cooHAValues_device,
			cooHAValues_host,
			numHoppingAmplitudes*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying cooHAValues.",
		""
	);
	for(unsigned int n = 0; n < to.size(); n++){
		TBTKAssert(
			cudaMemcpy(
				coefficients_device + numCoefficients*n,
				coefficients[n].data(),
				numCoefficients*sizeof(complex<double>),
				cudaMemcpyHostToDevice
			) == cudaSuccess,
			"ChebyshevExpander::calculateCoefficients()",
			"CUDA memcpy error while copying coefficients.",
			""
		);
	}
/*	TBTKAssert(
		cudaMemcpy(
			coefficients_device,
			coefficients.data(),
			to.size()*numCoefficients*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficients()",
		"CUDA memcpy error while copying coefficients.",
		""
	);*/
	TBTKAssert(
		cudaMemcpy(
			coefficientMap_device,
			coefficientMap,
			hoppingAmplitudeSet.getBasisSize()*sizeof(int),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying coefficientMap.",
		""
	);
	if(damping != NULL){
		TBTKAssert(
			cudaMemcpy(
				damping_device,
				damping,
				hoppingAmplitudeSet.getBasisSize()*sizeof(
					complex<double>
				),
				cudaMemcpyHostToDevice
			) == cudaSuccess,
			"ChebyshevExpander::calculateCoefficientsGPU()",
			"CUDA memcpy error while copying damping.",
			""
		);
	}

	cusparseHandle_t handle = NULL;
	TBTKAssert(
		cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"cuSPARSE create error.",
		""
	);

	cusparseMatDescr_t descr = NULL;
	TBTKAssert(
		cusparseCreateMatDescr(&descr) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"cuSPARSE create matrix descriptor error.",
		""
	);

	TBTKAssert(
		cusparseSetMatType(
			descr,
			CUSPARSE_MATRIX_TYPE_GENERAL
		) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"cuSPARSE set matrix type error.",
		""
	);
	TBTKAssert(
		cusparseSetMatIndexBase(
			descr,
			CUSPARSE_INDEX_BASE_ZERO
		) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"cuSPARSE set matrix index base error.",
		""
	);

	TBTKAssert(
		cusparseXcoo2csr(
			handle,
			cooHARowIndices_device,
			numHoppingAmplitudes,
			hoppingAmplitudeSet.getBasisSize(),
			csrHARowIndices_device,
			CUSPARSE_INDEX_BASE_ZERO
		) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"cuSPARSE COO to CSR error.",
		""
	);

	//Calculate |j1>
	int block_size = 1024;
	int num_blocks = hoppingAmplitudeSet.getBasisSize()/block_size
		+ (hoppingAmplitudeSet.getBasisSize()%block_size == 0 ? 0:1);
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA Block size: " << block_size << "\n";
		Streams::out << "\tCUDA Num blocks: " << num_blocks << "\n";
	}

	complex<double> multiplier = one/scaleFactor;
	TBTKAssert(
		cusparseZcsrmv(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			hoppingAmplitudeSet.getBasisSize(),
			hoppingAmplitudeSet.getBasisSize(),
			numHoppingAmplitudes,
			(cuDoubleComplex*)&multiplier,
			descr,
			(cuDoubleComplex*)cooHAValues_device,
			csrHARowIndices_device,
			cooHAColIndices_device,
			(cuDoubleComplex*)jIn1_device,
			(cuDoubleComplex*)&zero,
			(cuDoubleComplex*)jIn2_device
		) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficentsGPU()",
		"Matrix-vector multiplication error.",
		""
	);

	extractCoefficients <<< num_blocks, block_size >>> (
		(cuDoubleComplex*)jIn2_device,
		hoppingAmplitudeSet.getBasisSize(),
		(cuDoubleComplex*)coefficients_device,
		1,
		coefficientMap_device,
		numCoefficients
	);
	jTemp = jIn2_device;
	jIn2_device = jIn1_device;
	jIn1_device = jTemp;

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\tProgress (100 coefficients per dot): ";

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		multiplier = two/scaleFactor;
		TBTKAssert(
			cusparseZcsrmv(
				handle,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				hoppingAmplitudeSet.getBasisSize(),
				hoppingAmplitudeSet.getBasisSize(),
				numHoppingAmplitudes,
				(cuDoubleComplex*)&multiplier,
				descr,
				(cuDoubleComplex*)cooHAValues_device,
				csrHARowIndices_device,
				cooHAColIndices_device,
				(cuDoubleComplex*)jIn1_device,
				(cuDoubleComplex*)&minus_one,
				(cuDoubleComplex*)jIn2_device
			) == CUSPARSE_STATUS_SUCCESS,
			"ChebyshevExpander::calculateCoefficientsGPU()",
			"Matrix-vector multiplication error.",
			""
		);

		extractCoefficients <<< num_blocks, block_size >>> (
			(cuDoubleComplex*)jIn2_device,
			hoppingAmplitudeSet.getBasisSize(),
			(cuDoubleComplex*)coefficients_device,
			n,
			coefficientMap_device,
			numCoefficients
		);

		jTemp = jIn2_device;
		jIn2_device = jIn1_device;
		jIn1_device = jTemp;

		if(getGlobalVerbose() && getVerbose()){
			if(n%100 == 0)
				Streams::out << "." << flush;
			if(n%1000 == 0)
				Streams::out << " " << flush;
		}
	}
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "\n";

	for(unsigned int n = 0; n < to.size(); n++){
		TBTKAssert(
			cudaMemcpy(
				coefficients[n].data(),
				coefficients_device + numCoefficients*n,
				numCoefficients*sizeof(complex<double>),
				cudaMemcpyDeviceToHost
			) == cudaSuccess,
			"ChebyshevExpander::calculateCoefficientsGPU()",
			"CUDA memcpy error while copying coefficients.",
			""
		);
	}
/*	TBTKAssert(
		cudaMemcpy(
			coefficients.data(),
			coefficients_device,
			to.size()*numCoefficients*sizeof(complex<double>),
			cudaMemcpyDeviceToHost
		) == cudaSuccess,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying coefficients.",
		""
	);*/

	TBTKAssert(
		cusparseDestroyMatDescr(descr) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"cuSPARSE destroy matrix descriptor error.",
		""
	);
	descr = NULL;

	TBTKAssert(
		cusparseDestroy(handle) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevExpander::calculateCoefficientsGPU()",
		"cuSPARSE destroy error.",
		""
	);
	handle = NULL;

	delete [] jIn1;
	delete [] jIn2;
	delete [] coefficientMap;
	delete [] cooHARowIndices_host;
	delete [] cooHAColIndices_host;
	delete [] cooHAValues_host;

	cudaFree(jIn1_device);
	cudaFree(jIn2_device);
	cudaFree(cooHARowIndices_device);
	cudaFree(csrHARowIndices_device);
	cudaFree(cooHAColIndices_device);
	cudaFree(cooHAValues_device);
	cudaFree(coefficients_device);
	cudaFree(coefficientMap_device);
	if(damping != NULL)
		cudaFree(damping_device);

	GPUResourceManager::getInstance().freeDevice(device);

	//Lorentzian convolution
	if(broadening != 0){
		double lambda = broadening*numCoefficients;
		for(int n = 0; n < numCoefficients; n++){
			for(int c = 0; c < (int)to.size(); c++){
				coefficients[c][n] = coefficients[c][n]*sinh(
					lambda*(
						1 - n/(double)numCoefficients
					)
				)/sinh(lambda);
//				coefficients[n + c*numCoefficients] = coefficients[n + c*numCoefficients]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
			}
		}
	}

	return coefficients;
}

#endif // TBTK_CUDA_V_12

__global__
void calculateGreensFunction(
	cuDoubleComplex *greensFunction,
	cuDoubleComplex *coefficients,
	cuDoubleComplex *lookupTable,
	int numCoefficients,
	int energyResolution
){
	int e = blockIdx.x*blockDim.x + threadIdx.x;
	if(e < energyResolution){
		for(int n = 0; n < numCoefficients; n++){
			greensFunction[e] = cuCadd(
				greensFunction[e],
				cuCmul(
					lookupTable[n*energyResolution + e],
					coefficients[n]
				)
			);
		}
	}
}

void ChebyshevExpander::loadLookupTableGPU(){
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "CheyshevExpander::loadLookupTableGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable != NULL,
		"ChebyshevExpander::loadLookupTableGPU()",
		"Lookup table has not been generated.",
		"Call ChebyshevExpander::generateLokupTable() to generate"
		<< " lookup table."
	);
	if(generatingFunctionLookupTable_device != NULL)
		destroyLookupTableGPU();

	complex<double> *generatingFunctionLookupTable_host
		= new complex<double>[
			lookupTableNumCoefficients*lookupTableResolution
		];
	for(int n = 0; n < lookupTableNumCoefficients; n++){
		for(int e = 0; e < lookupTableResolution; e++){
			generatingFunctionLookupTable_host[
				n*lookupTableResolution + e
			] = generatingFunctionLookupTable[n][e];
		}
	}

	int memoryRequirement
		= lookupTableNumCoefficients*lookupTableResolution*sizeof(
			complex<double>
		);
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA memory requirement: ";
		if(memoryRequirement < 1024)
			Streams::out << memoryRequirement << "B\n";
		else if(memoryRequirement < 1024*1024)
			Streams::out << memoryRequirement/1024 << "KB\n";
		else
			Streams::out << memoryRequirement/1024/1024 << "MB\n";
	}

	generatingFunctionLookupTable_device = new complex<double>**[
		GPUResourceManager::getInstance().getNumDevices()
	];

	for(
		int n = 0;
		n < GPUResourceManager::getInstance().getNumDevices();
		n++
	){
		TBTKAssert(
			cudaSetDevice(n) == cudaSuccess,
			"ChebyshevExpander::loadLookupTableGPU()",
			"CUDA set device error for device " << n << ".",
			""
		);

		TBTKAssert(
			cudaMalloc(
				(void**)&generatingFunctionLookupTable_device[
					n
				],
				lookupTableNumCoefficients*lookupTableResolution*sizeof(
					complex<double>
				)
			)  == cudaSuccess,
			"ChebyshevExpander::loadLookupTableGPU()",
			"CUDA malloc error while allocating"
			<< " generatingFunctionLookupTable_device.",
			""
		);

		TBTKAssert(
			cudaMemcpy(
				generatingFunctionLookupTable_device[n],
				generatingFunctionLookupTable_host,
				lookupTableNumCoefficients*lookupTableResolution*sizeof(
					complex<double>
				),
				cudaMemcpyHostToDevice
			) == cudaSuccess,
			"ChebyshevExpander::loadLookupTableGPU()",
			"CUDA memcpy error while copying"
			<< " generatingFunctionLookupTable_device.",
			""
		);
	}

	delete [] generatingFunctionLookupTable_host;
}

void ChebyshevExpander::destroyLookupTableGPU(){
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "ChebyshevExpander::destroyLookupTableGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable_device != NULL,
		"ChebyshevExpander::destroyLookupTableGPU()",
		"No lookup table loaded onto GPU.\n",
		""
	);

	for(
		int n = 0;
		n < GPUResourceManager::getInstance().getNumDevices();
		n++
	){
		cudaFree(generatingFunctionLookupTable_device[n]);
	}

	delete [] generatingFunctionLookupTable_device;
	generatingFunctionLookupTable_device = NULL;
}

//Property::GreensFunction* ChebyshevExpander::generateGreensFunctionGPU(
vector<complex<double>> ChebyshevExpander::generateGreensFunctionGPU(
	const vector<complex<double>> &coefficients,
//	Property::GreensFunction::Type type
	Type type
){
	int device = GPUResourceManager::getInstance().allocateDevice();

	TBTKAssert(
		cudaSetDevice(device) == cudaSuccess,
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"CUDA set device error for device " << device << ".",
		""
	);

	ensureLookupTableIsReady();

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "ChebyshevExpander::generateGreensFunctionGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable_device != NULL,
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"No lookup table loaded onto GPU.",
		""
	);
	TBTKAssert(
//		type == Property::GreensFunction::Type::Retarded,
		type == Type::Retarded,
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"Only evaluation of retarded Green's function is implemented"
		<< " for GPU so far.",
		"Use CPU evaluation instead."
	);

/*	complex<double> *greensFunctionData = new complex<double>[
		lookupTableResolution
	];

	for(int e = 0; e < lookupTableResolution; e++)
		greensFunctionData[e] = 0.;*/

	vector<complex<double>> greensFunctionData(lookupTableResolution, 0.);

	complex<double> *greensFunctionData_device;
	complex<double> *coefficients_device;

	TBTKAssert(
		cudaMalloc(
			(void**)&greensFunctionData_device,
			lookupTableResolution*sizeof(complex<double>)
		)  == cudaSuccess,
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"CUDA malloc error while allocating greensFunction_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&coefficients_device,
			lookupTableNumCoefficients*sizeof(complex<double>)
		)  == cudaSuccess,
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"CUDA malloc error while allocating coefficients_device.",
		""
	);

	TBTKAssert(
		cudaMemcpy(
			greensFunctionData_device,
			greensFunctionData.data(),
			lookupTableResolution*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"CUDA memcpy error while copying greensFunctionData.",
		""
	);
	TBTKAssert(
		cudaMemcpy(
			coefficients_device,
			coefficients.data(),
			lookupTableNumCoefficients*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"CUDA memcpy error while copying coefficients.",
		""
	);

	int block_size = 1024;
	int num_blocks = lookupTableResolution/block_size
		+ (lookupTableResolution%block_size == 0 ? 0:1);

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA Block size: " << block_size << "\n";
		Streams::out << "\tCUDA Num blocks: " << num_blocks << "\n";
	}

	calculateGreensFunction <<< num_blocks, block_size>>> (
		(cuDoubleComplex*)greensFunctionData_device,
		(cuDoubleComplex*)coefficients_device,
		(cuDoubleComplex*)generatingFunctionLookupTable_device[device],
		lookupTableNumCoefficients,
		lookupTableResolution
	);

	TBTKAssert(
		cudaMemcpy(
			greensFunctionData.data(),
			greensFunctionData_device,
			lookupTableResolution*sizeof(complex<double>),
			cudaMemcpyDeviceToHost
		) == cudaSuccess,
		"ChebyshevExpander::generateGreensFunctionGPU()",
		"CUDA memcpy error while copying greensFunction_device.",
		""
	);

	cudaFree(greensFunctionData_device);
	cudaFree(coefficients_device);

	GPUResourceManager::getInstance().freeDevice(device);

/*	Property::GreensFunction *greensFunction = new Property::GreensFunction(
		type,
//		Property::GreensFunction::Format::Array,
		lookupTableLowerBound,
		lookupTableUpperBound,
		lookupTableResolution,
		greensFunctionData
	);
	delete [] greensFunctionData;

	return greensFunction;*/

	return greensFunctionData;
}

};	//End of namespace Solver
};	//End of namespace TBTK
