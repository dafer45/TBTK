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

/** @file ChebyshevSolver.cu
 *
 *  @author Kristofer Björnson
 */

#include "ChebyshevSolver.h"
#include "GPUResourceManager.h"
#include "HALinkedList.h"
#include "Streams.h"
#include "TBTKMacros.h"

#include <cuComplex.h>
#include <cusparse_v2.h>

#include <math.h>

using namespace std;

namespace TBTK{

complex<double> minus_one(-1., 0.);
complex<double> one(1., 0.);
complex<double> two(2., 0.);
complex<double> zero(0., 0.);
complex<double> i(0., 1.);

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
	if(to < basisSize && coefficientMap[to] != -1)
		coefficients[coefficientMap[to]*numCoefficients + currentCoefficient] = jResult[to];
}

void ChebyshevSolver::calculateCoefficientsGPU(
	Index to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double broadening
){
	vector<Index> toVector;
	toVector.push_back(to);
	calculateCoefficientsGPU(toVector, from, coefficients, numCoefficients, broadening);
}

void ChebyshevSolver::calculateCoefficientsGPU(
	vector<Index> &to,
	Index from,
	complex<double> *coefficients,
	int numCoefficients,
	double broadening
){
	TBTKAssert(
		getModel() != NULL,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"Model not set",
		"Use ChebyshevSolver::setModel() to set model."
	);
	TBTKAssert(
		scaleFactor > 0,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"Scale factor must be larger than zero.",
		"Use ChebyshevSolver::setScaleFactor() to set scale factor."
	);
	TBTKAssert(
		numCoefficients > 0,
		"ChebyshevSolver::calculateCoefficients()",
		"numCoefficients has to be larger than zero.",
		""
	);

	int device = GPUResourceManager::getInstance().allocateDevice();

	TBTKAssert(
		cudaSetDevice(device) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA set device error for device " << device << ".",
		""
	);

	const HoppingAmplitudeSet *hoppingAmplitudeSet = getModel()->getHoppingAmplitudeSet();

	int fromBasisIndex = hoppingAmplitudeSet->getBasisIndex(from);
	int *coefficientMap = new int[hoppingAmplitudeSet->getBasisSize()];
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++)
		coefficientMap[n] = -1;
	for(int n = 0; n < to.size(); n++)
		coefficientMap[hoppingAmplitudeSet->getBasisIndex(to.at(n))] = n;

	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "ChebyshevSolver::calculateCoefficientsGPU\n";
		Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Streams::out << "\tBasis size: " << hoppingAmplitudeSet->getBasisSize() << "\n";
		Streams::out << "\tUsing damping: ";
		if(damping != NULL)
			Streams::out << "Yes\n";
		else
			Streams::out << "No\n";
	}

	complex<double> *jIn1 = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[hoppingAmplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
	}

	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	for(int n = 0; n < hoppingAmplitudeSet->getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]*numCoefficients] = jIn1[n];

	const int numHoppingAmplitudes = hoppingAmplitudeSet->getNumMatrixElements();
	const int *cooHARowIndices_host = hoppingAmplitudeSet->getCOORowIndices();
	const int *cooHAColIndices_host = hoppingAmplitudeSet->getCOOColIndices();
	const complex<double> *cooHAValues_host = hoppingAmplitudeSet->getCOOValues();

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

	int totalMemoryRequirement = hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(int);
	totalMemoryRequirement += hoppingAmplitudeSet->getBasisSize()*sizeof(int);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(int);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(complex<double>);
	totalMemoryRequirement += to.size()*numCoefficients*sizeof(complex<double>);
	totalMemoryRequirement += hoppingAmplitudeSet->getBasisSize()*sizeof(int);
	if(damping != NULL)
		totalMemoryRequirement += hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>);
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA memory requirement: ";
		if(totalMemoryRequirement < 1024)
			Streams::out << totalMemoryRequirement/1024 << "B\n";
		else if(totalMemoryRequirement < 1024*1024)
			Streams::out << totalMemoryRequirement/1024 << "KB\n";
		else
			Streams::out << totalMemoryRequirement/1024/1024 << "MB\n";
	}

	TBTKAssert(
		cudaMalloc(
			(void**)&jIn1_device,
			hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>)
		) == cudaSuccess,
		"ChebyshevSOlver::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating jIn1_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&jIn2_device,
			hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>)
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating jIn2_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&cooHARowIndices_device,
			numHoppingAmplitudes*sizeof(int)
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating cooHARowIndices_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&csrHARowIndices_device,
			(hoppingAmplitudeSet->getBasisSize()+1)*sizeof(int)
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating csrHARowIndices_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&cooHAColIndices_device,
			numHoppingAmplitudes*sizeof(int)
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating cooHAColIndices_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&cooHAValues_device,
			numHoppingAmplitudes*sizeof(complex<double>)
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating cooHAValues_device.",
		""
	)
	TBTKAssert(
		cudaMalloc(
			(void**)&coefficients_device,
			to.size()*numCoefficients*sizeof(complex<double>)
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating coefficients_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&coefficientMap_device,
			hoppingAmplitudeSet->getBasisSize()*sizeof(int)
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA malloc error while allocating coefficientMap_device.",
		""
	);
	if(damping != NULL){
		TBTKAssert(
			cudaMalloc(
				(void**)&damping_device,
				hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>)
			) == cudaSuccess,
			"ChebyshevSolver::calculateCoefficientsGPU()",
			"CUDA malloc error while allocating damping_device.",
			""
		);
	}

	TBTKAssert(
		cudaMemcpy(
			jIn1_device,
			jIn1,
			hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying jIn1.",
		""
	);
	TBTKAssert(
		cudaMemcpy(
			jIn2_device,
			jIn2,
			hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
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
		"ChebyshevSolver::calculateCoefficientsGPU()",
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
		"ChebyshevSolver::calculateCoefficients()",
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
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying cooHAValues.",
		""
	);
	TBTKAssert(
		cudaMemcpy(
			coefficients_device,
			coefficients,
			to.size()*numCoefficients*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficients()",
		"CUDA memcpy error while copying coefficients.",
		""
	)
	TBTKAssert(
		cudaMemcpy(
			coefficientMap_device,
			coefficientMap,
			hoppingAmplitudeSet->getBasisSize()*sizeof(int),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying coefficientMap.",
		""
	);
	if(damping != NULL){
		TBTKAssert(
			cudaMemcpy(
				damping_device,
				damping,
				hoppingAmplitudeSet->getBasisSize()*sizeof(complex<double>),
				cudaMemcpyHostToDevice
			) == cudaSuccess,
			"ChebyshevSolver::calculateCoefficientsGPU()",
			"CUDA memcpy error while copying damping.",
			""
		);
	}

	cusparseHandle_t handle = NULL;
	TBTKAssert(
		cusparseCreate(&handle) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"cuSPARSE create error.",
		""
	);

	cusparseMatDescr_t descr = NULL;
	TBTKAssert(
		cusparseCreateMatDescr(&descr) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"cuSPARSE create matrix descriptor error.",
		""
	);

	TBTKAssert(
		cusparseSetMatType(
			descr,
			CUSPARSE_MATRIX_TYPE_GENERAL
		) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"cuSPARSE set matrix type error.",
		""
	);
	TBTKAssert(
		cusparseSetMatIndexBase(
			descr,
			CUSPARSE_INDEX_BASE_ZERO
		) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"cuSPARSE set matrix index base error.",
		""
	);

	TBTKAssert(
		cusparseXcoo2csr(
			handle,
			cooHARowIndices_device,
			numHoppingAmplitudes,
			hoppingAmplitudeSet->getBasisSize(),
			csrHARowIndices_device,
			CUSPARSE_INDEX_BASE_ZERO
		) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"cuSPARSE COO to CSR error.",
		""
	);

	//Calculate |j1>
	int block_size = 1024;
	int num_blocks = hoppingAmplitudeSet->getBasisSize()/block_size + (hoppingAmplitudeSet->getBasisSize()%block_size == 0 ? 0:1);
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA Block size: " << block_size << "\n";
		Streams::out << "\tCUDA Num blocks: " << num_blocks << "\n";
	}

	complex<double> multiplier = one/scaleFactor;
	TBTKAssert(
		cusparseZcsrmv(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			hoppingAmplitudeSet->getBasisSize(),
			hoppingAmplitudeSet->getBasisSize(),
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
		"ChebyshevSolver::calculateCoefficentsGPU()",
		"Matrix-vector multiplication error.",
		""
	);

	extractCoefficients <<< num_blocks, block_size >>> ((cuDoubleComplex*)jIn2_device,
								hoppingAmplitudeSet->getBasisSize(),
								(cuDoubleComplex*)coefficients_device,
								1,
								coefficientMap_device,
								numCoefficients);
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
				hoppingAmplitudeSet->getBasisSize(),
				hoppingAmplitudeSet->getBasisSize(),
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
			"ChebyshevSolver::calculateCoefficientsGPU()",
			"Matrix-vector multiplication error.",
			""
		);

		extractCoefficients <<< num_blocks, block_size >>> ((cuDoubleComplex*)jIn2_device,
									hoppingAmplitudeSet->getBasisSize(),
									(cuDoubleComplex*)coefficients_device,
									n,
									coefficientMap_device,
									numCoefficients);

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

	TBTKAssert(
		cudaMemcpy(
			coefficients,
			coefficients_device,
			to.size()*numCoefficients*sizeof(complex<double>),
			cudaMemcpyDeviceToHost
		) == cudaSuccess,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"CUDA memcpy error while copying coefficients.",
		""
	);

	TBTKAssert(
		cusparseDestroyMatDescr(descr) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"cuSPARSE destroy matrix descriptor error.",
		""
	);
	descr = NULL;

	TBTKAssert(
		cusparseDestroy(handle) == CUSPARSE_STATUS_SUCCESS,
		"ChebyshevSolver::calculateCoefficientsGPU()",
		"cuSPARSE destroy error.",
		""
	);
	handle = NULL;

	delete [] jIn1;
	delete [] jIn2;
	delete [] coefficientMap;

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
	double lambda = broadening*numCoefficients;
	for(int n = 0; n < numCoefficients; n++)
		for(int c = 0; c < to.size(); c++)
			coefficients[n + c*numCoefficients] = coefficients[n + c*numCoefficients]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
}

__global__
void calculateGreensFunction(
	cuDoubleComplex *greensFunction,
	cuDoubleComplex *coefficients,
	cuDoubleComplex *lookupTable,
	int numCoefficients,
	int energyResolution
){
	int e = blockIdx.x*blockDim.x + threadIdx.x;
	if(e < energyResolution)
		for(int n = 0; n < numCoefficients; n++)
			greensFunction[e] = cuCadd(greensFunction[e], cuCmul(lookupTable[n*energyResolution + e], coefficients[n]));
}

void ChebyshevSolver::loadLookupTableGPU(){
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "CheyshevSolver::loadLookupTableGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable != NULL,
		"ChebyshevSolver::loadLookupTableGPU()",
		"Lookup table has not been generated.",
		"Call ChebyshevSolver::generateLokupTable() to generate lookup table."
	);
	if(generatingFunctionLookupTable_device != NULL)
		destroyLookupTableGPU();

	complex<double> *generatingFunctionLookupTable_host = new complex<double>[lookupTableNumCoefficients*lookupTableResolution];
	for(int n = 0; n < lookupTableNumCoefficients; n++)
		for(int e = 0; e < lookupTableResolution; e++)
			generatingFunctionLookupTable_host[n*lookupTableResolution + e] = generatingFunctionLookupTable[n][e];

	int memoryRequirement = lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>);
	if(getGlobalVerbose() && getVerbose()){
		Streams::out << "\tCUDA memory requirement: ";
		if(memoryRequirement < 1024)
			Streams::out << memoryRequirement << "B\n";
		else if(memoryRequirement < 1024*1024)
			Streams::out << memoryRequirement/1024 << "KB\n";
		else
			Streams::out << memoryRequirement/1024/1024 << "MB\n";
	}

	generatingFunctionLookupTable_device = new complex<double>**[GPUResourceManager::getInstance().getNumDevices()];

	for(int n = 0; n < GPUResourceManager::getInstance().getNumDevices(); n++){
		TBTKAssert(
			cudaSetDevice(n) == cudaSuccess,
			"ChebyshevSolver::loadLookupTableGPU()",
			"CUDA set device error for device " << n << ".",
			""
		);

		TBTKAssert(
			cudaMalloc(
				(void**)&generatingFunctionLookupTable_device[n],
				lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>)
			)  == cudaSuccess,
			"ChebyshevSolver::loadLookupTableGPU()",
			"CUDA malloc error while allocating generatingFunctionLookupTable_device.",
			""
		);

		TBTKAssert(
			cudaMemcpy(
				generatingFunctionLookupTable_device[n],
				generatingFunctionLookupTable_host,
				lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>),
				cudaMemcpyHostToDevice
			) == cudaSuccess,
			"ChebyshevSolver::loadLookupTableGPU()",
			"CUDA memcpy error while copying generatingFunctionLookupTable_device.",
			""
		);
	}

	delete [] generatingFunctionLookupTable_host;
}

void ChebyshevSolver::destroyLookupTableGPU(){
	if(getGlobalVerbose() && getVerbose())
		Streams::out << "ChebyshevSolver::destroyLookupTableGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable_device != NULL,
		"ChebyshevSolver::destroyLookupTableGPU()",
		"No lookup table loaded onto GPU.\n",
		""
	);

	for(int n = 0; n < GPUResourceManager::getInstance().getNumDevices(); n++){
		cudaFree(generatingFunctionLookupTable_device[n]);
	}

	delete [] generatingFunctionLookupTable_device;
	generatingFunctionLookupTable_device = NULL;
}

Property::GreensFunction* ChebyshevSolver::generateGreensFunctionGPU(
	complex<double> *coefficients,
	Property::GreensFunction::Type type
){
	int device = GPUResourceManager::getInstance().allocateDevice();

	TBTKAssert(
		cudaSetDevice(device) == cudaSuccess,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"CUDA set device error for device " << device << ".",
		""
	);

	if(getGlobalVerbose() && getVerbose())
		Streams::out << "ChebyshevSolver::generateGreensFunctionGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable_device != NULL,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"No lookup table loaded onto GPU.",
		""
	);
	TBTKAssert(
		type == Property::GreensFunction::Type::Retarded,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"Only evaluation of retarded Green's function is implemented for GPU so far.",
		"Use CPU evaluation instead."
	);

	complex<double> *greensFunctionData = new complex<double>[lookupTableResolution];

	for(int e = 0; e < lookupTableResolution; e++)
		greensFunctionData[e] = 0.;

	complex<double> *greensFunctionData_device;
	complex<double> *coefficients_device;

	TBTKAssert(
		cudaMalloc(
			(void**)&greensFunctionData_device,
			lookupTableResolution*sizeof(complex<double>)
		)  == cudaSuccess,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"CUDA malloc error while allocating greensFunction_device.",
		""
	);
	TBTKAssert(
		cudaMalloc(
			(void**)&coefficients_device,
			lookupTableNumCoefficients*sizeof(complex<double>)
		)  == cudaSuccess,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"CUDA malloc error while allocating coefficients_device.",
		""
	);

	TBTKAssert(
		cudaMemcpy(
			greensFunctionData_device,
			greensFunctionData,
			lookupTableResolution*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"CUDA memcpy error while copying greensFunctionData.",
		""
	);
	TBTKAssert(
		cudaMemcpy(
			coefficients_device,
			coefficients,
			lookupTableNumCoefficients*sizeof(complex<double>),
			cudaMemcpyHostToDevice
		) == cudaSuccess,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"CUDA memcpy error while copying coefficients.",
		""
	);

	int block_size = 1024;
	int num_blocks = lookupTableResolution/block_size + (lookupTableResolution%block_size == 0 ? 0:1);

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
			greensFunctionData,
			greensFunctionData_device,
			lookupTableResolution*sizeof(complex<double>),
			cudaMemcpyDeviceToHost
		) == cudaSuccess,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"CUDA memcpy error while copying greensFunction_device.",
		""
	);

	cudaFree(greensFunctionData_device);
	cudaFree(coefficients_device);

	GPUResourceManager::getInstance().freeDevice(device);

	Property::GreensFunction *greensFunction = new Property::GreensFunction(
		type,
		Property::GreensFunction::Format::Array,
		lookupTableLowerBound,
		lookupTableUpperBound,
		lookupTableResolution,
		greensFunctionData
	);
	delete [] greensFunctionData;

	return greensFunction;
}

};	//End of namespace TBTK
