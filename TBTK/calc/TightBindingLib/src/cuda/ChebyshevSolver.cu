/** @file ChebyshevSolver.cu
 *
 *  @author Kristofer Björnson
 */

#include "../../include/ChebyshevSolver.h"
#include "../../include/HALinkedList.h"
#include "../../include/Util.h"
#include "../../include/TBTKMacros.h"
#include "../../include/GPUResourceManager.h"
#include "../../include/Streams.h"

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

void cusparseSafe(cusparseStatus_t type, string message){
	TBTKAssert(
		type == CUSPARSE_STATUS_SUCCESS,
		"cusparseSafe",
		message,
		""
	);
}

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
		model != NULL,
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

//	int device = allocateDeviceGPU();
	int device = GPUResourceManager::getInstance().allocateDevice();

	if(cudaSetDevice(device) != cudaSuccess)
		{	Util::Streams::err << "\tSet device error: " << device << "\n";	Util::Streams::closeLog();	exit(1);	}

	AmplitudeSet *amplitudeSet = model->getAmplitudeSet();

	int fromBasisIndex = amplitudeSet->getBasisIndex(from);
	int *coefficientMap = new int[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		coefficientMap[n] = -1;
	for(int n = 0; n < to.size(); n++)
		coefficientMap[amplitudeSet->getBasisIndex(to.at(n))] = n;

	if(isTalkative){
		Util::Streams::out << "ChebyshevSolver::calculateCoefficientsGPU\n";
		Util::Streams::out << "\tFrom Index: " << fromBasisIndex << "\n";
		Util::Streams::out << "\tBasis size: " << amplitudeSet->getBasisSize() << "\n";
		Util::Streams::out << "\tUsing damping: ";
		if(damping != NULL)
			Util::Streams::out << "Yes\n";
		else
			Util::Streams::out << "No\n";
	}

	complex<double> *jIn1 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
	}

	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]*numCoefficients] = jIn1[n];

	const int numHoppingAmplitudes = amplitudeSet->getNumMatrixElements();
	const int *cooHARowIndices_host = amplitudeSet->getCOORowIndices();
	const int *cooHAColIndices_host = amplitudeSet->getCOOColIndices();
	const complex<double> *cooHAValues_host = amplitudeSet->getCOOValues();

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

	int totalMemoryRequirement = amplitudeSet->getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += amplitudeSet->getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(int);
	totalMemoryRequirement += amplitudeSet->getBasisSize()*sizeof(int);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(int);
	totalMemoryRequirement += numHoppingAmplitudes*sizeof(complex<double>);
	totalMemoryRequirement += to.size()*numCoefficients*sizeof(complex<double>);
	totalMemoryRequirement += amplitudeSet->getBasisSize()*sizeof(int);
	if(damping != NULL)
		totalMemoryRequirement += amplitudeSet->getBasisSize()*sizeof(complex<double>);
	if(isTalkative){
		Util::Streams::out << "\tCUDA memory requirement: ";
		if(totalMemoryRequirement < 1024)
			Util::Streams::out << totalMemoryRequirement/1024 << "B\n";
		else if(totalMemoryRequirement < 1024*1024)
			Util::Streams::out << totalMemoryRequirement/1024 << "KB\n";
		else
			Util::Streams::out << totalMemoryRequirement/1024/1024 << "MB\n";
	}

	if(cudaMalloc((void**)&jIn1_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: jIn1_device\n";		Util::Streams::closeLog();	exit(1);	}
	if(cudaMalloc((void**)&jIn2_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: jIn2_device\n";		Util::Streams::closeLog();	exit(1);	}
	if(cudaMalloc((void**)&cooHARowIndices_device, numHoppingAmplitudes*sizeof(int)) != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: cooHARowIndices_device\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMalloc((void**)&csrHARowIndices_device, (amplitudeSet->getBasisSize()+1)*sizeof(int)) != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: csrHARowIndices_device\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMalloc((void**)&cooHAColIndices_device, numHoppingAmplitudes*sizeof(int)) != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: cooHAColIndices_device\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMalloc((void**)&cooHAValues_device, numHoppingAmplitudes*sizeof(complex<double>)) != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: cooHAValues_device\n";		Util::Streams::closeLog();	exit(1);	}
	if(cudaMalloc((void**)&coefficients_device, to.size()*numCoefficients*sizeof(complex<double>)) != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: coefficients_device\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMalloc((void**)&coefficientMap_device, amplitudeSet->getBasisSize()*sizeof(int)) != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: coefficientMap_device\n";	Util::Streams::closeLog();	exit(1);	}
	if(damping != NULL){
		if(cudaMalloc((void**)&damping_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
			{	Util::Streams::err << "\tMalloc error: damping_device\n";	Util::Streams::closeLog();	exit(1);	}
	}

	if(cudaMemcpy(jIn1_device, jIn1, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: jIn1\n";		Util::Streams::closeLog();	exit(1);	}
	if(cudaMemcpy(jIn2_device, jIn2, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: jIn2\n";		Util::Streams::closeLog();	exit(1);	}
	if(cudaMemcpy(cooHARowIndices_device, cooHARowIndices_host, numHoppingAmplitudes*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: cooHARowIndices\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMemcpy(cooHAColIndices_device, cooHAColIndices_host, numHoppingAmplitudes*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: cooHAColIndices\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMemcpy(cooHAValues_device, cooHAValues_host, numHoppingAmplitudes*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: cooHAValues\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMemcpy(coefficients_device, coefficients, to.size()*numCoefficients*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: coefficients\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMemcpy(coefficientMap_device, coefficientMap, amplitudeSet->getBasisSize()*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: coefficientMap\n";	Util::Streams::closeLog();	exit(1);	}
	if(damping != NULL){
		if(cudaMemcpy(damping_device, damping, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
			{	Util::Streams::err << "\tMemcpy error: damping\n";	Util::Streams::closeLog();	exit(1);	}
	}

	cusparseHandle_t handle = NULL;
	cusparseSafe(cusparseCreate(&handle), "cuSPARSE create error");

	cusparseMatDescr_t descr = NULL;
	cusparseSafe(cusparseCreateMatDescr(&descr), "cuSPARSE create matrix descriptor error");

	cusparseSafe(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL), "cuSPARSE set matrix type error");
	cusparseSafe(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO), "cuSPARSE set matrix index base error");

	cusparseSafe(cusparseXcoo2csr(handle,
					cooHARowIndices_device,
					numHoppingAmplitudes,
					amplitudeSet->getBasisSize(),
					csrHARowIndices_device,
					CUSPARSE_INDEX_BASE_ZERO),
			"cuSPARSE COO to CSR error");

	//Calculate |j1>
	int block_size = 1024;
	int num_blocks = amplitudeSet->getBasisSize()/block_size + (amplitudeSet->getBasisSize()%block_size == 0 ? 0:1);
	if(isTalkative){
		Util::Streams::out << "\tCUDA Block size: " << block_size << "\n";
		Util::Streams::out << "\tCUDA Num blocks: " << num_blocks << "\n";
	}

	complex<double> multiplier = one/scaleFactor;
	cusparseSafe(cusparseZcsrmv(handle,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					amplitudeSet->getBasisSize(),
					amplitudeSet->getBasisSize(),
					numHoppingAmplitudes,
					(cuDoubleComplex*)&multiplier,
					descr,
					(cuDoubleComplex*)cooHAValues_device,
					csrHARowIndices_device,
					cooHAColIndices_device,
					(cuDoubleComplex*)jIn1_device,
					(cuDoubleComplex*)&zero,
					(cuDoubleComplex*)jIn2_device),
			"Matrix-vector multiplication error");

	extractCoefficients <<< num_blocks, block_size >>> ((cuDoubleComplex*)jIn2_device,
								amplitudeSet->getBasisSize(),
								(cuDoubleComplex*)coefficients_device,
								1,
								coefficientMap_device,
								numCoefficients);
	jTemp = jIn2_device;
	jIn2_device = jIn1_device;
	jIn1_device = jTemp;

	if(isTalkative)
		Util::Streams::out << "\tProgress (100 coefficients per dot): ";

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		multiplier = two/scaleFactor;
		cusparseSafe(cusparseZcsrmv(handle,
						CUSPARSE_OPERATION_NON_TRANSPOSE,
						amplitudeSet->getBasisSize(),
						amplitudeSet->getBasisSize(),
						numHoppingAmplitudes,
						(cuDoubleComplex*)&multiplier,
						descr,
						(cuDoubleComplex*)cooHAValues_device,
						csrHARowIndices_device,
						cooHAColIndices_device,
						(cuDoubleComplex*)jIn1_device,
						(cuDoubleComplex*)&minus_one,
						(cuDoubleComplex*)jIn2_device),
				"Matrix-vector multiplication error");

		extractCoefficients <<< num_blocks, block_size >>> ((cuDoubleComplex*)jIn2_device,
									amplitudeSet->getBasisSize(),
									(cuDoubleComplex*)coefficients_device,
									n,
									coefficientMap_device,
									numCoefficients);

		jTemp = jIn2_device;
		jIn2_device = jIn1_device;
		jIn1_device = jTemp;

		if(isTalkative){
			if(n%100 == 0)
				Util::Streams::out << "." << flush;
			if(n%1000 == 0)
				Util::Streams::out << " " << flush;
		}
	}
	if(isTalkative)
		Util::Streams::out << "\n";

	if(cudaMemcpy(coefficients, coefficients_device, to.size()*numCoefficients*sizeof(complex<double>), cudaMemcpyDeviceToHost) != cudaSuccess){
		Util::Streams::err << "\tMemcpy error: coefficients\n";
		Util::Streams::closeLog();
		exit(1);
	}

	cusparseSafe(cusparseDestroyMatDescr(descr), "cuSPARSE destroy matrix descriptor error");
	descr = NULL;
	cusparseSafe(cusparseDestroy(handle), "cuSPARSE destroy error");
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

//	freeDeviceGPU(device);
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
//			greensFunction[e] += lookupTable[n*energyResolution + e]*coefficients[n];
}

void ChebyshevSolver::loadLookupTableGPU(){
	if(isTalkative)
		Util::Streams::out << "CheyshevSolver::loadLookupTableGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable != NULL,
		"ChebyshevSolver::loadLookupTableGPU()",
		"Lookup table has not been generated.",
		"Call ChebyshevSolver::generateLokupTable() to generate lookup table."
	);
	TBTKAssert(
		generatingFunctionLookupTable_device == NULL,
		"ChebyshevSolver::loadLookupTableGPU()",
		"Lookup table already loaded.",
		""
	);

	complex<double> *generatingFunctionLookupTable_host = new complex<double>[lookupTableNumCoefficients*lookupTableResolution];
	for(int n = 0; n < lookupTableNumCoefficients; n++)
		for(int e = 0; e < lookupTableResolution; e++)
			generatingFunctionLookupTable_host[n*lookupTableResolution + e] = generatingFunctionLookupTable[n][e];

	int memoryRequirement = lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>);
	if(isTalkative){
		Util::Streams::out << "\tCUDA memory requirement: ";
		if(memoryRequirement < 1024)
			Util::Streams::out << memoryRequirement << "B\n";
		else if(memoryRequirement < 1024*1024)
			Util::Streams::out << memoryRequirement/1024 << "KB\n";
		else
			Util::Streams::out << memoryRequirement/1024/1024 << "MB\n";
	}

//	generatingFunctionLookupTable_device = new complex<double>**[numDevices];
	generatingFunctionLookupTable_device = new complex<double>**[GPUResourceManager::getInstance().getNumDevices()];

//	for(int n = 0; n < numDevices; n++){
	for(int n = 0; n < GPUResourceManager::getInstance().getNumDevices(); n++){
		if(cudaSetDevice(n) != cudaSuccess)
			{	Util::Streams::err << "\tSet device error: " << n << "\n";	Util::Streams::closeLog();	exit(1);	}

		if(cudaMalloc((void**)&generatingFunctionLookupTable_device[n], lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>))  != cudaSuccess)
			{	Util::Streams::err << "\tMalloc error: generatingFunctionLookupTable_device\n";	Util::Streams::closeLog();	exit(1);	}

		if(cudaMemcpy(generatingFunctionLookupTable_device[n], generatingFunctionLookupTable_host, lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
			{	Util::Streams::err << "\tMemcpy error: generatingFunctionLookupTable_device\n";	Util::Streams::closeLog();	exit(1);	}
	}

	delete [] generatingFunctionLookupTable_host;
}

void ChebyshevSolver::destroyLookupTableGPU(){
	if(isTalkative)
		Util::Streams::out << "ChebyshevSolver::destroyLookupTableGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable_device != NULL,
		"ChebyshevSolver::destroyLookupTableGPU()",
		"No lookup table loaded onto GPU.\n",
		""
	);

//	for(int n = 0; n < numDevices; n++){
	for(int n = 0; n < GPUResourceManager::getInstance().getNumDevices(); n++){
		cudaFree(generatingFunctionLookupTable_device[n]);
	}

	delete [] generatingFunctionLookupTable_device;
	generatingFunctionLookupTable_device = NULL;
}

void ChebyshevSolver::generateGreensFunctionGPU(
	complex<double> *greensFunction,
	complex<double> *coefficients,
	GreensFunctionType type
){
//	int device = allocateDeviceGPU();
	int device = GPUResourceManager::getInstance().allocateDevice();

	if(cudaSetDevice(device) != cudaSuccess)
		{	Util::Streams::err << "\tSet device error: " << device << "\n";	Util::Streams::closeLog();	exit(1);	}

	if(isTalkative)
		Util::Streams::out << "ChebyshevSolver::generateGreensFunctionGPU\n";

	TBTKAssert(
		generatingFunctionLookupTable_device != NULL,
		"ChebyshevSolver::generateGreensFunctionGPU()",
		"No lookup table loaded onto GPU.",
		""
	);
	TBTKAssert(
		type == GreensFunctionType::Retarded,
		"ChebyshevSolver::generateGreensFunctionGPU",
		"Only evaluation of retarded Green's function is implemented for GPU so far.",
		"Use CPU evaluation instead."
	);

	for(int e = 0; e < lookupTableResolution; e++)
		greensFunction[e] = 0.;

	complex<double> *greensFunction_device;
	complex<double> *coefficients_device;

	if(cudaMalloc((void**)&greensFunction_device, lookupTableResolution*sizeof(complex<double>))  != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: greensFunction_device\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMalloc((void**)&coefficients_device, lookupTableNumCoefficients*sizeof(complex<double>))  != cudaSuccess)
		{	Util::Streams::err << "\tMalloc error: coefficients_device\n";	Util::Streams::closeLog();	exit(1);	}

	if(cudaMemcpy(greensFunction_device, greensFunction, lookupTableResolution*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: greensFunction\n";	Util::Streams::closeLog();	exit(1);	}
	if(cudaMemcpy(coefficients_device, coefficients, lookupTableNumCoefficients*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: coefficients\n";	Util::Streams::closeLog();	exit(1);	}

	int block_size = 1024;
	int num_blocks = lookupTableResolution/block_size + (lookupTableResolution%block_size == 0 ? 0:1);

	if(isTalkative){
		Util::Streams::out << "\tCUDA Block size: " << block_size << "\n";
		Util::Streams::out << "\tCUDA Num blocks: " << num_blocks << "\n";
	}

	calculateGreensFunction <<< num_blocks, block_size>>> ((cuDoubleComplex*)greensFunction_device,
								(cuDoubleComplex*)coefficients_device,
								(cuDoubleComplex*)generatingFunctionLookupTable_device[device],
								lookupTableNumCoefficients,
								lookupTableResolution);

	if(cudaMemcpy(greensFunction, greensFunction_device, lookupTableResolution*sizeof(complex<double>), cudaMemcpyDeviceToHost) != cudaSuccess)
		{	Util::Streams::err << "\tMemcpy error: greensFunction_device\n";	Util::Streams::closeLog();	exit(1);	}

	cudaFree(greensFunction_device);
	cudaFree(coefficients_device);

//	freeDeviceGPU(device);
	GPUResourceManager::getInstance().freeDevice(device);
}

/*void ChebyshevSolver::createDeviceTableGPU(){
	cudaGetDeviceCount(&numDevices);

	Util::Streams::out << "Num GPU devices: " << numDevices << "\n";

	if(numDevices > 0){
		busyDevices = new bool[numDevices];
		for(int n = 0; n < numDevices; n++)
			busyDevices[n] = false;
	}
}

void ChebyshevSolver::destroyDeviceTableGPU(){
	if(numDevices > 0)
		delete [] busyDevices;
}

int ChebyshevSolver::allocateDeviceGPU(){
	int device = 0;
	bool done = false;
	while(!done){
		omp_set_lock(&busyDevicesLock);
		#pragma omp flush
		{
			for(int n = 0; n < numDevices; n++){
				if(!busyDevices[n]){
					device = n;
					busyDevices[n] = true;
					done = true;
					break;
				}
			}
		}
		#pragma omp flush
		omp_unset_lock(&busyDevicesLock);
	}

	return device;
}

void ChebyshevSolver::freeDeviceGPU(int device){
	omp_set_lock(&busyDevicesLock);
	#pragma omp flush
	{
		busyDevices[device] = false;
	}
	#pragma omp flush
	omp_unset_lock(&busyDevicesLock);
}*/

};	//End of namespace TBTK
