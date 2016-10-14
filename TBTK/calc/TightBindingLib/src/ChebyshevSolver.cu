/** @file ChebyshevSolver.cu
 *
 *  @author Kristofer Björnson
 */

#include "../include/ChebyshevSolver.h"
#include "../include/HALinkedList.h"
#include "../include/Util.h"
#include "../include/TBTKMacros.h"

#include <math.h>

#include <cuComplex.h>
#include <cusparse_v2.h>

using namespace std;

namespace TBTK{

complex<double> minus_one(-1., 0.);
complex<double> one(1., 0.);
complex<double> two(2., 0.);
complex<double> zero(0., 0.);
complex<double> i(0., 1.);

void cusparseSafe(cusparseStatus_t type, string message){
	if(type != CUSPARSE_STATUS_SUCCESS){
		cout << "\t" << message << "\n";
		exit(1);
	}
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

	int device = allocateDeviceGPU();

	if(cudaSetDevice(device) != cudaSuccess)
		{	cout << "\tSet device error: " << device << "\n";	exit(1);	}

	AmplitudeSet *amplitudeSet = model->getAmplitudeSet();

	int fromBasisIndex = amplitudeSet->getBasisIndex(from);
	int *coefficientMap = new int[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		coefficientMap[n] = -1;
	for(int n = 0; n < to.size(); n++)
		coefficientMap[amplitudeSet->getBasisIndex(to.at(n))] = n;

	if(isTalkative){
		cout << "ChebyshevSolver::calculateCoefficientsGPU\n";
		cout << "\tFrom Index: " << fromBasisIndex << "\n";
		cout << "\tBasis size: " << amplitudeSet->getBasisSize() << "\n";
		cout << "\tUsing damping: ";
		if(damping != NULL)
			cout << "Yes\n";
		else
			cout << "No\n";
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
		cout << "\tCUDA memory requirement: ";
		if(totalMemoryRequirement < 1024)
			cout << totalMemoryRequirement/1024 << "B\n";
		else if(totalMemoryRequirement < 1024*1024)
			cout << totalMemoryRequirement/1024 << "KB\n";
		else
			cout << totalMemoryRequirement/1024/1024 << "MB\n";
	}

	if(cudaMalloc((void**)&jIn1_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "\tMalloc error: jIn1_device\n";		exit(1);	}
	if(cudaMalloc((void**)&jIn2_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "\tMalloc error: jIn2_device\n";		exit(1);	}
	if(cudaMalloc((void**)&cooHARowIndices_device, numHoppingAmplitudes*sizeof(int)) != cudaSuccess)
		{	cout << "\tMalloc error: cooHARowIndices_device\n";	exit(1);	}
	if(cudaMalloc((void**)&csrHARowIndices_device, (amplitudeSet->getBasisSize()+1)*sizeof(int)) != cudaSuccess)
		{	cout << "\tMalloc error: csrHARowIndices_device\n";	exit(1);	}
	if(cudaMalloc((void**)&cooHAColIndices_device, numHoppingAmplitudes*sizeof(int)) != cudaSuccess)
		{	cout << "\tMalloc error: cooHAColIndices_device\n";	exit(1);	}
	if(cudaMalloc((void**)&cooHAValues_device, numHoppingAmplitudes*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "\tMalloc error: cooHAValues_device\n";		exit(1);	}
	if(cudaMalloc((void**)&coefficients_device, to.size()*numCoefficients*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "\tMalloc error: coefficients_device\n";	exit(1);	}
	if(cudaMalloc((void**)&coefficientMap_device, amplitudeSet->getBasisSize()*sizeof(int)) != cudaSuccess)
		{	cout << "\tMalloc error: coefficientMap_device\n";	exit(1);	}
	if(damping != NULL){
		if(cudaMalloc((void**)&damping_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
			{	cout << "\tMalloc error: damping_device\n";	exit(1);	}
	}

	if(cudaMemcpy(jIn1_device, jIn1, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: jIn1\n";		exit(1);	}
	if(cudaMemcpy(jIn2_device, jIn2, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: jIn2\n";		exit(1);	}
	if(cudaMemcpy(cooHARowIndices_device, cooHARowIndices_host, numHoppingAmplitudes*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: cooHARowIndices\n";	exit(1);	}
	if(cudaMemcpy(cooHAColIndices_device, cooHAColIndices_host, numHoppingAmplitudes*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: cooHAColIndices\n";	exit(1);	}
	if(cudaMemcpy(cooHAValues_device, cooHAValues_host, numHoppingAmplitudes*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: cooHAValues\n";	exit(1);	}
	if(cudaMemcpy(coefficients_device, coefficients, to.size()*numCoefficients*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: coefficients\n";	exit(1);	}
	if(cudaMemcpy(coefficientMap_device, coefficientMap, amplitudeSet->getBasisSize()*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: coefficientMap\n";	exit(1);	}
	if(damping != NULL){
		if(cudaMemcpy(damping_device, damping, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
			{	cout << "\tMemcpy error: damping\n";	exit(1);	}
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
		cout << "\tCUDA Block size: " << block_size << "\n";
		cout << "\tCUDA Num blocks: " << num_blocks << "\n";
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
		cout << "\tProgress (100 coefficients per dot): ";

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
				cout << "." << flush;
			if(n%1000 == 0)
				cout << " " << flush;
		}
	}
	if(isTalkative)
		cout << "\n";

	if(cudaMemcpy(coefficients, coefficients_device, to.size()*numCoefficients*sizeof(complex<double>), cudaMemcpyDeviceToHost) != cudaSuccess){
		cout << "\tMemcpy error: coefficients\n";
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

	freeDeviceGPU(device);

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
		cout << "CheyshevSolver::loadLookupTableGPU\n";

	if(generatingFunctionLookupTable == NULL){
		cout << "\tError: Lookup table has not been generated.\n";
		exit(1);
	}
	if(generatingFunctionLookupTable_device != NULL){
		cout << "\tError: Lookup table already loaded.\n";
		exit(1);
	}

	complex<double> *generatingFunctionLookupTable_host = new complex<double>[lookupTableNumCoefficients*lookupTableResolution];
	for(int n = 0; n < lookupTableNumCoefficients; n++)
		for(int e = 0; e < lookupTableResolution; e++)
			generatingFunctionLookupTable_host[n*lookupTableResolution + e] = generatingFunctionLookupTable[n][e];

	int memoryRequirement = lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>);
	if(isTalkative){
		cout << "\tCUDA memory requirement: ";
		if(memoryRequirement < 1024)
			cout << memoryRequirement << "B\n";
		else if(memoryRequirement < 1024*1024)
			cout << memoryRequirement/1024 << "KB\n";
		else
			cout << memoryRequirement/1024/1024 << "MB\n";
	}

	generatingFunctionLookupTable_device = new complex<double>**[numDevices];

	for(int n = 0; n < numDevices; n++){
		if(cudaSetDevice(n) != cudaSuccess)
			{	cout << "\tSet device error: " << n << "\n";	exit(1);	}

		if(cudaMalloc((void**)&generatingFunctionLookupTable_device[n], lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>))  != cudaSuccess)
			{	cout << "\tMalloc error: generatingFunctionLookupTable_device\n";	exit(1);	}

		if(cudaMemcpy(generatingFunctionLookupTable_device[n], generatingFunctionLookupTable_host, lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
			{	cout << "\tMemcpy error: generatingFunctionLookupTable_device\n";	exit(1);	}
	}

	delete [] generatingFunctionLookupTable_host;
}

void ChebyshevSolver::destroyLookupTableGPU(){
	if(isTalkative)
		cout << "ChebyshevSolver::destroyLookupTableGPU\n";

	if(generatingFunctionLookupTable_device == NULL){
		cout << "Error: No lookup table loaded onto GPU.\n";
		exit(1);
	}

	for(int n = 0; n < numDevices; n++){
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
	int device = allocateDeviceGPU();

	if(cudaSetDevice(device) != cudaSuccess)
		{	cout << "\tSet device error: " << device << "\n";	exit(1);	}

	if(isTalkative)
		cout << "ChebyshevSolver::generateGreensFunctionGPU\n";

	if(generatingFunctionLookupTable_device == NULL){
		cout << "Error: No lookup table loaded onto GPU.\n";
		exit(1);
	}

	if(type != GreensFunctionType::Retarded){
		cout << "Error: Only evaluation of retarded Green's function is implemented for GPU so far. Use CPU evaluation instead.\n";
		exit(1);
	}

	for(int e = 0; e < lookupTableResolution; e++)
		greensFunction[e] = 0.;

	complex<double> *greensFunction_device;
	complex<double> *coefficients_device;

	if(cudaMalloc((void**)&greensFunction_device, lookupTableResolution*sizeof(complex<double>))  != cudaSuccess)
		{	cout << "\tMalloc error: greensFunction_device\n";	exit(1);	}
	if(cudaMalloc((void**)&coefficients_device, lookupTableNumCoefficients*sizeof(complex<double>))  != cudaSuccess)
		{	cout << "\tMalloc error: coefficients_device\n";	exit(1);	}

	if(cudaMemcpy(greensFunction_device, greensFunction, lookupTableResolution*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: greensFunction\n";	exit(1);	}
	if(cudaMemcpy(coefficients_device, coefficients, lookupTableNumCoefficients*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error: coefficients\n";	exit(1);	}

	int block_size = 1024;
	int num_blocks = lookupTableResolution/block_size + (lookupTableResolution%block_size == 0 ? 0:1);

	if(isTalkative){
		cout << "\tCUDA Block size: " << block_size << "\n";
		cout << "\tCUDA Num blocks: " << num_blocks << "\n";
	}

	calculateGreensFunction <<< num_blocks, block_size>>> ((cuDoubleComplex*)greensFunction_device,
								(cuDoubleComplex*)coefficients_device,
								(cuDoubleComplex*)generatingFunctionLookupTable_device[device],
								lookupTableNumCoefficients,
								lookupTableResolution);

	if(cudaMemcpy(greensFunction, greensFunction_device, lookupTableResolution*sizeof(complex<double>), cudaMemcpyDeviceToHost) != cudaSuccess)
		{	cout << "\tMemcpy error: greensFunction_device\n";	exit(1);	}

	cudaFree(greensFunction_device);
	cudaFree(coefficients_device);

	freeDeviceGPU(device);
}

void ChebyshevSolver::createDeviceTableGPU(){
	cudaGetDeviceCount(&numDevices);

	cout << "Num GPU devices: " << numDevices << "\n";

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
}

};	//End of namespace TBTK
