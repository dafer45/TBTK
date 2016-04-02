/** @file ChebyshevSolver.cu
 *
 *  @author Kristofer Björnson
 */

#include "../include/ChebyshevSolver.h"
#include <math.h>
#include "../include/HALinkedList.h"
#include <cuComplex.h>

using namespace std;

namespace TBTK{

__global__
void multiplyMatrixAndVector(cuDoubleComplex *jIn,
				cuDoubleComplex *jResult,
				cuDoubleComplex *hoppingAmplitudes,
				int *fromIndices,
				int maxHoppingAmplitudes,
				int basisSize,
				cuDoubleComplex *coefficients,
				int currentCoefficient,
				int *coefficientMap,
				int numCoefficients){
	int to = blockIdx.x*blockDim.x + threadIdx.x;
	if(to < basisSize)
		for(int n = 0; n < maxHoppingAmplitudes; n++)
			jResult[to] = cuCadd(jResult[to], cuCmul(hoppingAmplitudes[maxHoppingAmplitudes*to + n], jIn[fromIndices[maxHoppingAmplitudes*to + n]]));

/*	if(to == coefficientIndex)
		coefficients[currentCoefficient] = jResult[to];*/
	if(to < basisSize && coefficientMap[to] != -1)
		coefficients[coefficientMap[to]*numCoefficients + currentCoefficient] = jResult[to];
}

__global__
void subtractVector(cuDoubleComplex *jIn2, cuDoubleComplex *jResult, int basisSize){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < basisSize)
		jResult[idx] = make_cuDoubleComplex(-cuCreal(jIn2[idx]), -cuCimag(jIn2[idx]));
}

void debugCUDA(complex<double> *jIn1_device, complex<double> *jIn2_device, complex<double> *jResult_device, int basisSize){
	complex<double> *jIn1 = new complex<double>[basisSize];
	complex<double> *jIn2 = new complex<double>[basisSize];
	complex<double> *jResult = new complex<double>[basisSize];
	cudaMemcpy(jIn1, jIn1_device, basisSize*sizeof(complex<double>), cudaMemcpyDeviceToHost);
	cudaMemcpy(jIn2, jIn2_device, basisSize*sizeof(complex<double>), cudaMemcpyDeviceToHost);
	cudaMemcpy(jResult, jResult_device, basisSize*sizeof(complex<double>), cudaMemcpyDeviceToHost);
	for(int n = 0; n < basisSize; n++)
		cout << n << "\t" << jIn1[n] << "\t" << jIn2[n] << "\t" << jResult[n] << "\n";

	delete [] jIn1;
	delete [] jIn2;
	delete [] jResult;
}

void debugNormal(complex<double> *jIn1, complex<double> *jIn2, complex<double> *jResult, int basisSize){
	for(int n = 0; n < basisSize; n++)
		cout << n << "\t" << jIn1[n] << "\t" << jIn2[n] << "\t" << jResult[n] << "\n";
}

void ChebyshevSolver::calculateCoefficientsGPU(Index to, Index from, complex<double> *coefficients, int numCoefficients, double broadening){
	vector<Index> toVector;
	toVector.push_back(to);
	calculateCoefficientsGPU(toVector, from, coefficients, numCoefficients, broadening);
}

void ChebyshevSolver::calculateCoefficientsGPU(vector<Index> &to, Index from, complex<double> *coefficients, int numCoefficients, double broadening){
	AmplitudeSet *amplitudeSet = &model->amplitudeSet;

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
	}

	complex<double> *jIn1 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jIn2 = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jResult = new complex<double>[amplitudeSet->getBasisSize()];
	complex<double> *jTemp = NULL;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++){
		jIn1[n] = 0.;
		jIn2[n] = 0.;
		jResult[n] = 0.;
	}

	//Set up initial state (|j0>)
	jIn1[fromBasisIndex] = 1.;

	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		if(coefficientMap[n] != -1)
			coefficients[coefficientMap[n]*numCoefficients] = jIn1[n];

	//Generate a fixed hopping amplitude and index list, for speed.
	AmplitudeSet::iterator it = amplitudeSet->getIterator();
	HoppingAmplitude *ha;
	int *numHoppingAmplitudes = new int[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		numHoppingAmplitudes[n] = 0;
	while((ha = it.getHA())){
		numHoppingAmplitudes[amplitudeSet->getBasisIndex(ha->toIndex)]++;
		it.searchNextHA();
	}
	int maxHoppingAmplitudes = 0;
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		if(numHoppingAmplitudes[n] > maxHoppingAmplitudes)
			maxHoppingAmplitudes = numHoppingAmplitudes[n];

	delete [] numHoppingAmplitudes;

	int *currentHoppingAmplitudes = new int[amplitudeSet->getBasisSize()];
	for(int n = 0; n < amplitudeSet->getBasisSize(); n++)
		currentHoppingAmplitudes[n] = 0;

	complex<double> *hoppingAmplitudes = new complex<double>[maxHoppingAmplitudes*amplitudeSet->getBasisSize()];
	int *fromIndices = new int[maxHoppingAmplitudes*amplitudeSet->getBasisSize()];
	for(int n = 0; n < maxHoppingAmplitudes*amplitudeSet->getBasisSize(); n++){
		hoppingAmplitudes[n] = 0.;
		fromIndices[n] = 0;
	}

	it.reset();
	while((ha = it.getHA())){
		int to = amplitudeSet->getBasisIndex(ha->toIndex);
		int from = amplitudeSet->getBasisIndex(ha->fromIndex);

		hoppingAmplitudes[maxHoppingAmplitudes*to + currentHoppingAmplitudes[to]] = ha->getAmplitude()/scaleFactor;
		fromIndices[maxHoppingAmplitudes*to + currentHoppingAmplitudes[to]] = from;

		currentHoppingAmplitudes[to]++;

		it.searchNextHA();
	}

	delete [] currentHoppingAmplitudes;

	//Initialize GPU
	complex<double> *jIn1_device;
	complex<double> *jIn2_device;
	complex<double> *jResult_device;
	complex<double> *hoppingAmplitudes_device;
	int *fromIndices_device;
	complex<double> *coefficients_device;
	int *coefficientMap_device;

	int totalMemoryRequirement = amplitudeSet->getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += amplitudeSet->getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += amplitudeSet->getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(complex<double>);
	totalMemoryRequirement += maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(int);
	totalMemoryRequirement += to.size()*numCoefficients*sizeof(complex<double>);
	totalMemoryRequirement += amplitudeSet->getBasisSize()*sizeof(int);
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
		{	cout << "\tMalloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&jIn2_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&jResult_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&hoppingAmplitudes_device, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&fromIndices_device, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(int)) != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&coefficients_device, to.size()*numCoefficients*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&coefficientMap_device, amplitudeSet->getBasisSize()*sizeof(int)) != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}

	if(cudaMemcpy(jIn1_device, jIn1, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}
	if(cudaMemcpy(jIn2_device, jIn2, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}
	if(cudaMemcpy(jResult_device, jResult, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}
	if(cudaMemcpy(hoppingAmplitudes_device, hoppingAmplitudes, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}
	if(cudaMemcpy(fromIndices_device, fromIndices, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}
	if(cudaMemcpy(coefficients_device, coefficients, to.size()*numCoefficients*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}
	if(cudaMemcpy(coefficientMap_device, coefficientMap, amplitudeSet->getBasisSize()*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}

	//Calculate |j1>
	int block_size = 1024;
	int num_blocks = amplitudeSet->getBasisSize()/block_size + (amplitudeSet->getBasisSize()%block_size == 0 ? 0:1);
	if(isTalkative){
		cout << "\tCUDA Block size: " << block_size << "\n";
		cout << "\tCUDA Num blocks: " << num_blocks << "\n";
	}
	multiplyMatrixAndVector <<< num_blocks, block_size>>> ((cuDoubleComplex*)jIn1_device,
								(cuDoubleComplex*)jResult_device,
								(cuDoubleComplex*)hoppingAmplitudes_device,
								fromIndices_device,
								maxHoppingAmplitudes,
								amplitudeSet->getBasisSize(),
								(cuDoubleComplex*)coefficients_device,
								1,
								coefficientMap_device,
								numCoefficients);
	cudaError_t code = cudaGetLastError();
	if(code != cudaSuccess){
		cout << "\tMatrix vector multiplication error 1\n";
		cout << "\t" << cudaGetErrorString(code) << "\n";
		cout << "\tCUDA Block size: " << block_size << "\n";
		cout << "\tCUDA Num blocks: " << num_blocks << "\n";
		exit(1);
	}

	jTemp = jIn2_device;
	jIn2_device = jIn1_device;
	jIn1_device = jResult_device;
	jResult_device = jTemp;

	//Multiply hopping amplitudes by factor two, to speed up calculation of 2H|j(n-1)> - |j(n-2)>.
	for(int n = 0; n < maxHoppingAmplitudes*amplitudeSet->getBasisSize(); n++)
		hoppingAmplitudes[n] *= 2.;
	cudaMemcpy(hoppingAmplitudes_device, hoppingAmplitudes, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice);

	if(isTalkative)
		cout << "\tProgress (100 coefficients per dot): ";

	//Iteratively calculate |jn> and corresponding Chebyshev coefficients.
	for(int n = 2; n < numCoefficients; n++){
		subtractVector <<< num_blocks, block_size >>> ((cuDoubleComplex*)jIn2_device,
								(cuDoubleComplex*)jResult_device,
								amplitudeSet->getBasisSize());
		if(cudaGetLastError() != cudaSuccess){	cout << "Subtraction error\n";	exit(1);	}

		multiplyMatrixAndVector <<< num_blocks, block_size>>> ((cuDoubleComplex*)jIn1_device,
									(cuDoubleComplex*)jResult_device,
									(cuDoubleComplex*)hoppingAmplitudes_device,
									fromIndices_device,
									maxHoppingAmplitudes,
									amplitudeSet->getBasisSize(),
									(cuDoubleComplex*)coefficients_device,
									n,
									coefficientMap_device,
									numCoefficients);
		if(cudaGetLastError() != cudaSuccess){	cout << "Matrix vector multiplication error 2\n";	exit(1);	}

		jTemp = jIn2_device;
		jIn2_device = jIn1_device;
		jIn1_device = jResult_device;
		jResult_device = jTemp;

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
		cout << "\tMemcpy error\n";
		exit(1);
	}

	delete [] jIn1;
	delete [] jIn2;
	delete [] jResult;
	delete [] hoppingAmplitudes;
	delete [] fromIndices;
	delete [] coefficientMap;

	cudaFree(jIn1_device);
	cudaFree(jIn2_device);
	cudaFree(jResult_device);
	cudaFree(hoppingAmplitudes_device);
	cudaFree(fromIndices_device);
	cudaFree(coefficients_device);
	cudaFree(coefficientMap_device);

	//Lorentzian convolution
	double lambda = broadening*numCoefficients;
	for(int n = 0; n < numCoefficients; n++)
		for(int c = 0; c < to.size(); c++)
			coefficients[n + c*numCoefficients] = coefficients[n + c*numCoefficients]*sinh(lambda*(1 - n/(double)numCoefficients))/sinh(lambda);
}

__global__
void calculateGreensFunction(cuDoubleComplex *greensFunction,
				cuDoubleComplex *coefficients,
				cuDoubleComplex *lookupTable,
				int numCoefficients,
				int energyResolution){
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
			cout << memoryRequirement << "B";
		else if(memoryRequirement < 1024*1024)
			cout << memoryRequirement/1024 << "KB";
		else
			cout << memoryRequirement/1024/1024 << "MB";
	}

	if(cudaMalloc((void**)&generatingFunctionLookupTable_device, lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>))  != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}

	if(cudaMemcpy(generatingFunctionLookupTable_device, generatingFunctionLookupTable_host, lookupTableNumCoefficients*lookupTableResolution*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}

	delete [] generatingFunctionLookupTable_host;
}

void ChebyshevSolver::destroyLookupTableGPU(){
	if(isTalkative)
		cout << "ChebyshevSolver::destroyLookupTableGPU\n";

	if(generatingFunctionLookupTable_device == NULL){
		cout << "Error: No lookup table loaded onto GPU.\n";
		exit(1);
	}

	cudaFree(generatingFunctionLookupTable_device);
	generatingFunctionLookupTable_device = NULL;
}

void ChebyshevSolver::generateGreensFunctionGPU(complex<double> *greensFunction, complex<double> *coefficients){
	if(isTalkative)
		cout << "ChebyshevSolver::generateGreensFunctionGPU\n";

	if(generatingFunctionLookupTable_device == NULL){
		cout << "Error: No lookup table loaded onto GPU.\n";
		exit(1);
	}

	for(int e = 0; e < lookupTableResolution; e++)
		greensFunction[e] = 0.;

	complex<double> *greensFunction_device;
	complex<double> *coefficients_device;

	if(cudaMalloc((void**)&greensFunction_device, lookupTableResolution*sizeof(complex<double>))  != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&coefficients_device, lookupTableNumCoefficients*sizeof(complex<double>))  != cudaSuccess)
		{	cout << "\tMalloc error\n";	exit(1);	}

	if(cudaMemcpy(greensFunction_device, greensFunction, lookupTableResolution*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}
	if(cudaMemcpy(coefficients_device, coefficients, lookupTableNumCoefficients*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}

	int block_size = 1024;
	int num_blocks = lookupTableResolution/block_size + (lookupTableResolution%block_size == 0 ? 0:1);

	if(isTalkative){
		cout << "\tCUDA Block size: " << block_size << "\n";
		cout << "\tCUDA Num blocks: " << num_blocks << "\n";
	}

	calculateGreensFunction <<< num_blocks, block_size>>> ((cuDoubleComplex*)greensFunction_device,
								(cuDoubleComplex*)coefficients_device,
								(cuDoubleComplex*)generatingFunctionLookupTable_device,
								lookupTableNumCoefficients,
								lookupTableResolution);

	if(cudaMemcpy(greensFunction, greensFunction_device, lookupTableResolution*sizeof(complex<double>), cudaMemcpyDeviceToHost) != cudaSuccess)
		{	cout << "\tMemcpy error\n";	exit(1);	}

	cudaFree(greensFunction_device);
	cudaFree(coefficients_device);
}

};	//End of namespace TBTK
