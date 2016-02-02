#include "../include/ChebyshevSolver.h"
#include <math.h>
#include "../include/HALinkedList.h"
#include <cuComplex.h>

using namespace std;

clock_t start;
void tick(){
	start = clock();
}

double tock(){
	return (1000*(clock() - start))/CLOCKS_PER_SEC;
}

__global__
void multiplyMatrixAndVector(cuDoubleComplex *jIn,
				cuDoubleComplex *jResult,
				cuDoubleComplex *hoppingAmplitudes,
				int *fromIndices,
				int maxHoppingAmplitudes,
				int basisSize,
				cuDoubleComplex *coefficients,
				int currentCoefficient,
				int coefficientIndex){
	int to = blockIdx.x*blockDim.x + threadIdx.x;
	if(to < basisSize)
		for(int n = 0; n < maxHoppingAmplitudes; n++)
			jResult[to] = cuCadd(jResult[to], cuCmul(hoppingAmplitudes[maxHoppingAmplitudes*to + n], jIn[fromIndices[maxHoppingAmplitudes*to + n]]));

	if(to == coefficientIndex)
		coefficients[currentCoefficient] = jResult[to];
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

void ChebyshevSolver::calculateCoefficientsGPU(Index to, Index from, complex<double> *coefficients, int numCoefficients){
	AmplitudeSet *amplitudeSet = &system->amplitudeSet;

	int fromBasisIndex = amplitudeSet->getBasisIndex(from);
	int toBasisIndex = amplitudeSet->getBasisIndex(to);

	cout << "From Index: " << fromBasisIndex << "\n";
	cout << "To Index: " << toBasisIndex << "\n";
	cout << "Basis size: " << amplitudeSet->getBasisSize() << "\n";

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

	coefficients[0] = jIn1[toBasisIndex];

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

		hoppingAmplitudes[maxHoppingAmplitudes*to + currentHoppingAmplitudes[to]] = ha->getAmplitude();
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

	if(cudaMalloc((void**)&jIn1_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "Malloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&jIn2_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "Malloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&jResult_device, amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "Malloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&hoppingAmplitudes_device, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "Malloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&fromIndices_device, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(int)) != cudaSuccess)
		{	cout << "Malloc error\n";	exit(1);	}
	if(cudaMalloc((void**)&coefficients_device, numCoefficients*sizeof(complex<double>)) != cudaSuccess)
		{	cout << "Malloc error\n";	exit(1);	}

	if(cudaMemcpy(jIn1_device, jIn1, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "Memcpy error\n";	exit(1);	}
	if(cudaMemcpy(jIn2_device, jIn2, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "Memcpy error\n";	exit(1);	}
	if(cudaMemcpy(jResult_device, jResult, amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "Memcpy error\n";	exit(1);	}
	if(cudaMemcpy(hoppingAmplitudes_device, hoppingAmplitudes, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "Memcpy error\n";	exit(1);	}
	if(cudaMemcpy(fromIndices_device, fromIndices, maxHoppingAmplitudes*amplitudeSet->getBasisSize()*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "Memcpy error\n";	exit(1);	}
	if(cudaMemcpy(coefficients_device, coefficients, numCoefficients*sizeof(complex<double>), cudaMemcpyHostToDevice) != cudaSuccess)
		{	cout << "Memcpy error\n";	exit(1);	}

	//Calculate |j1>
	int block_size = 1024;
	int num_blocks = amplitudeSet->getBasisSize()/block_size + (amplitudeSet->getBasisSize()%block_size == 0 ? 0:1);
	cout << "CUDA Block size: " << block_size << "\n";
	cout << "CUDA Num blocks: " << num_blocks << "\n";
	multiplyMatrixAndVector <<< num_blocks, block_size>>> ((cuDoubleComplex*)jIn1_device,
								(cuDoubleComplex*)jResult_device,
								(cuDoubleComplex*)hoppingAmplitudes_device,
								fromIndices_device,
								maxHoppingAmplitudes,
								amplitudeSet->getBasisSize(),
								(cuDoubleComplex*)coefficients_device,
								1,
								toBasisIndex);
	cudaError_t code = cudaGetLastError();
	if(code != cudaSuccess){
		cout << "Matrix vector multiplication error 1\n";
		cout << cudaGetErrorString(code) << "\n";
		cout << "CUDA Block size: " << block_size << "\n";
		cout << "CUDA Num blocks: " << num_blocks << "\n";
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
									toBasisIndex);
		if(cudaGetLastError() != cudaSuccess){	cout << "Matrix vector multiplication error 2\n";	exit(1);	}

		jTemp = jIn2_device;
		jIn2_device = jIn1_device;
		jIn1_device = jResult_device;
		jResult_device = jTemp;

		if(n%100 == 0)
			cout << n << "\n";
	}

	if(cudaMemcpy(coefficients, coefficients_device, numCoefficients*sizeof(complex<double>), cudaMemcpyDeviceToHost) != cudaSuccess){
		cout << "Memcpy error\n";
		exit(1);
	}

	delete [] jIn1;
	delete [] jIn2;
	delete [] jResult;
	delete [] hoppingAmplitudes;
	delete [] fromIndices;

	cudaFree(jIn1_device);
	cudaFree(jIn2_device);
	cudaFree(jResult_device);
	cudaFree(hoppingAmplitudes_device);
	cudaFree(fromIndices_device);
	cudaFree(coefficients_device);

	//Lorentzian convolution
	double epsilon = 0.001;
	double lambda = epsilon*numCoefficients;
	for(int n = 0; n < numCoefficients; n++)
		coefficients[n] = coefficients[n]*sinh(lambda*(1 - n/(double)numCoefficients)/sinh(lambda));
}
