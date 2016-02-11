#ifndef COM_DAFER45_TBTK_CHEBYSHEV_SOLVER
#define COM_DAFER45_TBTK_CHEBYSHEV_SOLVER

#include "Model.h"
#include <complex>

class ChebyshevSolver{
public:
	ChebyshevSolver();
	~ChebyshevSolver();

	void setModel(Model *model);

	void calculateCoefficients(Index to, Index from, std::complex<double> *coefficients, int numCoefficients, double broadening = 0.0001);
	void calculateCoefficientsGPU(std::vector<Index> &to, Index from, std::complex<double> *coefficients, int numCoefficients, double broadening = 0.0001);
	void calculateCoefficientsGPU(Index to, Index from, std::complex<double> *coefficients, int numCoefficients, double broadening = 0.0001);
	void calculateCoefficientsWithCutoff(Index to, Index from, std::complex<double> *coefficients, int numCoefficients, double componentCutoff, double broadening = 0.0001);

	void generateLookupTable(int numCoefficeints, int energyResolution);
	void loadLookupTableGPU();
	void destroyLookupTableGPU();
	void generateGreensFunction(std::complex<double> *greensFunction, std::complex<double> *coefficients, int numCoefficients, int energyResolution);
	void generateGreensFunction(std::complex<double> *greensFunction, std::complex<double> *coefficients);
	void generateGreensFunctionGPU(std::complex<double> *greensFunction, std::complex<double> *coefficients);
private:
	Model *model;
	std::complex<double> **generatingFunctionLookupTable;
	std::complex<double> **generatingFunctionLookupTable_device;
	int lookupTableNumCoefficients;
	int lookupTableResolution;
};

#endif
