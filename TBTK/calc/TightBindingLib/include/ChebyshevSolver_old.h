#ifndef COM_DAFER45_BAND_STRUCTURE_CHEBYSHEV_SOLVER
#define COM_DAFER45_BAND_STRUCTURE_CHEBYSHEV_SOLVER

#include "System.h"
#include <complex>

class ChebyshevSolver{
public:
	ChebyshevSolver();
	~ChebyshevSolver();

	void setSystem(System *system);

	void calculateCoefficients(Index to, Index from, std::complex<double> *coefficients, int numCoefficients);
	void calculateCoefficientsGPU(Index to, Index from, std::complex<double> *coefficients, int numCoefficients);
	void calculateCoefficients(Index to, Index from, std::complex<double> *coefficients, int numCoefficients, double componentCutoff);

	void generateLookupTable(int numCoefficeints, int energyResolution);
	void generateGreensFunction(std::complex<double> *greensFunction, std::complex<double> *coefficients, int numCoefficients, int energyResolution);
	void generateGreensFunction(std::complex<double> *greensFunction, std::complex<double> *coefficients);
private:
	System *system;
	std::complex<double> **generatingFunctionLookupTable;
	int lookupTableNumCoefficients;
	int lookupTableResolution;
};

#endif
