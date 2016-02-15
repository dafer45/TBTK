/** @package TBTKcalc
 *  @file ChebyshevSolver.h
 *  @brief Solves a Model using the Chebyshev method
 *
 *  Based on PhysRevLett.105.167006
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CHEBYSHEV_SOLVER
#define COM_DAFER45_TBTK_CHEBYSHEV_SOLVER

#include "Model.h"
#include <complex>

/** The ChebyshevSolver can be used to calculate Green's function for a given
 *  Model. The implementation is based on PhysRevLett.105.167006.
*/
class ChebyshevSolver{
public:
	/** Constructor. */
	ChebyshevSolver();
	/** Destructor. */
	~ChebyshevSolver();

	/** Set model to work on. */
	void setModel(Model *model);

	/** Calculates the Chebyshev coefficients for \f$ G_{ij}(E)\f$, where
	 *  \f$i = \textrm{to}\f$ and \f$j = \textrm{from}\f$.
	 *  @param to 'To'-index, or \f$i\f$*/
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
