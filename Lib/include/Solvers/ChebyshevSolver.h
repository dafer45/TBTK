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

#include "Communicator.h"
#include "GreensFunction.h"
#include "Model.h"
#include "Solver.h"

#include <complex>
#include <omp.h>

namespace TBTK{

/** The ChebyshevSolver can be used to calculate Green's function for a given
 *  Model. The implementation is based on PhysRevLett.105.167006. The
 *  ChebyshevSolver can be run on CPU, GPU, or a mixture of both. The
 *  calculation of Chebyshev coefficients scales as \f$O(n)\f$ with each of the
 *  following: dimension of the Hilbert space and number of Chebyshev
 *  coefficients. The generation of Green's functions scales as \f$O(n)\f$ with
 *  the following: Number of coefficients, energy resolution, and the number of
 *  Green's functions.
 */
class ChebyshevSolver : public Solver, public Communicator{
public:
	/** Constructor. */
	ChebyshevSolver();

	/** Destructor. */
	virtual ~ChebyshevSolver();

	/** Overrides Solver::setModel(). */
	virtual void setModel(Model &model);

	/** Set scale factor. */
	void setScaleFactor(double scaleFactor);

	/** Get scale factor. */
	double getScaleFactor();

	/** Calculates the Chebyshev coefficients for \f$ G_{ij}(E)\f$, where
	 *  \f$i = \textrm{to}\f$ is a set of indices and \f$j =
	 *  \textrm{from}\f$. Runs on CPU.
	 *  @param to vector of 'to'-indeces, or \f$i\f$'s.
	 *  @param from 'From'-index, or \f$j\f$.
	 *  @param coefficients Pointer to array able to hold
	 *  numCoefficients\f$\times\f$toIndeices.size() coefficients.
	 *  @param numCoefficients Number of coefficients to calculate for each
	 *  to-index.
	 *  @param broadening Broadening to use in convolusion of coefficients
	 *  to remedy Gibb's osciallations.
	 */
	void calculateCoefficients(
		std::vector<Index> &to,
		Index from,
		std::complex<double> *coefficients,
		int numCoefficients,
		double broadening = 0.000001
	);

	/** Calculates the Chebyshev coefficients for \f$ G_{ij}(E)\f$, where
	 *  \f$i = \textrm{to}\f$ and \f$j = \textrm{from}\f$. Runs on CPU.
	 *  @param to 'To'-index, or \f$i\f$.
	 *  @param from 'From'-index, or \f$j\f$.
	 *  @param coefficients Pointer to array able to hold numCoefficients coefficients.
	 *  @param numCoefficients Number of coefficients to calculate.
	 *  @param broadening Broadening to use in convolusion of coefficients
	 *  to remedy Gibb's osciallations.
	 */
	void calculateCoefficients(
		Index to,
		Index from,
		std::complex<double> *coefficients,
		int numCoefficients,
		double broadening = 0.000001
	);

	/** Calculates the Chebyshev coefficients for \f$ G_{ij}(E)\f$, where
	 *  \f$i = \textrm{to}\f$ is a set of indices and \f$j =
	 *  \textrm{from}\f$. Runs on GPU.
	 *  @param to vector of 'to'-indeces, or \f$i\f$'s.
	 *  @param from 'From'-index, or \f$j\f$.
	 *  @param coefficients Pointer to array able to hold
	 *  numCoefficients\f$\times\f$toIndeices.size() coefficients.
	 *  @param numCoefficients Number of coefficients to calculate for each
	 *  to-index.
	 *  @param broadening Broadening to use in convolusion of coefficients
	 *  to remedy Gibb's osciallations.
	 */
	void calculateCoefficientsGPU(
		std::vector<Index> &to,
		Index from,
		std::complex<double> *coefficients,
		int numCoefficients,
		double broadening = 0.000001
	);

	/** Calculates the Chebyshev coefficients for \f$ G_{ij}(E)\f$, where
	 *  \f$i = \textrm{to}\f$ and \f$j = \textrm{from}\f$. Runs on GPU.
	 *  @param to 'To'-index, or \f$i\f$.
	 *  @param from 'From'-index, or \f$j\f$.
	 *  @param coefficients Pointer to array able to hold numCoefficients coefficients.
	 *  @param numCoefficients Number of coefficients to calculate.
	 *  @param broadening Broadening to use in convolusion of coefficients
	 *  to remedy Gibb's osciallations.
	 */
	void calculateCoefficientsGPU(
		Index to,
		Index from,
		std::complex<double> *coefficients,
		int numCoefficients,
		double broadening = 0.000001
	);

	/** Experimental. */
	void calculateCoefficientsWithCutoff(
		Index to,
		Index from,
		std::complex<double> *coefficients,
		int numCoefficients,
		double componentCutoff,
		double broadening = 0.000001
	);

	/** Generate lokup table for quicker generation of multiple Green's
	 *  functions. Required if evaluation is to be performed on GPU.
	 *  @param numCoefficeints Number of coefficients used in Chebyshev
	 *  @param lowerBound Lower bound, has to be larger or equal to
	 *  -scaleFactor set by setScaleFactor (default value 1).
	 *  @param upperBound Upper bound, has to be smaller or equal to
	 *  scaleFactor setBy setScaleFactor (default value 1).
	 *  expansion.*/
	void generateLookupTable(
		int numCoefficeints,
		int energyResolution,
		double lowerBound = -1.,
		double upperBound = 1.
	);

	/** Free memory allocated by ChebyshevSolver::generateLookupTable(). */
	void destroyLookupTable();

	/** Returns true if a lookup table has been generated. */
	bool getLookupTableIsGenerated();

	/** Load lookup table generated by ChebyshevSolver::generateLookupTable
	 *   onto GPU. */
	void loadLookupTableGPU();

	/** Free memory allocated on GPU with
	 *  ChebyshevSolver::loadLookupTableGPU() */
	void destroyLookupTableGPU();

	/** Returns true if the lookup table has been loaded to the GPU. */
	bool getLookupTableIsLoadedGPU();

	/** Genererate Green's function. Does not use lookup table generated by
	 *  ChebyshevSolver::generateLookupTable. Runs on CPU.
	 *  @param greensFunction Pointer to array able to hold Green's
	 *  function. Has to be able to hold energyResolution elements.
	 *  @param coefficients Chebyshev coefficients calculated by
	 *  ChebyshevSolver::calculateCoefficients.
	 *  @param numCoefficeints Number of coefficients in coefficients.
	 *  @param energyResolution Number of elements in greensFunction.
	 *  @param lowerBound Lower bound, has to be larger or equal to
	 *  -scaleFactor set by setScaleFactor (default value 1).
	 *  @param upperBound Upper bound, has to be smaller or equal to
	 *  scaleFactor setBy setScaleFactor (default value 1).
	 */
	Property::GreensFunction* generateGreensFunction(
		std::complex<double> *coefficients,
		int numCoefficients,
		int energyResolution,
		double lowerBound = -1.,
		double upperBound = 1.,
		Property::GreensFunction::Type type = Property::GreensFunction::Type::Retarded
	);

	/** Genererate Green's function. Uses lookup table generated by
	 *  ChebyshevSolver::generateLookupTable. Runs on CPU.
	 *  @param greensFunction Pointer to array able to hold Green's
	 *  function. Has to be able to hold energyResolution elements.
	 *  @param coefficients Chebyshev coefficients calculated by
	 *  ChebyshevSolver::calculateCoefficients.
	 *
	 * numCoefficients and energyResolution are here the values specified
	 * in the call to ChebyshevSolver::generateLookupTable
	 */
	Property::GreensFunction* generateGreensFunction(
		std::complex<double> *coefficients,
		Property::GreensFunction::Type type = Property::GreensFunction::Type::Retarded
	);

	/** Genererate Green's function. Uses lookup table generated by
	 *  ChebyshevSolver::generateLookupTable. Runs on GPU.
	 *  @param greensFunction Pointer to array able to hold Green's
	 *  function. Has to be able to hold energyResolution elements.
	 *  @param coefficients Chebyshev coefficients calculated by
	 *  ChebyshevSolver::calculateCoefficients.
	 *
	 * numCoefficients and energyResolution are here the values specified
	 * in the call to ChebyshevSolver::generateLookupTable
	 */
	Property::GreensFunction* generateGreensFunctionGPU(
		std::complex<double> *coefficients,
		Property::GreensFunction::Type type = Property::GreensFunction::Type::Retarded
	);

	/** Damping potential based on J. Chem. Phys. 117, 9552 (2002).
	 *
	 *  @param distanceToEdge Distance from edge to the point at which to
	 *  calculate the damping factor.
	 *  @param boundarySize Size of the boundary region.
	 *  @param b Tuning parameter for optimizing the potential
	 *  @param c Tuning parameter for optimizing the potential
	 *
	 *  @return exp(-gamma), where gamma = 0 in the interior, infty outside
	 *  the edge, and determined by the function described in J. Chem.
	 *  Phys. 117, 9552 (2002), inside the boundary region. */
	std::complex<double> getMonolopoulosABCDamping(
		double distanceToEdge,
		double boundarySize,
		double e = 1.,
		double c = 2.62
	);

	/** Set damping mask. The damping mask will be used as prefactor in the
	 *  modified Chebyshev expansion used for implementing absorbing
	 *  boundary conditions. If set to NULL (default), no damping term will
	 *  be applied.*/
	void setDamping(std::complex<double> *damping);
private:
	/** Scale factor. */
	double scaleFactor;

	/** Damping mask. */
	std::complex<double> *damping;

	/** Pointer to lookup table used to speed up evaluation of multiple
	 *  Green's functions. */
	std::complex<double> **generatingFunctionLookupTable;

	/** Pointer to lookup table on GPU. */
	std::complex<double> ***generatingFunctionLookupTable_device;

	/** Number of coefficients assumed in the generatino of Green's
	 *  function using the lookup tables*/
	int lookupTableNumCoefficients;

	/** Energy resolution assumed in the generation of Green's functions
	 *  using the lookup table. */
	int lookupTableResolution;

	/** Lower bound for energy used for the lookup table. */
	double lookupTableLowerBound;

	/** Upper bound for energy used for the lookup table. */
	double lookupTableUpperBound;
};

inline void ChebyshevSolver::setScaleFactor(double scaleFactor){
	this->scaleFactor = scaleFactor;
}

inline double ChebyshevSolver::getScaleFactor(){
	return scaleFactor;
}

inline bool ChebyshevSolver::getLookupTableIsGenerated(){
	if(generatingFunctionLookupTable != NULL)
		return true;
	else
		return false;
}

inline bool ChebyshevSolver::getLookupTableIsLoadedGPU(){
	if(generatingFunctionLookupTable_device != NULL)
		return true;
	else
		return false;
}

inline void ChebyshevSolver::setDamping(std::complex<double> *damping){
	this->damping = damping;
}

};	//End of namespace TBTK

#endif
