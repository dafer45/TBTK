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
 *  @file ChebyshevExpander.h
 *  @brief Solves a Model using the Chebyshev method.
 *
 *  Based on PhysRevLett.105.167006
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_CHEBYSHEV_EXPANDER
#define COM_DAFER45_TBTK_SOLVER_CHEBYSHEV_EXPANDER

#include "TBTK/Communicator.h"
#include "TBTK/Model.h"
#include "TBTK/Solver/Solver.h"

#include <complex>
#ifndef __APPLE__
#	include <omp.h>
#endif

namespace TBTK{
namespace Solver{

/** @brief Solves a Model using the Chebyshev method.
 *
 *  The ChebyshevExpander can be used to calculate Green's function for a given
 *  Model using an expansion of the form
 *  <br/>
 *  <center>\f$
 *    G_{\mathbf{i}\mathbf{j}}(E) = \frac{1}{\sqrt{s^2 - E^2}}
 *    \sum_{m=0}^{\infty}
 *    \frac{b_{\mathbf{i}\mathbf{j}}^{(m)}}{1 + \delta_{0m}}
 *    F(m\textrm{acos}(E/s)),
 *  \f$</center>
 *  <br/>
 *  where \f$F(x)\f$ is one of the functions \f$\cos(x)\f$, \f$\sin(x)\f$,
 *  \f$e^{ix}\f$, and \f$e^{-ix}\f$.
 *  The implementation is based on PhysRevLett.105.167006. The
 *  ChebyshevExpander can be run on CPU, GPU, or a mixture of both. The
 *  calculation of Chebyshev coefficients scales as \f$O(n)\f$ with each of the
 *  following: dimension of the Hilbert space and number of Chebyshev
 *  coefficients. The generation of Green's functions scales as \f$O(n)\f$ with
 *  the following: Number of coefficients, energy resolution, and the number of
 *  Green's functions.
 */
class ChebyshevExpander : public Solver, public Communicator{
public:
	/** Constructs a Solver::ChebyshevExpander. */
	ChebyshevExpander();

	/** Destructor. */
	virtual ~ChebyshevExpander();

	/** Overrides Solver::setModel(). */
	virtual void setModel(Model &model);

	/** Sets the scale factor that rescales the Hamiltonian to ensure that
	 *  the energy spectrum of the Hamiltonian is bounded on the interval
	 *  (-1, 1).
	 *
	 *  @param scaleFactor The scale factor. */
	void setScaleFactor(double scaleFactor);

	/** Get scale factor.
	 *
	 *  @return The scale factor that is used to rescale the Hamiltonian. */
	double getScaleFactor();

	/** Set the number of Chebyshev coefficients to use during
	 *  calculations. The default value is 1000.
	 *
	 *  @param numCoefficients The number of Chebyshev coefficients. */
	void setNumCoefficients(int numCoefficients);

	/** Get the number of Chebyshev coefficients to use during
	 *  calculations.
	 *
	 *  @return The number of Chebyshev coefficients. */
	int getNumCoefficients() const;

	/** Set the broadening to use in convolusion of coefficients to remedy
	 *  Gibb's osciallations.
	 *
	 *  @param broadening The broadening parameter to use. */
	void setBroadening(double broadening);

	/** Get the broadening to use in convolusion of coefficients to remedy
	 *  Gibb's osciallations.
	 *
	 *  @return The broadening parameter used. */
	double getBroadening() const;

	/** Set the number of energy point to use when generating Green's
	 *  functions. The default value is 1000.
	 *
	 *  @param energyResolution The number of energy points to use. */
	void setEnergyResolution(int energyResolution);

	/** Get the number of energy point to use when generating Green's
	 *  functions.
	 *
	 *  @return The number of energy points to use. */
	int getEnergyResolution() const;

	/** Set the lower bound to use for the energy when generating the
	 *  Green's function. The default value is -1.
	 *
	 *  @param lowerBound The lower bound for the energy. */
	void setLowerBound(double lowerBound);

	/** Get the lower bound to use for the energy when generating the
	 *  Green's function.
	 *
	 *  @return The lower bound for the energy. */
	double getLowerBound() const;

	/** Set the upper bound to use for the energy when generating the
	 *  Green's function. The default value is 1.
	 *
	 *  @param upperBound The upper bound for the energy. */
	void setUpperBound(double upperBound);

	/** Get the upper bound to use for the energy when generating the
	 *  Green's function.
	 *
	 *  @return The upper bound for the energy. */
	double getUpperBound() const;

	/** Set whether Chebyshev coefficients should be calculated on GPU. The
	 *  default value is false.
	 *
	 *  @param calculateCoefficientsOnGPU True to use GPU, false to use
	 *  CPU. */
	void setCalculateCoefficientsOnGPU(bool calculateCoefficientsOnGPU);

	/** Get whether Chebyshev coefficients are set to be calculated on GPU.
	 *
	 *  @return True if GPU is used. */
	bool getCalculateCoefficientsOnGPU() const;

	/** Set whether Green's functions should be generated on GPU. The
	 *  default value is false.
	 *
	 *  @param generateGreensFunctionsOnGPU True to use GPU, false to use
	 *  CPU. */
	void setGenerateGreensFunctionsOnGPU(
		bool generateGreensFunctionsOnGPU
	);

	/** Get whether Green's functions are set to be generated on GPU.
	 *
	 *  @return True if GPU is used. */
	bool getGenerateGreensFunctionsOnGPU() const;

	/** Set whether a lookup table should be used to when generating
	 *  Green's functions.
	 *
	 *  @param useLookupTable True to use a lookup table. */
	void setUseLookupTable(bool useLookupTable);

	/** Set whether a lookup table should be used to when generating
	 *  Green's functions.
	 *
	 *  @return True if a lookup table is used. */
	bool getUseLookupTable() const ;

	/** Calculates the Chebyshev coefficients for \f$ G_{ij}(E)\f$, where
	 *  \f$i = \textrm{to}\f$ is a set of indices and \f$j =
	 *  \textrm{from}\f$.
	 *
	 *  @param to vector of 'to'-indeces, or \f$i\f$'s.
	 *  @param from 'From'-index, or \f$j\f$.
	 *  @param coefficients Pointer to array able to hold
	 *  numCoefficients\f$\times\f$toIndeices.size() coefficients.
	 *
	 *  @param numCoefficients Number of coefficients to calculate for each
	 *  to-index.
	 *
	 *  @param broadening Broadening to use in convolusion of coefficients
	 *  to remedy Gibb's osciallations. */
	std::vector<std::vector<std::complex<double>>> calculateCoefficients(
		std::vector<Index> &to,
		Index from
	);

	/** Calculates the Chebyshev coefficients for \f$ G_{ij}(E)\f$, where
	 *  \f$i = \textrm{to}\f$ and \f$j = \textrm{from}\f$.
	 *
	 *  @param to 'To'-index, or \f$i\f$.
	 *  @param from 'From'-index, or \f$j\f$.
	 *  @param coefficients Pointer to array able to hold numCoefficients
	 *  coefficients.
	 *
	 *  @param numCoefficients Number of coefficients to calculate.
	 *  @param broadening Broadening to use in convolusion of coefficients
	 *  to remedy Gibb's osciallations. */
	std::vector<std::complex<double>> calculateCoefficients(
		Index to,
		Index from
	);

	/** Enum class describing the type of Green's function to calculate. */
	enum class Type{
		Advanced,
		Retarded,
		Principal,
		NonPrincipal
	};

	/** Genererate Green's function. Uses lookup table generated by
	 *  ChebyshevExpander::generateLookupTable. Runs on CPU.
	 *  @param greensFunction Pointer to array able to hold Green's
	 *  function. Has to be able to hold energyResolution elements.
	 *  @param coefficients Chebyshev coefficients calculated by
	 *  ChebyshevExpander::calculateCoefficients.
	 *
	 * numCoefficients and energyResolution are here the values specified
	 * in the call to ChebyshevExpander::generateLookupTable
	 */
	std::vector<std::complex<double>> generateGreensFunction(
		const std::vector<std::complex<double>> &coefficients,
		Type type = Type::Retarded
	);
private:
	//These three functions are experimental and therefore not part of the
	//public interface of released code.

	/** Experimental. */
	void calculateCoefficientsWithCutoff(
		Index to,
		Index from,
		std::complex<double> *coefficients,
		int numCoefficients,
		double componentCutoff,
		double broadening = 0.000001
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

	/** The number of Chebyshev coefficients to calculate and use. */
	int numCoefficients;

	/** Broadening parameter to use to remedy Gibbs oscilations. */
	double broadening;

	/** The number of of energy points to use when generating the Green's
	 *  function. */
	int energyResolution;

	/** The lower bound for the energy for the Green's function. */
	double lowerBound;

	/** The upper bound for the energy for the Green's function. */
	double upperBound;

	/** Flag indicating whether to use GPU to calculate Chebyshev
	 *  coefficients. */
	bool calculateCoefficientsOnGPU;

	/** Flag indicating whether to use GPU to generate Green's functions.
	 */
	bool generateGreensFunctionsOnGPU;

	/** Flag indicating whether to use a lookup table when generating
	 *  Green's functions. */
	bool useLookupTable;

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

	/** Ensure that the lookup table is in a ready state. */
	void ensureLookupTableIsReady();

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

	/** Free memory allocated by ChebyshevExpander::generateLookupTable(). */
	void destroyLookupTable();

	/** Returns true if a lookup table has been generated. */
	bool getLookupTableIsGenerated();

	/** Load lookup table generated by
	 *  ChebyshevExpander::generateLookupTable onto GPU. */
	void loadLookupTableGPU();

	/** Free memory allocated on GPU with
	 *  ChebyshevExpander::loadLookupTableGPU() */
	void destroyLookupTableGPU();

	/** Returns true if the lookup table has been loaded to the GPU. */
	bool getLookupTableIsLoadedGPU();

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
	std::vector<
		std::vector<std::complex<double>>
	> calculateCoefficientsCPU(
		std::vector<Index> &to,
		Index from
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
	std::vector<std::complex<double>> calculateCoefficientsCPU(
		Index to,
		Index from
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
	 *  to remedy Gibb's osciallations. */
	std::vector<
		std::vector<std::complex<double>>
	> calculateCoefficientsGPU(
		std::vector<Index> &to,
		Index from
	);

	/** Calculates the Chebyshev coefficients for \f$ G_{ij}(E)\f$, where
	 *  \f$i = \textrm{to}\f$ and \f$j = \textrm{from}\f$. Runs on GPU.
	 *  @param to 'To'-index, or \f$i\f$.
	 *  @param from 'From'-index, or \f$j\f$.
	 *  @param coefficients Pointer to array able to hold numCoefficients coefficients.
	 *  @param numCoefficients Number of coefficients to calculate.
	 *  @param broadening Broadening to use in convolusion of coefficients
	 *  to remedy Gibb's osciallations. */
	std::vector<std::complex<double>> calculateCoefficientsGPU(
		Index to,
		Index from
	);

	/** Genererate Green's function. Does not use lookup table generated by
	 *  ChebyshevExpander::generateLookupTable. Runs on CPU.
	 *  @param greensFunction Pointer to array able to hold Green's
	 *  function. Has to be able to hold energyResolution elements.
	 *  @param coefficients Chebyshev coefficients calculated by
	 *  ChebyshevExpander::calculateCoefficients.
	 *  @param numCoefficeints Number of coefficients in coefficients.
	 *  @param energyResolution Number of elements in greensFunction.
	 *  @param lowerBound Lower bound, has to be larger or equal to
	 *  -scaleFactor set by setScaleFactor (default value 1).
	 *  @param upperBound Upper bound, has to be smaller or equal to
	 *  scaleFactor setBy setScaleFactor (default value 1).
	 */
/*	std::complex<double>* generateGreensFunctionCPU(
		std::complex<double> *coefficients,
		int numCoefficients,
		int energyResolution,
		double lowerBound = -1.,
		double upperBound = 1.,
		Type type = Type::Retarded
	);*/

	/** Genererate Green's function. Uses lookup table generated by
	 *  ChebyshevExpander::generateLookupTable. Runs on CPU.
	 *  @param greensFunction Pointer to array able to hold Green's
	 *  function. Has to be able to hold energyResolution elements.
	 *  @param coefficients Chebyshev coefficients calculated by
	 *  ChebyshevExpander::calculateCoefficients.
	 *
	 * numCoefficients and energyResolution are here the values specified
	 * in the call to ChebyshevExpander::generateLookupTable
	 */
	std::vector<std::complex<double>> generateGreensFunctionCPU(
		const std::vector<std::complex<double>> &coefficients,
		Type type = Type::Retarded
	);

	/** Genererate Green's function. Uses lookup table generated by
	 *  ChebyshevExpander::generateLookupTable. Runs on GPU.
	 *  @param greensFunction Pointer to array able to hold Green's
	 *  function. Has to be able to hold energyResolution elements.
	 *  @param coefficients Chebyshev coefficients calculated by
	 *  ChebyshevExpander::calculateCoefficients.
	 *
	 * numCoefficients and energyResolution are here the values specified
	 * in the call to ChebyshevExpander::generateLookupTable
	 */
	std::vector<std::complex<double>> generateGreensFunctionGPU(
		const std::vector<std::complex<double>> &coefficients,
		Type type = Type::Retarded
	);
};

inline void ChebyshevExpander::setScaleFactor(double scaleFactor){
	if(generatingFunctionLookupTable != nullptr)
		destroyLookupTable();
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();

	this->scaleFactor = scaleFactor;
}

inline double ChebyshevExpander::getScaleFactor(){
	return scaleFactor;
}

inline void ChebyshevExpander::setNumCoefficients(int numCoefficients){
	if(generatingFunctionLookupTable != nullptr)
		destroyLookupTable();
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();

	this->numCoefficients = numCoefficients;
}

inline int ChebyshevExpander::getNumCoefficients() const{
	return numCoefficients;
}

inline void ChebyshevExpander::setBroadening(double broadening){
	this->broadening = broadening;
}

inline double ChebyshevExpander::getBroadening() const{
	return broadening;
}

inline void ChebyshevExpander::setEnergyResolution(int energyResolution){
	if(generatingFunctionLookupTable != nullptr)
		destroyLookupTable();
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();

	this->energyResolution = energyResolution;
}

inline int ChebyshevExpander::getEnergyResolution() const{
	return energyResolution;
}

inline void ChebyshevExpander::setLowerBound(double lowerBound){
	if(generatingFunctionLookupTable != nullptr)
		destroyLookupTable();
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();

	this->lowerBound = lowerBound;
}

inline double ChebyshevExpander::getLowerBound() const{
	return lowerBound;
}

inline void ChebyshevExpander::setUpperBound(double upperBound){
	if(generatingFunctionLookupTable != nullptr)
		destroyLookupTable();
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();

	this->upperBound = upperBound;
}

inline double ChebyshevExpander::getUpperBound() const{
	return upperBound;
}

inline void ChebyshevExpander::setCalculateCoefficientsOnGPU(
	bool calculateCoefficientsOnGPU
){
	this->calculateCoefficientsOnGPU = calculateCoefficientsOnGPU;
}

inline bool ChebyshevExpander::getCalculateCoefficientsOnGPU() const{
	return calculateCoefficientsOnGPU;
}

inline void ChebyshevExpander::setGenerateGreensFunctionsOnGPU(
	bool generateGreensFunctionsOnGPU
){
	this->generateGreensFunctionsOnGPU = generateGreensFunctionsOnGPU;
}

inline bool ChebyshevExpander::getGenerateGreensFunctionsOnGPU() const{
	return generateGreensFunctionsOnGPU;
}

inline void ChebyshevExpander::setUseLookupTable(bool useLookupTable){
	if(!useLookupTable){
		if(generatingFunctionLookupTable != nullptr)
			destroyLookupTable();
		if(generatingFunctionLookupTable_device != nullptr)
			destroyLookupTableGPU();
	}

	this->useLookupTable = useLookupTable;
}

inline bool ChebyshevExpander::getUseLookupTable() const{
	return useLookupTable;
}

inline std::vector<
		std::vector<std::complex<double>>
> ChebyshevExpander::calculateCoefficients(
	std::vector<Index> &to,
	Index from
){
	if(calculateCoefficientsOnGPU){
		return calculateCoefficientsGPU(
			to,
			from
		);
	}
	else{
		return calculateCoefficientsCPU(
			to,
			from
		);
	}
}

inline std::vector<std::complex<double>> ChebyshevExpander::calculateCoefficients(
	Index to,
	Index from
){
	if(calculateCoefficientsOnGPU){
		return calculateCoefficientsGPU(
			to,
			from
		);
	}
	else{
		return calculateCoefficientsCPU(
			to,
			from
		);
	}
}

inline bool ChebyshevExpander::getLookupTableIsGenerated(){
	if(generatingFunctionLookupTable != NULL)
		return true;
	else
		return false;
}

inline bool ChebyshevExpander::getLookupTableIsLoadedGPU(){
	if(generatingFunctionLookupTable_device != NULL)
		return true;
	else
		return false;
}

inline std::vector<std::complex<double>> ChebyshevExpander::generateGreensFunction(
	const std::vector<std::complex<double>> &coefficients,
	Type type
){
	if(generateGreensFunctionsOnGPU){
		return generateGreensFunctionGPU(coefficients, type);
	}
	else{
		return generateGreensFunctionCPU(coefficients, type);
	}
}

inline void ChebyshevExpander::setDamping(std::complex<double> *damping){
	this->damping = damping;
}

inline void ChebyshevExpander::ensureLookupTableIsReady(){
	if(useLookupTable){
		if(!generatingFunctionLookupTable)
			generateLookupTable(numCoefficients, energyResolution, lowerBound, upperBound);
		if(generateGreensFunctionsOnGPU && !generatingFunctionLookupTable_device)
			loadLookupTableGPU();
	}
	else if(generateGreensFunctionsOnGPU){
		TBTKExit(
			"Solver::ChebyshevSolver::ensureLookupTableIsReady()",
			"Green's functions can only be generated on GPU using"
			<< " lookup tables.",
			"Use Solver::ChebyshevExpander::setGenerateGreensFunctionOnGPU()"
			<< " and"
			<< " Solver::ChebyshevExpander::setUseLookupTable() to"
			<< " configure the Solver::ChebyshevExpander."
		);
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
