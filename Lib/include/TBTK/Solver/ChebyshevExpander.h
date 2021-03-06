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

#include "TBTK/CArray.h"
#include "TBTK/Communicator.h"
#include "TBTK/Invalidatable.h"
#include "TBTK/Model.h"
#include "TBTK/Range.h"
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
 *
 *  Use the PropertyExtractor::ChebyshevExpander to calculate @link
 *  Property::AbstractProperty Properties@endlink.
 *
 *  # Example
 *  \snippet Solver/ChebyshevExpander.cpp ChebyshevExpander
 *  ## Output
 *  \snippet output/Solver/ChebyshevExpander.txt ChebyshevExpander */
class ChebyshevExpander : public Solver, public Communicator{
	TBTK_DYNAMIC_TYPE_INFORMATION(ChebyshevExpander)
public:
	/** Constructs a Solver::ChebyshevExpander. */
	ChebyshevExpander();

	/** Destructor. */
	virtual ~ChebyshevExpander();

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

	/** Set the energy window. The energy window must be contained in the
	 *  window (-SCALE_FACTOR, SCALE_FACTOR).
	 *
	 *  @param energyWindow The energy window to use when calculating the
	 *  Green's function. */
	void setEnergyWindow(const Range &energyWindow);

	/** Get the energy window used by the ChebyshevExpander.
	 *
	 *  @return The energy window. */
	const Range& getEnergyWindow() const;

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
	/** Scale factor. */
	double scaleFactor;

	/** The number of Chebyshev coefficients to calculate and use. */
	int numCoefficients;

	/** Broadening parameter to use to remedy Gibbs oscilations. */
	double broadening;

	/** The energy window to calculate the Green's function over. */
	Range energyWindow;

	/** Flag indicating whether to use GPU to calculate Chebyshev
	 *  coefficients. */
	bool calculateCoefficientsOnGPU;

	/** Flag indicating whether to use GPU to generate Green's functions.
	 */
	bool generateGreensFunctionsOnGPU;

	/** Flag indicating whether to use a lookup table when generating
	 *  Green's functions. */
	bool useLookupTable;

	/** Pointer to lookup table used to speed up evaluation of multiple
	 *  Green's functions. */
	Invalidatable<
		CArray<CArray<std::complex<double>>>
	> generatingFunctionLookupTable;

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
	 *  -scaleFactor set by setScaleFactor (default value 1.1).
	 *  @param upperBound Upper bound, has to be smaller or equal to
	 *  scaleFactor setBy setScaleFactor (default value 1.1).
	 *  expansion.*/
	void generateLookupTable();

	/** Free memory allocated by ChebyshevExpander::generateLookupTable(). */
	void destroyLookupTable();

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
	TBTKAssert(
		scaleFactor > 0,
		"Solver::ChebyshevExapnder::setScaleFactor()",
		"The 'scaleFactor=" << scaleFactor << "' has to be larger than"
		<< " '0'.",
		""
	);

	destroyLookupTable();
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();

	this->scaleFactor = scaleFactor;
}

inline double ChebyshevExpander::getScaleFactor(){
	return scaleFactor;
}

inline void ChebyshevExpander::setNumCoefficients(int numCoefficients){
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

inline void ChebyshevExpander::setEnergyWindow(const Range &energyWindow){
	const double epsilon = std::numeric_limits<double>::epsilon();
	TBTKAssert(
		energyWindow[0] > -scaleFactor*(1 - 16*epsilon)
		&& energyWindow.getLast() < scaleFactor*(1 - 16*epsilon),
		"Solver::ChebyshevExpander::setEnergyWindow()",
		"Invalid energy window. The 'energyWindow=["
		<< energyWindow[0] << ", " << energyWindow.getLast() << "]' is"
		<< " not contained in the interval (-scaleFactor, scaleFactor)"
		<< "=(" << -scaleFactor << ", " << scaleFactor << ").",
		""
	);
	destroyLookupTable();
	if(generatingFunctionLookupTable_device != nullptr)
		destroyLookupTableGPU();

	this->energyWindow = energyWindow;
}

inline const Range& ChebyshevExpander::getEnergyWindow() const{
	return energyWindow;
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

inline void ChebyshevExpander::ensureLookupTableIsReady(){
	if(useLookupTable){
		if(!generatingFunctionLookupTable.getIsValid())
			generateLookupTable();
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
