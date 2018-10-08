/* Copyright 2017 Kristofer Björnson
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
 *  @file SuscesptibilityCalculator.h
 *  @brief Calculates the RPA susceptibility
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RPA_SUSCEPTIBILITY_CALCULATOR
#define COM_DAFER45_TBTK_RPA_SUSCEPTIBILITY_CALCULATOR

#include "TBTK/InteractionAmplitude.h"
#include "TBTK/RPA/MomentumSpaceContext.h"
#include "TBTK/SerializableVector.h"
#include "TBTK/RPA/SusceptibilityCalculator.h"
#include "TBTK/RPA/LindhardSusceptibilityCalculator.h"

#include <complex>

//#include <omp.h>

namespace TBTK{

class RPASusceptibilityCalculator{
public:
	/** Constructor. */
	RPASusceptibilityCalculator(
		const RPA::MomentumSpaceContext &momentumSpaceContext,
		SusceptibilityCalculator::Algorithm algorithm
			= SusceptibilityCalculator::Algorithm::Lindhard
	);

	/** Destructor. */
	~RPASusceptibilityCalculator();

	/** Create slave RPASusceptibilityCalcuator. The slave reuses internal
	 *  lookup tables used to speed up the calculations and should not be
	 *  used after the generating master have been destructed. */
	RPASusceptibilityCalculator* createSlave();

	/** Precompute susceptibilities. Will calculate the susceptibility for
	 *  all values using a parallel algorithm. Can speed up calculations if
	 *  most of the susceptibilities are needed. */
	void precompute(unsigned int numWorkers = 129);

	const RPA::MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Set interaction amplitudes. */
	void setInteractionAmplitudes(
		const std::vector<InteractionAmplitude> &interactionAmplitudes
	);

	/** Enum class for indicating whether the energy is an arbitrary comlex
	 *  number, or if it is restricted to the real or imaginary axis. */
	enum class EnergyType {Real, Imaginary, Complex};

	/** Set the mode used to calculate the susceptibility. */
//	void setSusceptibilityMode(SusceptibilityCalculator::Mode mode);

	/** Get the mode used to calculate the susceptibility. */
//	SusceptibilityCalculator::Mode getSusceptibilityMode() const;

	/** Set energy type. */
	void setEnergyType(EnergyType energyType);

	/** Get energy type. */
	EnergyType getEnergyType() const;

	/** Set the energies for which the susceptibility should be
	 *  calculated. */
	void setEnergies(
		const std::vector<std::complex<double>> &energies
	);

	/** Set to true if the susceptibility energies can be assumed
	 *  to be inversion symmetric in the complex plane.
	 *
	 *  Important note:
	 *  Only set this to true if the energies passed to
	 *  setSusceptibilityEnergies() are on the form
	 *  (-E_n, -E_{n-1}, ..., E_{n-1}, E_n). Setting this flag to
	 *  true without fullfilling this condition will result in
	 *  undefined behavior. */
	void setEnergiesAreInversionSymmetric(
		bool energiesAreInversionSymmetric
	);

	/** Get whether the susceptibility energies are inversion
	 *  symmetric. */
	bool getEnergiesAreInversionSymmetric() const;

	/** Set to true if the susceptibility is known to only be
	 *  evaluated at points away from poles. */
	void setSusceptibilityIsSafeFromPoles(
		bool susceptibilityIsSafeFromPoles
	);

	/** Get whether the susceptibility is known to only be
	 *  evaluated at points away from poles. */
	bool getSusceptibilityIsSafeFromPoles() const;

	/** Set the number of summation energies to use in Mode::Matsubara. */
//	void setNumSummationEnergies(unsigned int numSummationEnergies);

	/** Save susceptibilities. */
	void saveSusceptibilities(const std::string &filename) const;

	/** Load susceptibilities. */
	void loadSusceptibilities(const std::string &filename);
private:
	/** IndexedDataTree storing the RPA susceptibilities. */
	IndexedDataTree<SerializableVector<std::complex<double>>> rpaSusceptibilityTree;

	/** IndexedDataTree storing the RPA charge susceptibility. */
	IndexedDataTree<SerializableVector<std::complex<double>>> rpaChargeSusceptibilityTree;

	/** IndexedDataTree storing the RPA spin susceptibility. */
	IndexedDataTree<SerializableVector<std::complex<double>>> rpaSpinSusceptibilityTree;

	/** Energy type for the susceptibility. */
	EnergyType energyType;

	/** Energies to calculate the susceptibility for. */
	std::vector<std::complex<double>> energies;
public:
	/** Calculate RPA Susceptibility. */
	std::vector<std::complex<double>> calculateRPASusceptibility(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Calculate RPA Susceptibility. */
	std::vector<std::complex<double>> calculateRPASusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
	);

	/** Calculate charge RPA Susceptibility. */
	std::vector<std::complex<double>> calculateChargeRPASusceptibility(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Calculate charge RPA Susceptibility. */
	std::vector<std::complex<double>> calculateChargeRPASusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
	);

	/** Calculate spin RPA Susceptibility. */
	std::vector<std::complex<double>> calculateSpinRPASusceptibility(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Calculate spin RPA Susceptibility. */
	std::vector<std::complex<double>> calculateSpinRPASusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
	);

	/** Set U. */
	void setU(std::complex<double> U);

	/** Set Up. */
	void setUp(std::complex<double> Up);

	/** Set J. */
	void setJ(std::complex<double> J);

	/** Set Jp. */
	void setJp(std::complex<double> Jp);
private:
	/** SusceptibilityCalculator. */
	SusceptibilityCalculator *susceptibilityCalculator;

	/** InteractionAmplitudes. */
	std::vector<InteractionAmplitude> interactionAmplitudes;

	/** Interaction amplitudes for charge susceptibility. */
	std::vector<InteractionAmplitude> interactionAmplitudesCharge;

	/** Interaction amplitudes for charge susceptibility. */
	std::vector<InteractionAmplitude> interactionAmplitudesSpin;

	/** Flag indicating whether the interaction amplitudes for charge- and
	 *  spin-susceptibilities has been initialized. */
	bool interactionAmplitudesAreGenerated;

	/** Slave constructor. */
	RPASusceptibilityCalculator(
		SusceptibilityCalculator &susceptibilityCalculator
	);

	/** Get Susceptibility result Index. */
	Index getSusceptibilityResultIndex(
		const Index &kIndex,
		const std::vector<int> &orbitalIndices
	) const;

	/** Invert matix. */
	void invertMatrix(std::complex<double> *matrix, unsigned int dimensions);

	/** Multiply matrices. */
	void multiplyMatrices(
		std::complex<double> *matrix1,
		std::complex<double> *matrix2,
		std::complex<double> *result,
		unsigned int dimensions
	);

	/** RPA-susceptibility main algorithm. */
	std::vector<std::vector<std::vector<std::complex<double>>>> rpaSusceptibilityMainAlgorithm(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices,
		const std::vector<InteractionAmplitude> &interactionAmpltiudes
	);

	/** Interaction parameters. */
	std::complex<double> U, Up, J, Jp;

	/** Generate chare- and spin-interaction amplitudes. Can be called
	 *  multiple times and will only regenerate the interaction amplitudes
	 *  when needed. */
	void generateInteractionAmplitudes();
};

inline const RPA::MomentumSpaceContext& RPASusceptibilityCalculator::getMomentumSpaceContext(
) const{
	return susceptibilityCalculator->getMomentumSpaceContext();
}

inline void RPASusceptibilityCalculator::setInteractionAmplitudes(
	const std::vector<InteractionAmplitude> &interactionAmplitudes
){
	this->interactionAmplitudes = interactionAmplitudes;
}

inline Index RPASusceptibilityCalculator::getSusceptibilityResultIndex(
	const Index &kIndex,
	const std::vector<int> &orbitalIndices
) const{
	return Index(
		kIndex,
		{
			orbitalIndices.at(0),
			orbitalIndices.at(1),
			orbitalIndices.at(2),
			orbitalIndices.at(3)
		}
	);
}

/*inline void RPASusceptibilityCalculator::setSusceptibilityMode(
	SusceptibilityCalculator::Mode mode
){
	susceptibilityCalculator->setMode(mode);
}

inline SusceptibilityCalculator::Mode RPASusceptibilityCalculator::getSusceptibilityMode(
) const{
	return susceptibilityCalculator->getMode();
}*/

inline void RPASusceptibilityCalculator::setEnergyType(
	EnergyType energyType
){
	this->energyType = energyType;
	switch(energyType){
	case EnergyType::Real:
		susceptibilityCalculator->setEnergyType(
			SusceptibilityCalculator::EnergyType::Real
		);
		break;
	case EnergyType::Imaginary:
		susceptibilityCalculator->setEnergyType(
			SusceptibilityCalculator::EnergyType::Imaginary
		);
		break;
	case EnergyType::Complex:
		susceptibilityCalculator->setEnergyType(
			SusceptibilityCalculator::EnergyType::Complex
		);
		break;
	default:
		TBTKExit(
			"RPASusceptibilityCalculator::setEnergyType()",
			"Unknown energy type.",
			"This should never happen, contact the developer."
		);
	}
}

inline RPASusceptibilityCalculator::EnergyType RPASusceptibilityCalculator::getEnergyType(
) const{
	return energyType;
}

inline void RPASusceptibilityCalculator::setEnergies(
	const std::vector<std::complex<double>> &energies
){
	susceptibilityCalculator->setEnergies(
		energies
	);

	this->energies = energies;

	rpaSusceptibilityTree.clear();
	rpaChargeSusceptibilityTree.clear();
	rpaSpinSusceptibilityTree.clear();
}

inline void RPASusceptibilityCalculator::setEnergiesAreInversionSymmetric(
	bool energiesAreInversionSymmetric
){
	susceptibilityCalculator->setEnergiesAreInversionSymmetric(
		energiesAreInversionSymmetric
	);
}

inline bool RPASusceptibilityCalculator::getEnergiesAreInversionSymmetric(
) const{
	return susceptibilityCalculator->getEnergiesAreInversionSymmetric();
}

inline void RPASusceptibilityCalculator::setSusceptibilityIsSafeFromPoles(
	bool susceptibilityIsSafeFromPoles
){
	TBTKAssert(
		susceptibilityCalculator->getAlgorithm()
			== SusceptibilityCalculator::Algorithm::Lindhard,
		"RPASusceptibilityCalculator::setSusceptibilityIsSafeFromPoles()",
		"Only valid function call if the underlying algorithm is"
		<< " SusceptibilityCalculator::Algorithm::Lindhard.",
		""
	);

	((LindhardSusceptibilityCalculator*)susceptibilityCalculator)->setSusceptibilityIsSafeFromPoles(
		susceptibilityIsSafeFromPoles
	);
}

inline bool RPASusceptibilityCalculator::getSusceptibilityIsSafeFromPoles() const{
	TBTKAssert(
		susceptibilityCalculator->getAlgorithm()
			== SusceptibilityCalculator::Algorithm::Lindhard,
		"RPASusceptibilityCalculator::getSusceptibilityIsSafeFromPoles()",
		"Only valid function call if the underlying algorithm is"
		<< " SusceptibilityCalculator::Algorithm::Lindhard.",
		""
	);

	return ((LindhardSusceptibilityCalculator*)susceptibilityCalculator)->getSusceptibilityIsSafeFromPoles();
}

/*inline void RPASusceptibilityCalculator::setNumSummationEnergies(
	unsigned int numSummationEnergies
){
	susceptibilityCalculator->setNumSummationEnergies(
		numSummationEnergies
	);
}*/

inline void RPASusceptibilityCalculator::saveSusceptibilities(
	const std::string &filename
) const{
	susceptibilityCalculator->saveSusceptibilities(filename);
}

inline void RPASusceptibilityCalculator::loadSusceptibilities(
	const std::string &filename
){
	susceptibilityCalculator->loadSusceptibilities(filename);
}

inline std::vector<std::complex<double>> RPASusceptibilityCalculator::calculateRPASusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
){
	return calculateRPASusceptibility(
		DualIndex(
			susceptibilityCalculator->getMomentumSpaceContext().getKIndex(k),
			k
		),
		orbitalIndices
	);
}

inline std::vector<std::complex<double>> RPASusceptibilityCalculator::calculateChargeRPASusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
){
	return calculateChargeRPASusceptibility(
		DualIndex(
			susceptibilityCalculator->getMomentumSpaceContext().getKIndex(k),
			k
		),
		orbitalIndices
	);
}

inline std::vector<std::complex<double>> RPASusceptibilityCalculator::calculateSpinRPASusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
){
	return calculateSpinRPASusceptibility(
		DualIndex(
			susceptibilityCalculator->getMomentumSpaceContext().getKIndex(k),
			k
		),
		orbitalIndices
	);
}

inline void RPASusceptibilityCalculator::setU(std::complex<double> U){
	this->U = U;
	rpaSusceptibilityTree.clear();
	rpaChargeSusceptibilityTree.clear();
	rpaSpinSusceptibilityTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void RPASusceptibilityCalculator::setUp(std::complex<double> Up){
	this->Up = Up;
	rpaSusceptibilityTree.clear();
	rpaChargeSusceptibilityTree.clear();
	rpaSpinSusceptibilityTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void RPASusceptibilityCalculator::setJ(std::complex<double> J){
	this->J = J;
	rpaSusceptibilityTree.clear();
	rpaChargeSusceptibilityTree.clear();
	rpaSpinSusceptibilityTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void RPASusceptibilityCalculator::setJp(std::complex<double> Jp){
	this->Jp = Jp;
	rpaSusceptibilityTree.clear();
	rpaChargeSusceptibilityTree.clear();
	rpaSpinSusceptibilityTree.clear();
	interactionAmplitudesAreGenerated = false;
}

};	//End of namespace TBTK

#endif
