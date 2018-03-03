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
 *  @file SelfEnergyCalculator.h
 *  @brief Calculates the self-energy using the RPA approximation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ELECTRON_FLUCTUATION_VERTEX_CALCULATOR
#define COM_DAFER45_TBTK_ELECTRON_FLUCTUATION_VERTEX_CALCULATOR

#include "TBTK/BrillouinZone.h"
#include "TBTK/IndexedDataTree.h"
#include "TBTK/RPA/RPASusceptibilityCalculator.h"

namespace TBTK{

class ElectronFluctuationVertexCalculator{
public:
	/** Constructor. */
	ElectronFluctuationVertexCalculator(
		const MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	~ElectronFluctuationVertexCalculator();

	/** Create slave ElectronFluctuationVertexCalculator. The slave reuses
	 *  internal lookup tables used to speed up the calculations and should
	 *  not be used after the generating master have been destructed. */
	ElectronFluctuationVertexCalculator* createSlave();

	/** Get momentum cpsace context. */
	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Enum class for indicating whether the energy is an arbitrary comlex
	 *  number, or if it is restricted to the real or imaginary axis. */
	enum class EnergyType {Real, Imaginary, Complex};

	/** Set self-energy energy type. */
	void setEnergyType(EnergyType energyType);

	/** Get energy type. */
	EnergyType getEnergyType() const;

	/** Set the energies for which the self-energy should be
	 *  calculated. */
	void setEnergies(
		const std::vector<std::complex<double>> &energies
	);

	/** Set to true if the energies can be assumed to be inversion
	 *  symmetric in the complex plane.
	 *
	 *  Important note:
	 *  Only set this to true if the energies passed to setEnergies() are
	 *  on the form (-E_n, -E_{n-1}, ..., E_{n-1}, E_{n}. Setting this flag
	 *  to true without fullfilling this condition will result in undefined
	 *  behavior. */
	void setEnergiesAreInversionSymmetric(
		bool energiesAreInversionSymmetric
	);

	/** Get wheter the energies are inversion symmetric. */
	bool getEnergiesAreInversionSymmetric() const;

	/** Calculate self-energy vertex. */
	std::vector<std::complex<double>> calculateSelfEnergyVertex(
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

	/** Generate interaction amplitudes. Can be called multiple times and
	 *  will only regenerate the interaction amplitudes when needed. */
	void generateInteractionAmplitudes();

	/** Save susceptibilities. */
	void saveSusceptibilities(const std::string &filename) const;

	/** Load susceptibilities. */
	void loadSusceptibilities(const std::string &filename);
private:
	/** SusceptibilityCalculator. */
	RPASusceptibilityCalculator *rpaSusceptibilityCalculator;

	/** Energy type. */
	EnergyType energyType;

	/** Energies to calculate the vertex for. */
	std::vector<std::complex<double>> energies;

	/** Flag indicating whether the SelfEnergyCalculator is initialized. */
	bool isInitialized;

	/** IndexedDataTree storing the vertex. */
	IndexedDataTree<SerializeableVector<std::complex<double>>> vertexTree;

	/** Interaction parameters. */
	std::complex<double> U, Up, J, Jp;

	/** Interaction amplitudes used to calculate the self-energy vertex. */
	std::vector<InteractionAmplitude> u1;
	std::vector<InteractionAmplitude> u2;
	std::vector<InteractionAmplitude> u3;

	/** Flag indicating whether the interaction amplitudes are initialized.
	 */
	bool interactionAmplitudesAreGenerated;

	/** Slave constructor. */
	ElectronFluctuationVertexCalculator(
		RPASusceptibilityCalculator &rpaSusceptibilityCalculator
	);
};

inline const MomentumSpaceContext& ElectronFluctuationVertexCalculator::getMomentumSpaceContext(
) const{
	return rpaSusceptibilityCalculator->getMomentumSpaceContext();
}

inline void ElectronFluctuationVertexCalculator::setEnergyType(
	EnergyType energyType
){
	this->energyType = energyType;
	switch(energyType){
	case EnergyType::Real:
		rpaSusceptibilityCalculator->setEnergyType(
			RPASusceptibilityCalculator::EnergyType::Real
		);
		break;
	case EnergyType::Imaginary:
		rpaSusceptibilityCalculator->setEnergyType(
			RPASusceptibilityCalculator::EnergyType::Imaginary
		);
		break;
	case EnergyType::Complex:
		rpaSusceptibilityCalculator->setEnergyType(
			RPASusceptibilityCalculator::EnergyType::Complex
		);
		break;
	default:
		TBTKExit(
			"ElectronFluctuationVertexCalculator::setEnergyType()",
			"Unknown energy type.",
			"This should never happen, contact the developer."
		);
	}
}

inline ElectronFluctuationVertexCalculator::EnergyType ElectronFluctuationVertexCalculator::getEnergyType(
) const{
	return energyType;
}

inline void ElectronFluctuationVertexCalculator::setEnergies(
	const std::vector<std::complex<double>> &energies
){
	this->energies = energies;
	vertexTree.clear();

	rpaSusceptibilityCalculator->setEnergies(energies);
}

inline void ElectronFluctuationVertexCalculator::setEnergiesAreInversionSymmetric(
	bool energiesAreInversionSymmetric
){
	rpaSusceptibilityCalculator->setEnergiesAreInversionSymmetric(
		energiesAreInversionSymmetric
	);
}

inline bool ElectronFluctuationVertexCalculator::getEnergiesAreInversionSymmetric(
) const{
	return rpaSusceptibilityCalculator->getEnergiesAreInversionSymmetric();
}

inline void ElectronFluctuationVertexCalculator::setU(std::complex<double> U){
	this->U = U;
	rpaSusceptibilityCalculator->setU(U);
	vertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void ElectronFluctuationVertexCalculator::setUp(std::complex<double> Up){
	this->Up = Up;
	rpaSusceptibilityCalculator->setUp(Up);
	vertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void ElectronFluctuationVertexCalculator::setJ(std::complex<double> J){
	this->J = J;
	rpaSusceptibilityCalculator->setJ(J);
	vertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void ElectronFluctuationVertexCalculator::setJp(std::complex<double> Jp){
	this->Jp = Jp;
	rpaSusceptibilityCalculator->setJp(Jp);
	vertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void ElectronFluctuationVertexCalculator::saveSusceptibilities(
	const std::string &filename
) const{
	rpaSusceptibilityCalculator->saveSusceptibilities(filename);
}

inline void ElectronFluctuationVertexCalculator::loadSusceptibilities(
	const std::string &filename
){
	rpaSusceptibilityCalculator->loadSusceptibilities(filename);
}

};	//End of namespace TBTK

#endif
