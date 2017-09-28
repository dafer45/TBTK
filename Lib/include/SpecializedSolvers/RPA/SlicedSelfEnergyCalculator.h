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
 *  @file SlicedSelfEnergyCalculator.h
 *  @brief Calculates the self-energy using the RPA approximation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SLICED_SELF_ENERGY_CALCULATOR
#define COM_DAFER45_TBTK_SLICED_SELF_ENERGY_CALCULATOR

#include "BrillouinZone.h"
#include "BlockDiagonalizationSolver.h"
#include "BPropertyExtractor.h"
#include "IndexedDataTree.h"
#include "SusceptibilityCalculator.h"

namespace TBTK{

class SlicedSelfEnergyCalculator{
public:
	/** Constructor. */
	SlicedSelfEnergyCalculator(
		const MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	~SlicedSelfEnergyCalculator();

	/** Get momentum cpsace context. */
	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Initialize the SlicedSelfEnergyCalculator. */
	void init();

	/** Enum class for indicating whether the energy is an arbitrary comlex
	 *  number, or if it is restricted to the real or imaginary axis. */
	enum class EnergyType {Real, Imaginary, Complex};

	/** Set number of energies. */
	void setNumSummationEnergies(
		unsigned int numSummationEnergies,
		unsigned int numSlices = 1,
		unsigned int slice = 0
	);

	/** Set self-energy energy type. */
	void setSelfEnergyEnergyType(EnergyType energyType);

	/** Get self-energy energy type. */
	EnergyType getSelfEnergyEnergyType() const;

	/** Set the energies for which the self-energy should be
	 *  calculated. */
	void setSelfEnergyEnergies(
		const std::vector<std::complex<double>> &selfEnergyEnergies
	);

	/** Calculate self-energy. */
	std::vector<std::complex<double>> calculateSelfEnergy(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
	);

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

	/** Precompute susceptibilities. */
	void precomputeSusceptibilities(unsigned int numWorkers = 128);

	/** Save susceptibilities. */
	void saveSusceptibilities(const std::string &filename) const;

	/** Load susceptibilities. */
	void loadSusceptibilities(const std::string &filename);
private:
	/** SusceptibilityCalculator. */
	SusceptibilityCalculator susceptibilityCalculator;

	/** Lookup table for calculating k+q. */
	int *kMinusQLookupTable;

	/** Generate lookup table for the k-q linear index. Can be called
	 *  repeatedly, and the lookup table is only generated once. */
	void generateKMinusQLookupTable();

	/** Returns the linear index for k+q. */
	template<bool useKPlusKLookupTable>
	int getKMinusQLinearIndex(
		unsigned int meshIndex,
		const std::vector<double> &k,
		int kLinearIndex
	) const;

	/** Number of energies to sum over. */
	unsigned int numSummationEnergies;

	/** Number of slices to divide the summation energies over. */
	unsigned int numSlices;

	/** Slice of the summation energies handled by this SlicedSelfEnergyCalculator. */
	unsigned int slice;

	/** Energies to sum over. */
	std::vector<std::complex<double>> summationEnergies;

	/** Energy type for the slef-energy. */
	EnergyType selfEnergyEnergyType;

	/** Energies to calculate the self-energy for. */
	std::vector<std::complex<double>> selfEnergyEnergies;

	/** Flag indicating whether the SelfEnergyCalculator is initialized. */
	bool isInitialized;

	/** IndexedDataTree storing the self-energy. */
	IndexedDataTree<SerializeableVector<std::complex<double>>> selfEnergyTree;

	/** IndexedDataTree storing the self-energy vertex. */
	IndexedDataTree<SerializeableVector<std::complex<double>>> selfEnergyVertexTree;

	/** Invert matix. */
	void invertMatrix(
		std::complex<double> *matrix,
		unsigned int dimensions
	);

	/** Multiply matrices. */
	void multiplyMatrices(
		std::complex<double> *matrix1,
		std::complex<double> *matrix2,
		std::complex<double> *result,
		unsigned int dimensions
	);

	/** Print matrix. */
	void printMatrix(
		std::complex<double> *matrix,
		unsigned int dimensions
	);

	/** Self-energy main loop. */
	void selfEnergyMainLoop(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::vector<std::complex<double>> &result
	);

	/** Interaction parameters. */
	std::complex<double> U, Up, J, Jp;

	/** Interaction amplitudes used to calculate the self-energy vertex. */
	std::vector<InteractionAmplitude> u1;
	std::vector<InteractionAmplitude> u2;
	std::vector<InteractionAmplitude> u3;

	/** Flag indicating whether the interaction amplitudes are initialized.
	 */
	bool interactionAmplitudesAreGenerated;

	/** Generate interaction amplitudes. Can be called multiple times and
	 *  will only regenerate the interaction amplitudes when needed. */
	void generateInteractionAmplitudes();
};

inline const MomentumSpaceContext& SlicedSelfEnergyCalculator::getMomentumSpaceContext(
) const{
	return susceptibilityCalculator.getMomentumSpaceContext();
}

inline void SlicedSelfEnergyCalculator::setNumSummationEnergies(
	unsigned int numSummationEnergies,
	unsigned int numSlices,
	unsigned int slice
){
	this->numSummationEnergies = numSummationEnergies;
}

inline void SlicedSelfEnergyCalculator::setSelfEnergyEnergyType(
	EnergyType energyType
){
	selfEnergyEnergyType = energyType;
}

inline SlicedSelfEnergyCalculator::EnergyType SlicedSelfEnergyCalculator::getSelfEnergyEnergyType(
) const{
	return selfEnergyEnergyType;
}

inline void SlicedSelfEnergyCalculator::setSelfEnergyEnergies(
	const std::vector<std::complex<double>> &selfEnergyEnergies
){
	this->selfEnergyEnergies = selfEnergyEnergies;
	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
}

inline void SlicedSelfEnergyCalculator::setU(std::complex<double> U){
	this->U = U;
	susceptibilityCalculator.setU(U);
	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void SlicedSelfEnergyCalculator::setUp(std::complex<double> Up){
	this->Up = Up;
	susceptibilityCalculator.setUp(Up);
	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void SlicedSelfEnergyCalculator::setJ(std::complex<double> J){
	this->J = J;
	susceptibilityCalculator.setJ(J);
	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void SlicedSelfEnergyCalculator::setJp(std::complex<double> Jp){
	this->Jp = Jp;
	susceptibilityCalculator.setJp(Jp);
	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void SlicedSelfEnergyCalculator::precomputeSusceptibilities(
	unsigned int numWorkers
){
	susceptibilityCalculator.precompute(numWorkers);
}

inline void SlicedSelfEnergyCalculator::saveSusceptibilities(
	const std::string &filename
) const{
	susceptibilityCalculator.saveSusceptibilities(filename);
}

inline void SlicedSelfEnergyCalculator::loadSusceptibilities(
	const std::string &filename
){
	susceptibilityCalculator.loadSusceptibilities(filename);
}

};	//End of namespace TBTK

#endif
