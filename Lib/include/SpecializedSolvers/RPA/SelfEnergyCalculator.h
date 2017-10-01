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

#ifndef COM_DAFER45_TBTK_SELF_ENERGY_CALCULATOR
#define COM_DAFER45_TBTK_SELF_ENERGY_CALCULATOR

#include "BrillouinZone.h"
#include "BlockDiagonalizationSolver.h"
#include "BPropertyExtractor.h"
#include "IndexedDataTree.h"
#include "SusceptibilityCalculator.h"

namespace TBTK{

class SelfEnergyCalculator{
public:
	/** Constructor. */
	SelfEnergyCalculator(const MomentumSpaceContext &momentumSpaceContext);

	/** Destructor. */
	~SelfEnergyCalculator();

	/** Get momentum cpsace context. */
	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Initialize the SelfEnergyCalculator. */
	void init();

	/** Enum class for indicating whether the energy is an arbitrary comlex
	 *  number, or if it is restricted to the real or imaginary axis. */
	enum class EnergyType {Real, Imaginary, Complex};

	/** Set number of energies. */
	void setNumSummationEnergies(unsigned int numSummationEnergies);

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
//	void precomputeSusceptibilities(unsigned int numWorkers = 128);

	/** Save susceptibilities. */
	void saveSusceptibilities(const std::string &filename) const;

	/** Load susceptibilities. */
	void loadSusceptibilities(const std::string &filename);
private:
	/** SusceptibilityCalculator. */
	std::vector<SusceptibilityCalculator*> susceptibilityCalculators;

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

inline const MomentumSpaceContext& SelfEnergyCalculator::getMomentumSpaceContext(
) const{
	return susceptibilityCalculators[0]->getMomentumSpaceContext();
}

inline void SelfEnergyCalculator::setNumSummationEnergies(
	unsigned int numSummationEnergies
){
	this->numSummationEnergies = numSummationEnergies;
}

inline void SelfEnergyCalculator::setSelfEnergyEnergyType(
	EnergyType energyType
){
	selfEnergyEnergyType = energyType;
}

inline SelfEnergyCalculator::EnergyType SelfEnergyCalculator::getSelfEnergyEnergyType(
) const{
	return selfEnergyEnergyType;
}

inline void SelfEnergyCalculator::setSelfEnergyEnergies(
	const std::vector<std::complex<double>> &selfEnergyEnergies
){
	this->selfEnergyEnergies = selfEnergyEnergies;
	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
}

inline void SelfEnergyCalculator::setU(std::complex<double> U){
	this->U = U;

	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++)
		susceptibilityCalculators[n]->setU(U);

	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void SelfEnergyCalculator::setUp(std::complex<double> Up){
	this->Up = Up;

	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++)
		susceptibilityCalculators[n]->setUp(Up);

	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void SelfEnergyCalculator::setJ(std::complex<double> J){
	this->J = J;

	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++)
		susceptibilityCalculators[n]->setJ(J);

	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

inline void SelfEnergyCalculator::setJp(std::complex<double> Jp){
	this->Jp = Jp;

	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++)
		susceptibilityCalculators[n]->setJp(Jp);

	selfEnergyTree.clear();
	selfEnergyVertexTree.clear();
	interactionAmplitudesAreGenerated = false;
}

/*inline void SelfEnergyCalculator::precomputeSusceptibilities(
	unsigned int numWorkers
){
	susceptibilityCalculator.precompute(numWorkers);
}*/

inline void SelfEnergyCalculator::saveSusceptibilities(
	const std::string &filename
) const{
	std::string path;
	std::string fname = filename;
	if(lastPos != std::string::npos){
		path = filename.substr(0, lastPos+1);
		fname = filename.substr(lastPos+1, filename.size());
	}

	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++){
		susceptibilityCalculators[n]->saveSusceptibilities(
                        path + "Slice" + std::to_string(n) + "_" + fname
                );
        }
}

inline void SelfEnergyCalculator::loadSusceptibilities(
	const std::string &filename
){
	std::string path;
	std::string fname = filename;
	if(lastPos != std::string::npos){
		path = filename.substr(0, lastPos+1);
		fname = filename.substr(lastPos+1, filename.size());
	}

	for(unsigned int n = 0; n < susceptibilityCalculators.size(); n++){
		susceptibilityCalculators[n]->loadSusceptibilities(
                        path + "Slice" + std::to_string(n) + "_" + fname
                );
        }
}

};	//End of namespace TBTK

#endif
