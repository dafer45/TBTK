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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file SelfEnergyCalculator.h
 *  @brief Calculates the self-energy using the RPA approximation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SELF_ENERGY_CALCULATOR
#define COM_DAFER45_TBTK_SELF_ENERGY_CALCULATOR

#include "TBTK/BrillouinZone.h"
#include "TBTK/IndexedDataTree.h"
#include "TBTK/RPA/ElectronFluctuationVertexCalculator.h"

namespace TBTK{

class SelfEnergyCalculator{
public:
	/** Constructor. */
	SelfEnergyCalculator(
		const RPA::MomentumSpaceContext &momentumSpaceContext,
		unsigned int numWorkers
	);

	/** Destructor. */
	~SelfEnergyCalculator();

	/** Get momentum cpsace context. */
	const RPA::MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Initialize the SelfEnergyCalculator. */
	void init();

	/** Enum class for indicating whether the energy is an arbitrary comlex
	 *  number, or if it is restricted to the real or imaginary axis. */
//	enum class EnergyType {Real, Imaginary, Complex};

	/** Set number of energies. */
	void setNumSummationEnergies(unsigned int numSummationEnergies);

	/** Set energy type. */
//	void setEnergyType(EnergyType energyType);

	/** Get self-energy energy type. */
//	EnergyType getEnergyType() const;

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

	/** Calculate self-energy self-consistently. */
	std::vector<std::complex<double>> calculateSelfEnergySelfConsistently(
		unsigned int numMatsubaraFrequencies
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
	/** ElectronFluctuationVertexCalculator. */
	std::vector<ElectronFluctuationVertexCalculator*> electronFluctuationVertexCalculators;

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
//	EnergyType energyType;

	/** Energies to calculate the self-energy for. */
	std::vector<std::complex<double>> selfEnergyEnergies;

	/** Flag indicating whether the SelfEnergyCalculator is initialized. */
	bool isInitialized;

	/** IndexedDataTree storing the self-energy. */
	IndexedDataTree<SerializableVector<std::complex<double>>> selfEnergyTree;

	/** Self-energy main loop. */
	template<bool singleSelfEnergyEnergy>
	void selfEnergyMainLoop(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::vector<std::complex<double>> &result
	);

	/** Interaction parameters. */
	std::complex<double> U, Up, J, Jp;
};

inline const RPA::MomentumSpaceContext& SelfEnergyCalculator::getMomentumSpaceContext(
) const{
	return electronFluctuationVertexCalculators[0]->getMomentumSpaceContext();
}

inline void SelfEnergyCalculator::setNumSummationEnergies(
	unsigned int numSummationEnergies
){
	this->numSummationEnergies = numSummationEnergies;
}

/*inline void SelfEnergyCalculator::setEnergyType(
	EnergyType energyType
){
	this->energyType = energyType;
}*/

/*inline SelfEnergyCalculator::EnergyType SelfEnergyCalculator::getEnergyType(
) const{
	return energyType;
}*/

inline void SelfEnergyCalculator::setSelfEnergyEnergies(
	const std::vector<std::complex<double>> &selfEnergyEnergies
){
	this->selfEnergyEnergies = selfEnergyEnergies;
	selfEnergyTree.clear();
}

inline void SelfEnergyCalculator::setU(std::complex<double> U){
	this->U = U;

	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		electronFluctuationVertexCalculators[n]->setU(U);
	}

	selfEnergyTree.clear();
}

inline void SelfEnergyCalculator::setUp(std::complex<double> Up){
	this->Up = Up;

	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		electronFluctuationVertexCalculators[n]->setUp(Up);
	}

	selfEnergyTree.clear();
}

inline void SelfEnergyCalculator::setJ(std::complex<double> J){
	this->J = J;

	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		electronFluctuationVertexCalculators[n]->setJ(J);
	}

	selfEnergyTree.clear();
}

inline void SelfEnergyCalculator::setJp(std::complex<double> Jp){
	this->Jp = Jp;

	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		electronFluctuationVertexCalculators[n]->setJp(Jp);
	}

	selfEnergyTree.clear();
}

/*inline void SelfEnergyCalculator::precomputeSusceptibilities(
	unsigned int numWorkers
){
	susceptibilityCalculator.precompute(numWorkers);
}*/

inline void SelfEnergyCalculator::saveSusceptibilities(
	const std::string &filename
) const{
	size_t lastPos = filename.find_last_of('/');
	std::string path;
	std::string fname = filename;
	if(lastPos != std::string::npos){
		path = filename.substr(0, lastPos+1);
		fname = filename.substr(lastPos+1, filename.size());
	}

	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		electronFluctuationVertexCalculators[n]->saveSusceptibilities(
                        path + "Slice" + std::to_string(n) + "_" + fname
		);
	}
}

inline void SelfEnergyCalculator::loadSusceptibilities(
	const std::string &filename
){
	size_t lastPos = filename.find_last_of('/');
	std::string path;
	std::string fname = filename;
	if(lastPos != std::string::npos){
		path = filename.substr(0, lastPos+1);
		fname = filename.substr(lastPos+1, filename.size());
	}

	for(
		unsigned int n = 0;
		n < electronFluctuationVertexCalculators.size();
		n++
	){
		electronFluctuationVertexCalculators[n]->loadSusceptibilities(
                        path + "Slice" + std::to_string(n) + "_" + fname
		);
	}
}

};	//End of namespace TBTK

#endif
/// @endcond
