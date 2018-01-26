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
 *  @brief Property container for density
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SUSCEPTIBILITY_CALCULATOR
#define COM_DAFER45_TBTK_SUSCEPTIBILITY_CALCULATOR

#include "DualIndex.h"
#include "IndexedDataTree.h"
#include "InteractionAmplitude.h"
#include "MomentumSpaceContext.h"
#include "Resource/Resource.h"
#include "SerializeableVector.h"
#include "UnitHandler.h"

#include <complex>

#include <omp.h>

namespace TBTK{

class SusceptibilityCalculator{
public:
	/** List of algorithm identifiers. Officilly supported algorithms are
	 *  given unique identifiers. Algorithms not (yet) supported should
	 *  make sure they use an identifier that does not clash with the
	 *  officially supported ones. [ideally a large random looking number
	 *  (magic number) to also minimize accidental clashes with other
	 *  algorithms that are not (yet) supported. */
	enum Algorithm {
		Lindhard = 0,
		Matsubara = 1
	};

	/** Constructor. */
	SusceptibilityCalculator(
		Algorithm algorithm,
		const MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	virtual ~SusceptibilityCalculator();

	/** Create slave SusceptibilityCalcuator. The slave reuses internal
	 *  lookup tables used to speed up the calculations and should not be
	 *  used after the generating master have been destructed. */
	virtual SusceptibilityCalculator* createSlave() = 0;

	/** Calculate the susceptibility. */
	virtual std::complex<double> calculateSusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::complex<double> energy
	) = 0;

	/** Calculate the susceptibility. */
	virtual std::vector<std::complex<double>> calculateSusceptibility(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	) = 0;

	/** Calculate the susceptibility. */
	std::vector<std::complex<double>> calculateSusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
	);

	/** Precompute susceptibilities. Will calculate the susceptibility for
	 *  all values using a parallel algorithm. Can speed up calculations if
	 *  most of the susceptibilities are needed. */
//	void precompute(unsigned int numWorkers = 129);

	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Generate lookup table for the k+q linear index. Can be called
	 *  repeatedly, and the lookup table is only generated the first time.
	 */
	void generateKPlusQLookupTable();

	/** Enum class for indicating whether the energy is an arbitrary comlex
	 *  number, or if it is restricted to the real or imaginary axis. */
	enum class EnergyType {Real, Imaginary, Complex};

	/** Get the algorithm used to calculate the susceptibility. */
	Algorithm getAlgorithm() const;

	/** Set energy type. */
	void setEnergyType(EnergyType energyType);

	/** Get energy type. */
	EnergyType getEnergyType() const;

	/** Set the energies for which the susceptibility should be
	 *  calculated. */
	void setEnergies(
		const std::vector<std::complex<double>> &energies
	);

	/** Get the energies for which the susceptibility should be calculated.
	 */
	const std::vector<std::complex<double>>& getEnergies() const;

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

	/** Save susceptibilities. */
	void saveSusceptibilities(const std::string &filename) const;

	/** Load susceptibilities. */
	void loadSusceptibilities(const std::string &filename);
protected:
	/** Slave constructor. */
	SusceptibilityCalculator(
		Algorithm algorithm,
		const MomentumSpaceContext &momentumSpaceContext,
		int *kPlusQLookupTable/*,
		double *fermiDiracLookupTable*/
	);

	/** Returns true if the SusceptibilityCalculator is a master. */
	bool getIsMaster() const;

	/** Returns the k+q lookup table. */
	int* getKPlusQLookupTable();

	/** Returns the k+q lookup table. */
	const int* getKPlusQLookupTable() const;

	/** Get Susceptibility result Index. */
	Index getSusceptibilityResultIndex(
		const Index &kIndex,
		const std::vector<int> &orbitalIndices
	) const;

	/** Get susceptibility tree. */
	const IndexedDataTree<SerializeableVector<std::complex<double>>>& getSusceptibilityTree() const;

	/** Returns the linear index for k+q. */
	template<bool useKPlusQLookupTable>
	int getKPlusQLinearIndex(
		unsigned int meshIndex,
		const std::vector<double> &k,
		int kLinearIndex
	) const;

	/** Cache susceptibility. */
	void cacheSusceptibility(
		const std::vector<std::complex<double>> &result,
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		const Index &kIndex,
		const Index &resultIndex
	);
private:
	/** IndexedDataTree storing the bare susceptibilities. */
	IndexedDataTree<SerializeableVector<std::complex<double>>> susceptibilityTree;

	/** Algorithm. */
	Algorithm algorithm;

	/** Energy type for the susceptibility. */
	EnergyType energyType;

	/** Energies to calculate the susceptibility for. */
	std::vector<std::complex<double>> energies;

	/** Flag indicating whether the the energies in
	 *  susceptibilityEnergies are inversion symmetric in the
	 *  complex plane. */
	bool energiesAreInversionSymmetric;

	/** Momentum space context. */
	const MomentumSpaceContext *momentumSpaceContext;

	/** Lookup table for calculating k+q. */
	int *kPlusQLookupTable;

	/** Flag indicating whether the SusceptibilityCalculator is a master.
	 *  Masters owns resources shared between masters and slaves and is
	 *  responsible for cleaning up. */
	bool isMaster;
};

inline const MomentumSpaceContext& SusceptibilityCalculator::getMomentumSpaceContext(
) const{
	return *momentumSpaceContext;
}

inline Index SusceptibilityCalculator::getSusceptibilityResultIndex(
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

inline void SusceptibilityCalculator::setEnergyType(
	EnergyType energyType
){
	this->energyType = energyType;
}

inline SusceptibilityCalculator::EnergyType SusceptibilityCalculator::getEnergyType(
) const{
	return energyType;
}

inline void SusceptibilityCalculator::setEnergies(
	const std::vector<std::complex<double>> &energies
){
	this->energies = energies;
	susceptibilityTree.clear();
}

inline const std::vector<std::complex<double>>& SusceptibilityCalculator::getEnergies() const{
	return energies;
}

inline void SusceptibilityCalculator::setEnergiesAreInversionSymmetric(
	bool energiesAreInversionSymmetric
){
	this->energiesAreInversionSymmetric = energiesAreInversionSymmetric;
}

inline bool SusceptibilityCalculator::getEnergiesAreInversionSymmetric(
) const{
	return energiesAreInversionSymmetric;
}

inline void SusceptibilityCalculator::saveSusceptibilities(
	const std::string &filename
) const{
	Resource resource;
	resource.setData(
		susceptibilityTree.serialize(Serializeable::Mode::JSON)
	);
	resource.write(filename);
}

inline void SusceptibilityCalculator::loadSusceptibilities(
	const std::string &filename
){
	Resource resource;
	resource.read(filename);
	susceptibilityTree = IndexedDataTree<SerializeableVector<std::complex<double>>>(
		resource.getData(),
		Serializeable::Mode::JSON
	);
}

inline std::vector<std::complex<double>> SusceptibilityCalculator::calculateSusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
){
	return calculateSusceptibility(
		DualIndex(
			momentumSpaceContext->getKIndex(k),
			k
		),
		orbitalIndices
	);
}

inline bool SusceptibilityCalculator::getIsMaster() const{
	return isMaster;
}

inline int* SusceptibilityCalculator::getKPlusQLookupTable(){
	return kPlusQLookupTable;
}

inline const int* SusceptibilityCalculator::getKPlusQLookupTable() const{
	return kPlusQLookupTable;
}

inline const IndexedDataTree<SerializeableVector<std::complex<double>>>& SusceptibilityCalculator::getSusceptibilityTree() const{
	return susceptibilityTree;
}

};	//End of namespace TBTK

#endif
