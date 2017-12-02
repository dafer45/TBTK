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

#include <complex>

#include <omp.h>

namespace TBTK{

class SusceptibilityCalculator{
public:
	/** Constructor. */
	SusceptibilityCalculator(
		const MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	~SusceptibilityCalculator();

	/** Create slave SusceptibilityCalcuator. The slave reuses internal
	 *  lookup tables used to speed up the calculations and should not be
	 *  used after the generating master have been destructed. */
	SusceptibilityCalculator* createSlave();

	/** Precompute susceptibilities. Will calculate the susceptibility for
	 *  all values using a parallel algorithm. Can speed up calculations if
	 *  most of the susceptibilities are needed. */
	void precompute(unsigned int numWorkers = 129);

	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Generate lookup table for the k+q linear index. Can be called
	 *  repeatedly, and the lookup table is only generated the first time.
	 */
	void generateKPlusQLookupTable();

	/** Calculation modes. */
	enum class Mode {Lindhard, Matsubara};

	/** Enum class for indicating whether the energy is an arbitrary comlex
	 *  number, or if it is restricted to the real or imaginary axis. */
	enum class EnergyType {Real, Imaginary, Complex};

	/** Set the mode used to calculate the susceptibility. */
	void setMode(Mode mode);

	/** Get the mode used to calculate the susceptibility. */
	Mode getMode() const;

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

	/** Save susceptibilities. */
	void saveSusceptibilities(const std::string &filename) const;

	/** Load susceptibilities. */
	void loadSusceptibilities(const std::string &filename);
private:
	/** IndexedDataTree storing the bare susceptibilities. */
	IndexedDataTree<SerializeableVector<std::complex<double>>> susceptibilityTree;

	/** Mode to use for calculating the susceptibility. */
	Mode mode;

	/** Energy type for the susceptibility. */
	EnergyType energyType;

	/** Energies to calculate the susceptibility for. */
	std::vector<std::complex<double>> energies;

	/** Flag indicating whether the the energies in
	 *  susceptibilityEnergies are inversion symmetric in the
	 *  complex plane. */
	bool energiesAreInversionSymmetric;

	/** Flag indicating whether the susceptibility is known to only
	 *  be evaluated at points away from poles. */
	bool susceptibilityIsSafeFromPoles;
public:
	/** Calculate the susceptibility. */
	std::complex<double> calculateSusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::complex<double> energy,
		Mode mode
	);

	/** Calculate the susceptibility. */
	std::vector<std::complex<double>> calculateSusceptibility(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Calculate the susceptibility. */
	std::vector<std::complex<double>> calculateSusceptibility(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices
	);
private:
	/** Momentum space context. */
	const MomentumSpaceContext *momentumSpaceContext;

	/** Lookup table for calculating k+q. */
	int *kPlusQLookupTable;

	/** Fermi-Dirac distribution lookup table. */
	double *fermiDiracLookupTable;

	/** Flag indicating whether the SusceptibilityCalculator is
	 *  initialized. */
	bool isInitialized;

	/** Flag indicating whether the SusceptibilityCalculator is a master.
	 *  Masters owns resources shared between masters and slaves and is
	 *  responsible for cleaning up. */
	bool isMaster;

	/** Slave constructor. */
	SusceptibilityCalculator(
		const MomentumSpaceContext &momentumSpaceContext,
		int *kPlusQLookupTable,
		double *fermiDiracLookupTable
	);

	/** Get Susceptibility result Index. */
	Index getSusceptibilityResultIndex(
		const Index &kIndex,
		const std::vector<int> &orbitalIndices
	) const;

	/** Calculate the susceptibility using the Lindhard function. */
	std::complex<double> calculateSusceptibilityLindhard(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::complex<double> energy
	);

	/** Calculate the susceptibility using the Lindhard function. */
	template<bool useKPlusQLookupTable, bool isSafeFromPoles>
	std::vector<std::complex<double>> calculateSusceptibilityLindhard(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Calculate the susceptibility using the Matsubara sum. */
	std::complex<double> calculateSusceptibilityMatsubara(
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		std::complex<double> energy
	);

	/** Calculate the susceptibility using the Matsubara sum. */
	std::vector<std::complex<double>> calculateSusceptibilityMatsubara(
		const DualIndex &kDual,
		const std::vector<int> &orbitalIndices
	);

	/** Returns the linear index for k+q. */
	template<bool useKPlusQLookupTable>
	int getKPlusQLinearIndex(
		unsigned int meshIndex,
		const std::vector<double> &k,
		int kLinearIndex
	) const;

	/** Get polt times two Fermi functions for use in the Linhard
	 *  function. */
	std::complex<double> getPoleTimesTwoFermi(
		std::complex<double> energy,
		double e2,
		double e1,
		double chemicalPotential,
		double temperature,
		int kPlusQLinearIndex,
		unsigned int meshPoint,
		unsigned int state2,
		unsigned int state1,
		unsigned int numOrbitals
	) const;

	/** Cache susceptibility. */
	void cacheSusceptibility(
		const std::vector<std::complex<double>> &result,
		const std::vector<double> &k,
		const std::vector<int> &orbitalIndices,
		const Index &kIndex,
		const Index &resultIndex
	);

	/** Lock used to prevent multiple threads from executing the main part
	 *  of SusceptibilityCalculator::calculateSusceptibilityLindhard()
	 *  simultaneously. */
	omp_lock_t lindhardLock;
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

inline void SusceptibilityCalculator::setMode(
	Mode mode
){
	this->mode = mode;
}

inline SusceptibilityCalculator::Mode SusceptibilityCalculator::getMode(
) const{
	return mode;
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

inline void SusceptibilityCalculator::setEnergiesAreInversionSymmetric(
	bool energiesAreInversionSymmetric
){
	this->energiesAreInversionSymmetric = energiesAreInversionSymmetric;
}

inline bool SusceptibilityCalculator::getEnergiesAreInversionSymmetric(
) const{
	return energiesAreInversionSymmetric;
}

inline void SusceptibilityCalculator::setSusceptibilityIsSafeFromPoles(
	bool susceptibilityIsSafeFromPoles
){
	this->susceptibilityIsSafeFromPoles = susceptibilityIsSafeFromPoles;
}

inline bool SusceptibilityCalculator::getSusceptibilityIsSafeFromPoles() const{
	return susceptibilityIsSafeFromPoles;
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

};	//End of namespace TBTK

#endif
