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
 *  @file Suscesptibility.h
 *  @brief Property container for density
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_SUSCEPTIBILITY
#define COM_DAFER45_TBTK_SOLVER_SUSCEPTIBILITY

//#include "TBTK/RPA/DualIndex.h"
//#include "TBTK/IndexedDataTree.h"
//#include "TBTK/InteractionAmplitude.h"
#include "TBTK/RPA/MomentumSpaceContext.h"
//#include "TBTK/Resource.h"
//#include "TBTK/SerializableVector.h"
//#include "TBTK/UnitHandler.h"

#include <complex>
#include <vector>

//#include <omp.h>

namespace TBTK{
namespace Solver{

class Susceptibility : public Solver{
public:
	/** List of algorithm identifiers. Officilly supported algorithms are
	 *  given unique identifiers. Algorithms not (yet) supported should
	 *  make sure they use an identifier that does not clash with the
	 *  officially supported ones. [ideally a large random looking number
	 *  (magic number) to also minimize accidental clashes with other
	 *  algorithms that are not (yet) supported. */
	enum Algorithm {
		Lindhard = 0,
		Matsubara = 1,
		RPA = 2
	};

	/** Constructor. */
	Susceptibility(
		Algorithm algorithm,
		const MomentumSpaceContext &momentumSpaceContext
	);

	/** Destructor. */
	virtual ~Susceptibility();

	/** Create slave SusceptibilityCalcuator. The slave reuses internal
	 *  lookup tables used to speed up the calculations and should not be
	 *  used after the generating master have been destructed. */
	virtual Susceptibility* createSlave() = 0;

	/** Calculate the susceptibility. */
	virtual std::vector<std::complex<double>> calculateSusceptibility(
		const Index &index,
		const std::vector<std::complex<double>> &energies
	) = 0;

	const MomentumSpaceContext& getMomentumSpaceContext() const;

	/** Generate lookup table for the k+q linear index. Can be called
	 *  repeatedly, and the lookup table is only generated the first time.
	 */
	void generateKPlusQLookupTable();

	/** Get the algorithm used to calculate the susceptibility. */
	Algorithm getAlgorithm() const;

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
protected:
	/** Slave constructor. */
	Susceptibility(
		Algorithm algorithm,
		const MomentumSpaceContext &momentumSpaceContext,
		int *kPlusQLookupTable
	);

	/** Returns true if the Susceptibility is a master. */
	bool getIsMaster() const;

	/** Returns the k+q lookup table. */
	int* getKPlusQLookupTable();

	/** Returns the k+q lookup table. */
	const int* getKPlusQLookupTable() const;

	/** Returns the linear index for k+q. */
	template<bool useKPlusQLookupTable>
	int getKPlusQLinearIndex(
		unsigned int meshIndex,
		const std::vector<double> &k,
		int kLinearIndex
	) const;
private:
	/** Algorithm. */
	Algorithm algorithm;

	/** Flag indicating whether the the energies in
	 *  susceptibilityEnergies are inversion symmetric in the
	 *  complex plane. */
	bool energiesAreInversionSymmetric;

	/** Momentum space context. */
	const MomentumSpaceContext *momentumSpaceContext;

	/** Lookup table for calculating k+q. */
	int *kPlusQLookupTable;

	/** Flag indicating whether the Susceptibility is a master.
	 *  Masters owns resources shared between masters and slaves and is
	 *  responsible for cleaning up. */
	bool isMaster;
};

inline const MomentumSpaceContext& Susceptibility::getMomentumSpaceContext(
) const{
	return *momentumSpaceContext;
}

inline Susceptibility::Algorithm Susceptibility::getAlgorithm() const{
	return algorithm;
}

inline void Susceptibility::setEnergiesAreInversionSymmetric(
	bool energiesAreInversionSymmetric
){
	this->energiesAreInversionSymmetric = energiesAreInversionSymmetric;
}

inline bool Susceptibility::getEnergiesAreInversionSymmetric(
) const{
	return energiesAreInversionSymmetric;
}

inline bool Susceptibility::getIsMaster() const{
	return isMaster;
}

inline int* Susceptibility::getKPlusQLookupTable(){
	return kPlusQLookupTable;
}

inline const int* Susceptibility::getKPlusQLookupTable() const{
	return kPlusQLookupTable;
}

template<>
inline int Susceptibility::getKPlusQLinearIndex<false>(
	unsigned int meshIndex,
	const std::vector<double> &k,
	int kLinearIndex
) const{
	const std::vector<std::vector<double>> &mesh
		= momentumSpaceContext->getMesh();

	Index kPlusQIndex
		= momentumSpaceContext->getBrillouinZone().getMinorCellIndex(
			{mesh[meshIndex][0] + k[0], mesh[meshIndex][1] + k[1]},
			momentumSpaceContext->getNumMeshPoints()
		);
	return momentumSpaceContext->getModel().getHoppingAmplitudeSet().getFirstIndexInBlock(
		kPlusQIndex
	);
}

template<>
inline int Susceptibility::getKPlusQLinearIndex<true>(
	unsigned int meshIndex,
	const std::vector<double> &k,
	int kLinearIndex
) const{
	return kPlusQLookupTable[
		meshIndex*momentumSpaceContext->getMesh().size()
		+ kLinearIndex/momentumSpaceContext->getNumOrbitals()
	];
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
