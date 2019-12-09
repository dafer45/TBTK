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


/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file ReciprocalLattice.h
 *  @brief A ReciprocalLattice allows for the creation of a momentum space
	Model from a UnitCells.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_RECIPROCAL_LATTICE
#define COM_DAFER45_TBTK_RECIPROCAL_LATTICE

#include "TBTK/Model.h"
#include "TBTK/StateTreeNode.h"
#include "TBTK/UnitCell.h"

#include <initializer_list>
#include <vector>

namespace TBTK{

/** The ReciprocalLattice provides methods for constructing momentum space
 *  Models from a UnitCell. Let the real space Hamiltonian be written on the
 *  form
 *  \f[
 *	H = \sum_{RiR'i'}a_{RiR'i'}c_{Ri'}^{\dagger}c_{R'i'},
 *  \f]
 *  where R and R' are UnitCell-indices and i and i' are intra
 *  UnitCell-indices. Expanding the operators in the momentum basis we have
 *  \f[
 *	H = \sum_{RkiR'k'i'}a_{RiR'i'}c_{ki}^{\dagger}c_{k'i'}e^{i(k\cdot R - k'\cdot R')}.
 *  \f]
 *  Now assuming translational invaraince, such that the coefficients only
 *  depend on the relative UnitCell positions, we can write
 *  \f[
 *	H = \sum_{\bar{R}kiR'k'i'}a_{\bar{R}i0i'}c_{ki}^{\dagger}c_{k'i'}e^{ik\cdot\bar{R}}e^{i(k-k')\cdot R'},
 *  \f]
 *  where \f[\bar{R} = R - R'\f].
 *  Carrying out the sum over R' we arrive at
 *  \f[
 *	H = \sum_{kik'i'}\left(\sum_{\bar{R}}a_{\bar{R}i0i'}e^{ik\cdot\bar{R}}\right)c_{ki}^{\dagger}c_{k'i'}.
 *  \f]
 *  It therefore follows that the momentum space HoppingAmplitudes are given by
 *  \f[
 *	a_{kik'i'} = \sum_{\bar{R}}a_{\bar{R}i0i'}e^{ik\cdot\bar{R}}.
 *  \f]
 *  The purpose of this class is to provide method for constructing Models for
 *  given k and k' using these coefficients. This is done by specifying a real
 *  space environment around a reference UnitCell, large enough to ensure that
 *  the sum can run over all relevant \f[\bar{R}\f], and then using this to
 *  calcualte the coefficeints when Models with given k and k' is demanded.
 **/
class ReciprocalLattice{
public:
	/** Constructor. */
	ReciprocalLattice(UnitCell *unitCell/*, std::initializer_list<int> size*/);

	/** Destructor. */
	~ReciprocalLattice();

	/** Genearates a Model for give momentum. */
	Model* generateModel(std::initializer_list<double> momentum) const;

	/** Genearates a Model for give momentum. */
	Model* generateModel(std::vector<double> momentum) const;

	/** Genearates a Model for give momentum. */
	Model* generateModel(
		const std::vector<std::vector<double>> &momentums,
		const std::vector<Index> &blockIndices
	) const;

	/** Get reciprocal lattice vectors. */
	const std::vector<std::vector<double>>& getReciprocalLatticeVectors() const;

	/** Get number of bands. */
	unsigned int getNumBands() const;
private:
	/** Unit cell used to create reciprocal Model. */
	UnitCell *unitCell;

	/** A real space lattice constructed from the unit cell that is large
	 *  enough to allow for calculation of all relevant real space matrix
	 *  elements. The real space lattice is used to determine the matrix
	 *  elements before they are Fourier transformed to k-space. */
	StateSet *realSpaceEnvironment;

	/** StateTreeNode for quick access of states in realSpaceEnvironment. */
	StateTreeNode *realSpaceEnvironmentStateTree;

	/** StateSet containing the subset of states of realSpaceEnvironment
	 *  that belongs to the reference unit cell. That is, the cell at the
	 *  center of the realSpaceEnvironment, which contains the states used
	 *  as kets when calculating matrix elements. */
	StateSet *realSpaceReferenceCell;

	/** Reciprocal lattice vectors. */
	std::vector<std::vector<double>> reciprocalLatticeVectors;

	/** Constant used to provide a margin that protects from roundoff
	 *  errors. */
	static constexpr double ROUNDOFF_MARGIN_MULTIPLIER = 1.01;

	/** Setup reciprocal lattice vectors. */
	void setupReciprocalLatticeVectors(const UnitCell *unitCell);

	/** Setup real space environment. */
	void setupRealSpaceEnvironment(const UnitCell *unitCell);
};

inline const std::vector<std::vector<double>>& ReciprocalLattice::getReciprocalLatticeVectors() const{
	return reciprocalLatticeVectors;
}

inline unsigned int ReciprocalLattice::getNumBands() const{
	return unitCell->getNumStates();
}

};	//End of namespace TBTK

#endif
/// @endcond
