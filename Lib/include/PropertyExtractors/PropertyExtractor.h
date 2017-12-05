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
 *  @file PropertyExtractor.h
 *  @brief Base class PropertyExtractors
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR

#include "AbstractProperty.h"
#include "Density.h"
#include "DOS.h"
#include "HoppingAmplitudeSet.h"
#include "Index.h"
#include "LDOS.h"
#include "Magnetization.h"
#include "SpinPolarizedLDOS.h"

#include <complex>
#include <initializer_list>

namespace TBTK{

/** The PropertyExtractor extracts is a base class for derived
 *  PropertyExtractors that are used to extract common physical properties such
 *  as DOS, Density, LDOS, etc. from a Solvers. */
class PropertyExtractor{
public:
	/** Constructor. */
	PropertyExtractor();

	/** Destructor. */
	virtual ~PropertyExtractor();

	/** Set the energy window used for energy dependent quantities. */
	virtual void setEnergyWindow(
		double lowerBound,
		double upperBound,
		int energyResolution
	);

	/** Calculate density.
	 *
	 *  @param pattern Specifies the index pattern for which to calculate
	 *  the density. For example, assume that the index scheme is
	 *  {x, y, z, spin}. {ID_X, 5, 10, IDX_SUM_ALL} will calculate the
	 *  density for each x along (y,z)=(5,10) by summing over spin.
	 *  Similarly {ID_X, 5, IDX_Y, IDX_SUM_ALL} will return a two
	 *  dimensional density for all x and z and y = 5. Note that IDX_X
	 *  IDX_Y, and IDX_Z refers to the first, second, and third index used
	 *  by the routine to create a one-, two-, or three-dimensional output,
	 *  rather than being tied to the x, y, and z used as physical
	 *  subindices.
	 *
	 *  @param ranges Speifies the number of elements for each subindex. Is
	 *   ignored for indices specified with positive integers in the
	 *  pattern, but is used to loop from 0 to the value in ranges for
	 *  IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A density array with size equal to the number of points
	 *  included by specified patter-range combination.
	 */
	virtual Property::Density calculateDensity(
		Index pattern,
		Index ranges
	);

	/** Calculate density. */
	virtual Property::Density calculateDensity(
		std::initializer_list<Index> patterns
	);

	/** Calculate magnetization.
	 *
	 *  @param pattern Specifies the index pattern for which to calculate
	 *  the magnetization. For example, assume that the index scheme is
	 *  {x, y, z, spin}. {ID_X, 5, 10, IDX_SPIN} will calculate the
	 *  magnetization for each x along (y,z)=(5,10). Similarly
	 *  {ID_X, 5, IDX_Y, IDX_SPIN} will return a two dimensional
	 *  magnetiation for all x and z and y = 5. Note that IDX_X, IDX_Y, and
	 *  IDX_Z refers to the first, second, and third index used by the
	 *  routine to create a one-, two-, or three-dimensional output, rather
	 *  than being tied to the x, y, and z used as physical subindices.
	 *
	 *  @param ranges Speifies the number of elements for each subindex. Is
	 *  ignored for indices specified with positive integers in the
	 *  pattern, but is used to loop from 0 to the value in ranges for
	 *  IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A magnetization array with size equal to four times the
	 *  number of points included by specified patter-range combination.
	 *  The four entries are
	 *  \f[
	 *      \left[\begin{array}{cc}
	 *          0   & 1\\
	 *          2   & 3
	 *      \end{array}\right] =
	 *      \left[\begin{array}{cc}
	 *          \langle c_{i\uparrow}^{\dagger}c_{i\uparrow}\rangle         & \langle c_{i\uparrow}^{\dagger}c_{i\downarrow}\rangle\\
	 *          \langle c_{i\downarrow}^{\dagger}c_{u\uparrow}\rangle       & \langle c_{i\downarrow}^{\dagger}c_{i\downarrow}\rangle
	 *      \end{array}\right].
	 *  \f]
	 */
	virtual Property::Magnetization calculateMagnetization(
		Index pattern,
		Index ranges
	);

	/** Calculate Magnetization. */
	virtual Property::Magnetization calculateMagnetization(
		std::initializer_list<Index> patterns
	);

	/** Calculate local density of states.
	 *
	 *  @param pattern Specifies the index pattern for which to calculate
	 *  the LDOS. For example, assume that the index scheme is
	 *  {x, y, z, spin}. {ID_X, 5, 10, IDX_SUM_ALL} will calculate the
	 *  LDOS for each x along (y,z)=(5,10) by summing over spin. Similarly
	 *  {ID_X, 5, IDX_Y, IDX_SUM_ALL} will return a two dimensional LDOS
	 *  for all x and z and y = 5. Note that IDX_X, IDX_Y, and IDX_Z refers
	 *  to the first, second, and third index used by the routine to create
	 *  a one-, two-, or three-dimensional output, rather than being tied
	 *  to the x, y, and z used as physical subindices.
	 *
	 *  @param ranges Speifies the number of elements for each subindex. Is
	 *  ignored for indices specified with positive integers in the
	 *  pattern, but is used to loop from 0 to the value in ranges for
	 *  IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A density array with size equal to the number of points
	 *  included by specified patter-range combination.
	 */
	virtual Property::LDOS calculateLDOS(Index pattern, Index ranges);

	/** Calculate local density of states. */
	virtual Property::LDOS calculateLDOS(
		std::initializer_list<Index> patterns
	);

	/** Calculate spin-polarized local density of states.
	 *
	 *  @param pattern Specifies the index pattern for which to calculate
	 *  the spin-polarized LDOS. For example, assume that the index scheme
	 *  is {x, y, z, spin}. {ID_X, 5, 10, IDX_SPIN} will calculate the
	 *  spin-polarized LDOS for each x along (y,z)=(5,10). Similarly
	 *  {ID_X, 5, IDX_Y, IDX_SPIN} will return a two dimensional
	 *  spin-polarized LDOS for all x and z and y = 5. Note that IDX_X,
	 *  IDX_Y, and IDX_Z refers to the first, second, and third index used
	 *  by the routine to create a one-, two-, or three-dimensional output,
	 *  rather than being tied to the x, y, and z used as physical
	 *  subindices.
	 *
	 *  @param ranges Speifies the number of elements for each subindex. Is
	 *  ignored for indices specified with positive integers in the
	 *  pattern, but is used to loop from 0 to the value in ranges for
	 *  IDX_X, IDX_Y, IDX_Z, and IDX_SUM_ALL. Appropriate ranges
	 *  corresponding to the two pattern examples above are
	 *  {SIZE_X, 1, 1, NUM_SPINS} and {SIZE_X, 1, SIZE_Z, NUM_SPINS},
	 *  respectively.
	 *
	 *  @return A spin-polarized LDOS array with size equal to four times
	 *  the number of points included by specified patter-range
	 *  combination.
	 *  The four entries are
	 *  \f[
	 *      \left[\begin{array}{cc}
	 *          0   & 1\\
	 *          2   & 3
	 *      \end{array}\right] =
	 *      \left[\begin{array}{cc}
	 *          \rho_{i\uparrow i\uparrow}(E)       & \rho_{i\uparrow i\downarrow}(E)\\
	 *          \rho_{i\downarrow i\uparrow}(E)     & \rho_{i\downarrow i\downarrow}(E)\\
	 *      \end{array}\right],
	 *  \f]
	 *  where
	 *  \f[
	 *      \rho_{i\sigma i\sigma'}(E) = \sum_{E_n}\langle\Psi_n|c_{i\sigma}^{\dagger}c_{i\sigma'}|\Psi_n\rangle\delta(E - E_n) .
	 *  \f]
	 */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges
	);

	/** Calculate spin-polarized local density of states. */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		std::initializer_list<Index> patterns
	);

	/** Calculate expectation value. */
	virtual std::complex<double> calculateExpectationValue(
		Index to,
		Index from
	);

	/** Calculate density of states. */
	virtual Property::DOS calculateDOS();
protected:
	/** Default energy resolution. */
	static constexpr int ENERGY_RESOLUTION = 1000;

	/** Energy resolution used for energy dependent quantities. */
	int energyResolution;

	/** Default lower bound. */
	static constexpr double LOWER_BOUND = -1.;

	/** Lower bound used for energy dependent quantities. */
	double lowerBound;

	/** Default upper bound. */
	static constexpr double UPPER_BOUND = 1.;

	/** Upper bound used for energy dependent quantities. */
	double upperBound;

	/** Loops over range indices and calls the appropriate callback
	 *  function to calculate the correct quantity. */
	void calculate(
		void (*callback)(
			PropertyExtractor *cb_this,
			void *memory,
			const Index &index,
			int offset
		),
		void *memory,
		Index pattern,
		const Index &ranges,
		int currentOffset,
		int offsetMultiplier
	);

	/** Loops over the indices satisfying the specified patterns and calls
	 *  the appropriate callback function to calculate the correct
	 *  quantity. */
	template<typename DataType>
	void calculate(
		void (*callback)(
			PropertyExtractor *cb_this,
			void *memory,
			const Index &index,
			int offset
		),
		const IndexTree &allIndices,
		const IndexTree &memoryLayout,
		Property::AbstractProperty<DataType> &abstractProperty,
		int *spinIndexHint = nullptr
	);

	/** Hint used to pass information between calculate[Property] and
	 *  calculate[Property]Callback. */
	void *hint;

	/** Ensure that range indices are on compliant format. (Set range to
	 *  one for indices with non-negative pattern value.) */
	void ensureCompliantRanges(const Index &pattern, Index &ranges);

	/** Extract ranges for loop indices. */
	void getLoopRanges(
		const Index &pattern,
		const Index &ranges,
		int *lDimensions,
		int **lRanges
	);

	/** Generate IndexTree. */
	IndexTree generateIndexTree(
		std::initializer_list<Index> patterns,
		const HoppingAmplitudeSet &hoppingAmplitudeSet,
		bool keepSumationWildcards,
		bool keepSpinWildcards
	);
};

template<typename DataType>
void PropertyExtractor::calculate(
	void (*callback)(
		PropertyExtractor *cb_this,
		void *memory,
		const Index &index,
		int offset
	),
	const IndexTree &allIndices,
	const IndexTree &memoryLayout,
	Property::AbstractProperty<DataType> &abstractProperty,
	int *spinIndexHint
){
/*	IndexTree::Iterator it = allIndices.begin();
	const Index *index;
//	int counter = 0;
	while((index = it.getIndex())){
		if(spinIndexHint != nullptr){
			std::vector<unsigned int> spinIndices = memoryLayout.getSubindicesMatching(
				IDX_SPIN,
				*index,
				IndexTree::SearchMode::MatchWildcards
			);
			TBTKAssert(
				spinIndices.size() == 1,
				"PropertyExtractor::calculate()",
				"Zero or several spin indeces found.",
				"Use IDX_SPIN once and only once per pattern to indicate spin index."
			);
			*spinIndexHint = spinIndices.at(0);
		}

		callback(
			this,
			abstractProperty.getDataRW(),
			*index,
			abstractProperty.getOffset(*index)
		);

		it.searchNext();
	}*/
	IndexTree::Iterator it = allIndices.begin();
	while(!it.getHasReachedEnd()){
		Index index = it.getIndex();
		if(spinIndexHint != nullptr){
			std::vector<unsigned int> spinIndices = memoryLayout.getSubindicesMatching(
				IDX_SPIN,
				index,
				IndexTree::SearchMode::MatchWildcards
			);
			TBTKAssert(
				spinIndices.size() == 1,
				"PropertyExtractor::calculate()",
				"Zero or several spin indeces found.",
				"Use IDX_SPIN once and only once per pattern to indicate spin index."
			);
			*spinIndexHint = spinIndices.at(0);
		}

		callback(
			this,
			abstractProperty.getDataRW(),
			index,
			abstractProperty.getOffset(index)
		);

		it.searchNext();
	}
}

};	//End of namespace TBTK

#endif
