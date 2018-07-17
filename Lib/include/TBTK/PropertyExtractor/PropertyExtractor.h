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

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_PROPERTY_EXTRACTOR
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_PROPERTY_EXTRACTOR

#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/Index.h"
#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/Property/Density.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"

#include <complex>
#include <initializer_list>

namespace TBTK{
namespace PropertyExtractor{

/** The PropertyExtractor extracts is a base class for derived
 *  PropertyExtractors that are used to extract common physical properties such
 *  as DOS, Density, LDOS, etc. from a Solvers. */
class PropertyExtractor{
public:
	/** Constructs a PropertyExtractor::PropertyExtractor. */
	PropertyExtractor();

	/** Destructor. */
	virtual ~PropertyExtractor();

	/** Set the energy window used for energy dependent quantities.
	 *
	 *  @param lowerBound The lower bound for the energy window.
	 *  @param upperBound The upper bound for the energy window.
	 *  @param energyResolution The number of energy points used to resolve
	 *  the energy window. */
	virtual void setEnergyWindow(
		double lowerBound,
		double upperBound,
		int energyResolution
	);

	/** Calculate the density. This function should be overriden by those
	 *  deriving classes that provide support for calculating the density.
	 *  By default the PropertyExtractor prints an error message that the
	 *  given property is not supported.
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
	 *  included by specified patter-range combination. */
	virtual Property::Density calculateDensity(
		Index pattern,
		Index ranges
	);

	/** Calculate the density. This function should be overriden by those
	 *  deriving classes that provide support for calculating the density.
	 *  By default the PropertyExtractor prints an error message that the
	 *  given property is not supported.
	 *
	 *  @param patterns A list of patterns that will be matched against the
	 *  @link Index Indices @endlink in the Model to determine which @link
	 *  Index Indices @endlink for which to calculate the Density.
	 *
	 *  @return A Property::Density for the @link Index Indices @endlink
	 *  that match the patterns. */
	virtual Property::Density calculateDensity(
		std::initializer_list<Index> patterns
	);

	/** Calculate the magnetization. This function should be overriden by
	 *  those deriving classes that provide support for calculating the
	 *  magnetization. By default the PropertyExtractor prints an error
	 *  message that the given property is not supported.
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
	 *  \f] */
	virtual Property::Magnetization calculateMagnetization(
		Index pattern,
		Index ranges
	);

	/** Calculate the Magnetization. This function should be overriden by
	 *  those deriving classes that provide support for calculating the
	 *  magnetization. By default the PropertyExtractor prints an error
	 *  message that the given property is not supported.
	 *
	 *  @param patterns A list of patterns that will be matched against the
	 *  @link Index Indices @endlink in the Model to determine which @link
	 *  Index Indices @endlink for which to calculate the Magnetization.
	 *
	 *  @return A Property::Magnetization for the @link Index Indices
	 *  @endlink that match the patterns. */
	virtual Property::Magnetization calculateMagnetization(
		std::initializer_list<Index> patterns
	);

	/** Calculate the local density of states. This function should be
	 *  overriden by those deriving classes that provide support for
	 *  calculating the local density of states. By default the
	 *  PropertyExtractor prints an error message that the given property
	 *  is not supported.
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
	 *  included by specified patter-range combination. */
	virtual Property::LDOS calculateLDOS(Index pattern, Index ranges);

	/** Calculate the local density of states. This function should be
	 *  overriden by those deriving classes that provide support for
	 *  calculating the local density of states. By default the
	 *  PropertyExtractor prints an error message that the given property
	 *  is not supported.
	 *
	 *  @param patterns A list of patterns that will be matched against the
	 *  @link Index Indices @endlink in the Model to determine which @link
	 *  Index Indices @endlink for which to calculate the local density of
	 *  states.
	 *
	 *  @return A Property::LDOS for the @link Index Indices @endlink that
	 *  match the patterns. */
	virtual Property::LDOS calculateLDOS(
		std::initializer_list<Index> patterns
	);

	/** Calculate the spin-polarized local density of states. This function
	 *  should be overriden by those deriving classes that provide support
	 *  for calculating the spin-polarized local density of states. By
	 *  default the PropertyExtractor prints an error message that the
	 *  given property is not supported.
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
	 *  \f] */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		Index pattern,
		Index ranges
	);

	/** Calculate the spin-polarized local density of states. This function
	 *  should be overriden by those deriving classes that provide support
	 *  for calculating the spin-polarized local density of states. By
	 *  default the PropertyExtractor prints an error message that the
	 *  given property is not supported.
	 *
	 *  @param patterns A list of patterns that will be matched against the
	 *  @link Index Indices @endlink in the Model to determine which @link
	 *  Index Indices @endlink for which to calculate the spin-polarized
	 *  local density of states.
	 *
	 *  @return A Property::SpinPolarizedLDOS for the @link Index Indices
	 *  @endlink that match the patterns. */
	virtual Property::SpinPolarizedLDOS calculateSpinPolarizedLDOS(
		std::initializer_list<Index> patterns
	);

	/** Calculate the expectation value \f$\langle
	 *  c_{to}^{\dagger}c_{from}\f$. This function should be overriden by
	 *  those deriving classes that provide support for calculating the
	 *  expecation value. By default the PropertyExtractor prints an error
	 *  message that the given property is not supported.
	 *
	 *  @param to The Index on the left operator.
	 *  @param from The index on the right operator.
	 *
	 *  @return The expectation value \f$\langle
	 *  c_{to}^{\dagger}c_{from}\f$. */
	virtual std::complex<double> calculateExpectationValue(
		Index to,
		Index from
	);

	/** Calculate the density of states. This function should be overriden
	 *  by those deriving classes that provide support for calculating the
	 *  density of states. By default the PropertyExtractor prints an error
	 *  message that the given property is not supported.
	 *
	 *  @return A Property::DOS containing the density of states. */
	virtual Property::DOS calculateDOS();

	/** Calculate the entropy. This function should be overriden by those
	 *  deriving classes that provide support for calculating the entropy.
	 *  By default the PropertyExtractor prints an error message that the
	 *  given property is not supported.
	 *
	 *  @return The entropy. */
	virtual double calculateEntropy();
protected:
	/** Get the energy resolution.
	 *
	 *  @return The energy resolution for the energy window. */
	int getEnergyResolution() const;

	/** Get lower bound for the energy window.
	 *
	 *  @return The lower bound for the energy window. */
	double getLowerBound() const;

	/** Get the upper bound for the energy window.
	 *
	 *  @return The upper bound for the energy window. */
	double getUpperBound() const;

	/** Loops over range indices and calls the given callback function to
	 *  calculate the correct quantity. The function recursively calls
	 *  itself replacing any IDX_SUM_ALL, IDX_X, IDX_Y, and IDX_Z
	 *  specifiers by actual subindices in the range [0, ranges[s]), where
	 *  s is the subindex at which the specifier appears. For example, the
	 *  pattern ranges pair {IDX_SUM_ALL, 2, IDX_X} and {2, 1, 3} will
	 *  result in the callback being called for {0, 2, 0}, {0, 2, 1}, {0,
	 *  2, 2}, {1, 2, 0}, {1, 2, 1}, and {1, 2, 2}. The first and fourth,
	 *  second and fifth, and third and sixth Index will further be passed
	 *  to the callback with the same memory offset since their result
	 *  should be summed.
	 *
	 *  The memory offset is further calculated by traversing the
	 *  subindices of the apttern from right to left and multiplying the
	 *  current offset multiplier by the number of indices in the range
	 *  size for the given subindex. This results in an offset that places
	 *  the elements in consequtive order in increasing order of the Index
	 *  order. Where an Index is considered to come before another Index if
	 *  the first subindex to differ between two @link Index Indices
	 *  @endlink from the left is smaller than the other Index.
	 *
	 *  @param callback A callback function that is called to perform the
	 *  actual calculation for a given Index.
	 *
	 *  @param memory Pointer to the memory where the result is to be
	 *  stored.
	 *
	 *  @param pattern An Index specifying the pattern for which to perform
	 *  the calculation.
	 *
	 *  @param ranges The upper limit (exclusive) for which subindices with
	 *  wildcard specifiers will be replaced. The lower limit is 0.
	 *
	 *  @param currentOffset The memory offset calculated for the given
	 *  pattern Index. Should be zero for the initial call to the function.
	 *
	 *  @param offsetMultiplier Number indicating the block size associated
	 *  with increasing the current subindex by one. Should be equal to the
	 *  number of data elements per Index for the initial call to the
	 *  function. */
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
	 *  quantity.
	 *
	 *  @param callback A callback function that is called to perform the
	 *  actual calculation for a given Index.
	 *
	 *  @param allIndices An IndexTree containing all the Indices for which
	 *  the callback should be called.
	 *
	 *  @param memoryLayout The memory layout used for the Property.
	 *  @param abstractProperty The Property that is being calculated.
	 *  @param spinIndexHint Pointer to a memory location that provides
	 *  information about which subindex that corresponds to a spin index.
	 *  If the specification of a spin subindex is not necessary for the
	 *  given Property, this should be nullptr. */
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

	/** Ensure that range indices are on compliant format. I.e., sets the
	 *  range to  one for indices with non-negative pattern value.
	 *
	 *  @param pattern The pattern.
	 *  @param ranges The ranges that will have its subindices set to one
	 *  for every pattern subindex that is non negative. */
	void ensureCompliantRanges(const Index &pattern, Index &ranges);

	/** Extract ranges for loop indices. The subindices with IDX_X, IDX_Y
	 *  and IDX_Z are identified and counted and an array of the same size
	 *  as the number of loop indices is created and filled with the ranges
	 *  for the corrsponding loop subindices.
	 *
	 *  @param pattern A pattern.
	 *  @param ranges The ranges for the given pattern.
	 *  @param loopDimensions Pointer to int that will hold the number of
	 *  loop dimensions after the call has completed.
	 *
	 *  @param loopRanges *loopRange will point to an array of size
	 *  *loopDimensions that contain the ranges for the loop subindices. */
	void getLoopRanges(
		const Index &pattern,
		const Index &ranges,
		int *loopDimensions,
		int **loopRanges
	);

	/** Generate an IndexTree containing all the @link Index Indices
	 *  @endlink in the HoppingAmplitudeSet that matches the given
	 *  patterns. Before being added to the IndexTree, the @link Index
	 *  Indices @endlink may be modified to replace subindices by their
	 *  corresponding pattern value. I.e. A summation or spin subindex may
	 *  still be labeld such in the IndexTree depending on the flags that
	 *  are passed to the function.
	 *
	 *  @param patterns List of patterns to match against.
	 *  @param The HoppingAmplitudeSet cntaining all the @link Index
	 *  Indices @endlink that will be matched against the patterns.
	 *
	 *  @param keepSummationWildcards If true, summation wildcards in the
	 *  pattern will be preserved in the IndexTree.
	 *
	 *  @param keepSpinWildcards If true, spin wildcards in the pattern
	 *  will be preserved in the IndexTree. */
	IndexTree generateIndexTree(
		std::initializer_list<Index> patterns,
		const HoppingAmplitudeSet &hoppingAmplitudeSet,
		bool keepSummationWildcards,
		bool keepSpinWildcards
	);
private:
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
};

inline int PropertyExtractor::getEnergyResolution() const{
	return energyResolution;
}

inline double PropertyExtractor::getLowerBound() const{
	return lowerBound;
}

inline double PropertyExtractor::getUpperBound() const{
	return upperBound;
}

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
	for(
		IndexTree::ConstIterator iterator = allIndices.cbegin();
		iterator != allIndices.end();
		++iterator
	){
		Index index = *iterator;
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
			abstractProperty.getDataRW().data(),
			index,
			abstractProperty.getOffset(index)
		);
	}
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
