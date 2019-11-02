/* Copyright 2019 Kristofer Björnson
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
 *  @brief Generates @link IndexTree IndexTrees@endlink to be used as loop
 *  ranges and memory layout for @link Property Properties@endlink in the @link
 *  PropertyExtractor::PropertyExtractor PropertyExtractors@endlink.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_INDEX_TREE_GENERATOR
#define COM_DAFER45_TBTK_PROPERTY_EXTRACTOR_INDEX_TREE_GENERATOR

#include "TBTK/Index.h"
#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/Property/Density.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"

#include <complex>
//#include <initializer_list>

namespace TBTK{
namespace PropertyExtractor{

/** @brief Generates @link IndexTree IndexTrees@endlink to be used as loop
 *  ranges and memory layout for @link Property Properties@endlink in the @link
 *  PropertyExtractor::PropertyExtractor PropertyExtractors@endlink.
 *
 *  The IndexTreeGenerator generates an IndexTree that is compatible with a
 *  given Model and set of patterns. For example, given a Model with the Index
 *  structure {x, y, z} and a pattern {IDX_ALL, 2, IDX_ALL}, an Index tree
 *  containing all @link Index Indices@endlink in the Model that has the form
 *  {x, 2, z} can be generated.
 *
 *  # Keeping wildcards
 *  It is possible to generate IndexTree with @link Index Indices@endlink that
 *  keep one or more Subindex flags.
 *
 *  ## Summation Subindex
 *  To keep summation @link Subindex Subindices@endlink, we configure the
 *  IndexTreeGenerator using
 *  ```cpp
 *    indexTreeGenerator.setKeepSummationWildcards(true);
 *  ```
 *
 *  ## Spin Subindex
 *  To kepp spin @link Subindex Subindices@endlink, we configure the
 *  IndexTreeGenerator using
 *  ```cpp
 *    indexTreeGenerator.setKeepSpinWildcards(true);
 *  ```
 *
 *  # Example
 *  \snippet PropertyExtractor/IndexTreeGenerator.cpp IndexTreeGenerator
 *  ##Output
 *  \snippet output/PropertyExtractor/IndexTreeGenerator.output IndexTreeGenerator */
class IndexTreeGenerator{
public:
	/** Constructs a PropertyExtractor::PropertyExtractor.
	 *
	 *  @param model The Model that the generated IndexTree is compatible
	 *  with. */
	IndexTreeGenerator(const Model &model);

	/** Set whether or not to keep summation wildcards.
	 *
	 *  @param keepSummationWildcards Flag indicating whether or not to
	 *  keep summation wildcards. */
	void setKeepSummationWildcards(bool keepSummationWildcards);

	/** Set whether or not to keep spin wildcards.
	 *
	 *  @param keepSpinWildcards Flag indicating whether or not to keep
	 *  spin wildcards. */
	void setKeepSpinWildcards(bool keepSpinWildcards);

	/** Generate an IndexTree containing the @link Index Indices@endlink
	 *  that satisfies on of the patterns in the given list of patterns.
	 *
	 *  @param patterns List of patterns to match. */
	IndexTree generate(const std::vector<Index> &patterns) const;

	/** Generate an IndexTree containing all the @link Index Indices
	 *  @endlink in the HoppingAmplitudeSet that matches the given
	 *  patterns. Before being added to the IndexTree, the @link Index
	 *  Indices @endlink may be modified to replace subindices by their
	 *  corresponding pattern value. I.e. A summation or spin subindex may
	 *  still be labeld such in the IndexTree depending on the flags that
	 *  are passed to the function.
	 *
	 *  The pattern can also be a compund Index consisting of two Indices,
	 *  in which case the pattern matching is applied to each component
	 *  Index separately.
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
/*	IndexTree generateIndexTree(
		std::vector<Index> patterns,
		const HoppingAmplitudeSet &hoppingAmplitudeSet,
		bool keepSummationWildcards,
		bool keepSpinWildcards
	);*/
private:
	/** Model. */
	const Model &model;

	/** Flag indicating whether or not to keep summation wildcards. */
	bool keepSummationWildcards;

	/** Flag indicating whether or not to keep spin wildcards. */
	bool keepSpinWildcards;
};

inline void IndexTreeGenerator::setKeepSummationWildcards(
	bool keepSummationWildcards
){
	this->keepSummationWildcards = keepSummationWildcards;
}

inline void IndexTreeGenerator::setKeepSpinWildcards(bool keepSpinWildcards){
	this->keepSpinWildcards = keepSpinWildcards;
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
