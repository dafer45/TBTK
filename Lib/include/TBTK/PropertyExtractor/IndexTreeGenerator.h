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
 *  # Generate a list of all compatible @link Index Indices@endlink
 *  Assume a Mode with Index structure {x, y, z} and a list of patterns such as
 *  ```cpp
 *    std::vector<Index> patterns = {
 *      {1, IDX_ALL, IDX_SUM_ALL},
 *      {IDX_ALL, 5, IDX_SUM_ALL}
 *    };
 *  ```
 *  We can get an IndexTree that contains all @link Index Indices@endlink in
 *  the Model that are compatible with one of the patterns using
 *  ```cpp
 *    IndexTree allIndices = indexTreeGenerator.generateAllIndices(patterns);
 *  ```
 *
 *  # Generate memory layout
 *  A Property that results from summing over one or more @link Subindex
 *  Subindices@endlink is stored with the flag IDX_SUM_ALL in the Subindex that
 *  is summed over. It is possible to generate such an IndexTree using
 *  ```cpp
 *    IndexTree memoryLayout
 *      = indexTreeGenerator.generateMemoryLayout(patterns);
 *  ```
 *
 *  # Spin Subindex
 *  If IDX_SPIN is present in a Subindex, it will be preserved just like
 *  IDX_SUM_ALL is preserved when generating a mameory layout.
 *
 *  # Example
 *  \snippet PropertyExtractor/IndexTreeGenerator.cpp IndexTreeGenerator
 *  ##Output
 *  \snippet output/PropertyExtractor/IndexTreeGenerator.txt IndexTreeGenerator */
class IndexTreeGenerator{
public:
	/** Constructs a PropertyExtractor::PropertyExtractor.
	 *
	 *  @param model The Model that the generated IndexTree is compatible
	 *  with. */
	IndexTreeGenerator(const Model &model);

	/** Generate an IndexTree that contains all the @link Index
	 *  Indices@endlink in the Model that satisfies one of the given
	 *  patterns.
	 *
	 *  @param patterns Index patters to match against.
	 *
	 *  @return An IndexTree containing all the @link Index Indices@endlink
	 *  in the Model that satisfies the given patterns. */
	IndexTree generateAllIndices(const std::vector<Index> &patterns) const;

	/** Generate an IndexTree that contains all the @link Index
	 *  Indices@endlink in the Model that satisfies one of the given
	 *  patterns. @link Subindex Subindices@endlink marked with IDX_SUM_ALL
	 *  will keep this flag in the coresponding position.
	 *
	 *  @param patterns Index patters to match against.
	 *
	 *  @return An IndexTree containing all the @link Index Indices@endlink
	 *  in the Model that satisfies the given patterns. */
	IndexTree generateMemoryLayout(const std::vector<Index> &patterns) const;
private:
	/** Model. */
	const Model &model;

	/** Generate an IndexTree containing the @link Index Indices@endlink
	 *  that satisfies on of the patterns in the given list of patterns.
	 *
	 *  @param patterns List of patterns to match.
	 *  @param keepSummationWildcard Flag indicating whether or not to keep
	 *  summation wildcards.
	 *
	 *  @return An IndexTree containing all indices in the Model that
	 *  satisifies any of the given patterns. */
	IndexTree generate(
		const std::vector<Index> &patterns,
		bool keepSummationWildcard
	) const;
};

inline IndexTree IndexTreeGenerator::generateAllIndices(
	const std::vector<Index> &patterns
) const{
	return generate(patterns, false);
}

inline IndexTree IndexTreeGenerator::generateMemoryLayout(
	const std::vector<Index> &patterns
) const{
	return generate(patterns, true);
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK

#endif
