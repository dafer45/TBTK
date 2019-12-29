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
 *  @file SquareLattice.h
 *  @brief Square lattice model.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MODELS_SQUARE_LATTICE
#define COM_DAFER45_TBTK_MODELS_SQUARE_LATTICE

#include "TBTK/Model.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Models{

/** @brief Square lattice model.
 *
 *  The Square lattice allows for nearest-neighbor or next nearest-neighbor
 *  Models on a square lattice to be set up easily. A Model with the side
 *  lengths SIZE_X and SIZE_Y is created as follows.
 *  ```cpp
 *    Model model = Models::SquareLattice({SIZE_X, SIZE_Y}, {U, t, tp});
 *  ```
 *  Here U is the on-site potential, t is the nearest neighbor hopping
 *  amplitude, and tp is an optional next-nearest neighbor hopping amplitude.
 *
 *  It is also possible to create a spinful model by adding a IDX_SPIN flag as
 *  follows.
 *  ```cpp
 *    Model model = Models::SquareLattice(
 *      {SIZE_X, SIZE_Y, IDX_SPIN},
 *      {U, t, tp}
 *    );
 *  ```
 *
 *  The resulting Model has not yet been *constructed*, which means that it is
 *  possible to add further @link HoppingAmplitude HoppingAmplitudes@endlink to
 *  it. Before using it, remember to call
 *  ```cpp
 *    model.construct();
 *  ``` */
class SquareLattice : public Model{
public:
	/** Constructs a square lattice.
	 *
	 *  @param size The size of the lattice. One Subindex can have the
	 *  value IDX_SPIN to indicate a spin Subindex. Must have two
	 *  components (plus one if IDX_SPIN is included).
	 *
	 *  @param parameters The Model parameters. Must have two components,
	 *  where the first component is the on-site term, and the second is
	 *  the nearest neighbor hopping amplitude. */
	SquareLattice(
		const Index &size,
		const std::vector<std::complex<double>> &parameters
	);
private:
	/** Create Model with spin-Subindex. */
	void createModelWithSpin(
		const Index &size,
		const std::vector<std::complex<double>> &parameters
	);

	/** Create Model without spin-Subindex. */
	void createModelWithoutSpin(
		const Index &size,
		const std::vector<std::complex<double>> &parameters
	);
};

};	//End of namespace Models
};	//End of namespace TBTK

#endif
