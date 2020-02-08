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
 *  @file LDOS.h
 *  @brief Property container for spectral function.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SPECTRAL_FUNCTION
#define COM_DAFER45_TBTK_SPECTRAL_FUNCTION

#include "TBTK/Property/EnergyResolvedProperty.h"

#include <complex>

namespace TBTK{
namespace Property{

/** @brief Property container for spectral function. */
class SpectralFunction : public EnergyResolvedProperty<std::complex<double>>{
public:
	/** Constructor. */
	SpectralFunction();

	/** Constructor. */
	SpectralFunction(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		int resolution
	);

	/** Constructor. */
	SpectralFunction(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		int resolution,
		const std::complex<double> *data
	);

	/** Overrider LDOS::serialize(). */
	virtual std::string serialize(Mode mode) const;
private:
};

};	//End namespace Property
};	//End namespace TBTK

#endif
/// @endcond
