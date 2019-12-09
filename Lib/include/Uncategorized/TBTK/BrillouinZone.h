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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file BrillouinZone.h
 *  @brief Brillouin zone.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_WIGNER_BRILLOUIN_ZONE
#define COM_DAFER45_TBTK_WIGNER_BRILLOUIN_ZONE

#include "TBTK/WignerSeitzCell.h"

#include <vector>

namespace TBTK{

/** Brillouin zone. */
class BrillouinZone : public WignerSeitzCell{
public:
	/** Constructor. */
	BrillouinZone(
		const std::vector<std::vector<double>> &basisVectors,
		MeshType meshType
	);

	/** Destructor. */
	virtual ~BrillouinZone();
};

};	//End namespace TBTK

#endif
/// @endcond
