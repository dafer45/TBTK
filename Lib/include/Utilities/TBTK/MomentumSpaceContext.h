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
 *  @file MomentumSpaceContext.h
 *  @brief Container of information about the momentum space.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MOMENTUM_SPACE_CONTEXT
#define COM_DAFER45_TBTK_MOMENTUM_SPACE_CONTEXT

#include "TBTK/BrillouinZone.h"

#include <vector>

namespace TBTK{

class MomentumSpaceContext{
public:
	/** Constructor. */
	MomentumSpaceContext(
		const BrillouinZone &brillouinZone,
		const std::vector<unsigned int> &numMeshPoints
	);

	/** Get Brillouin zone. */
	const BrillouinZone& getBrillouinZone() const;

	/** Get number of mesh points. */
	const std::vector<unsigned int>& getNumMeshPoints() const;

	/** Get mesh. */
	const std::vector<std::vector<double>>& getMesh() const;

	/** Get Index corresponding to given k-vector. */
	Index getKIndex(const std::vector<double> &k) const;
private:
	/** BrillouinZone. */
	const BrillouinZone *brillouinZone;

	/** Number of mesh points. */
	std::vector<unsigned int> numMeshPoints;

	/** Mesh. */
	std::vector<std::vector<double>> mesh;
};

inline const BrillouinZone& MomentumSpaceContext::getBrillouinZone() const{
	return *brillouinZone;
}

inline const std::vector<unsigned int>& MomentumSpaceContext::getNumMeshPoints(
) const{
	return numMeshPoints;
}

inline const std::vector<std::vector<double>>& MomentumSpaceContext::getMesh(
) const{
	return mesh;
}

inline Index MomentumSpaceContext::getKIndex(
	const std::vector<double> &k
) const{
	return brillouinZone->getMinorCellIndex(
		k,
		numMeshPoints
	);
}

};	//End of namespace TBTK

#endif
/// @endcond
