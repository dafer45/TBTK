/* Copyright 2018 Kristofer Björnson
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
 *  @file CeteredDifference.h
 *  @brief HoppingAmplitudeList corresponding to a centered difference.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CENTERED_DIFFERENCE
#define COM_DAFER45_TBTK_CENTERD_DIFFERENCE

#include "TBTK/HoppingAmplitudeList.h"

namespace TBTK{

/** @brief HoppingAmplitudeList corresponding to a centered difference. */
class CenteredDifference : public HoppingAmplitudeList{
public:
	/** Constructs a CenteredDifference.
	 *
	 *  @param subindex Subindex corresponding to the direction of
	 *  differentiation.
	 *
	 *  @param index Index at which the difference is taken.
	 *  @param dx Step length. */
	CenteredDifference(
		unsigned int subindex,
		const Index &index,
		double dx = 1
	);
};

inline CenteredDifference::CenteredDifference(
	unsigned int subindex,
	const Index &index,
	double dx
){
	TBTKAssert(
		subindex < index.getSize(),
		"CenteredDifference::CenteredDifference()",
		"Invalid subindex. The subindex '" << subindex << "' is larger"
		<< " than the size of the Index '" << index.getSize() << "'.",
		""
	);
	TBTKAssert(
		index[subindex] > 0,
		"CenteredDifference::CenteredDifference()",
		"Invalid subindex value. Unable to add a centered difference"
		" for subindex '" << subindex << "' at '" << index.toString()
		<< "' since the centered difference would contain a negative"
		<< " subindex.",
		"Modify the domain of the differential equation to ensure that"
		<< " no negative subindices are needed."
	);

	Index backward = index;
	backward[subindex]--;
	Index forward = index;
	forward[subindex]++;
	pushBack(HoppingAmplitude(1/(2*dx), index, backward));
	pushBack(HoppingAmplitude(-1/(2*dx), index, forward));
}

};	//End of namespace TBTK

#endif
