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
 *  @file BackwardDifference.h
 *  @brief HoppingAmplitudeList corresponding to a backward difference.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BACKWARD_DIFFERENCE
#define COM_DAFER45_TBTK_BACKWARD_DIFFERENCE

#include "TBTK/HoppingAmplitudeList.h"

namespace TBTK{

/** @brief HoppingAmplitudeList corresponding to a backward difference. */
class BackwardDifference : public HoppingAmplitudeList{
public:
	/** Constructs a BackwardDifference.
	 *
	 *  @param subindex Subindex corresponding to the direction of
	 *  differentiation.
	 *
	 *  @param index Index at which the difference is taken.
	 *  @pragma dx Step length. */
	BackwardDifference(
		unsigned int subindex,
		const Index &index,
		double dx = 1.
	);
};

inline BackwardDifference::BackwardDifference(
	unsigned int subindex,
	const Index &index,
	double dx
){
	TBTKAssert(
		subindex < index.getSize(),
		"BackwardDifference::BackwardDifference()",
		"Invalid subindex. The subindex '" << subindex << "' is larger"
		<< " than the size of the Index '" << index.getSize() << "'.",
		""
	);
	TBTKAssert(
		index[subindex] > 0,
		"BackwardDifference::BackwardDifference()",
		"Invalid subindex value. Unable to add a backward difference"
		" for subindex '" << subindex << "' at '" << index.toString()
		<< "' since the backward difference would contain a negative"
		<< " subindex.",
		"Modify the domain of the differential equation to ensure that"
		<< " no negative subindices are needed."
	);

	Index backward = index;
	backward[subindex]--;
	add(HoppingAmplitude(1/dx, index, backward));
	add(HoppingAmplitude(-1/dx, index, index));
}

};	//End of namespace TBTK

#endif
