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
 *  @file InteractionAmplitudeSet.h
 *  @brief InteractionAmplitudeSet.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INTERACTION_AMPLITUDE_SET
#define COM_DAFER45_TBTK_INTERACTION_AMPLITUDE_SET

#include "TBTK/HoppingAmplitudeSet.h"
#include "TBTK/InteractionAmplitude.h"

#include <complex>

namespace TBTK{

class InteractionAmplitudeSet{
public:
	/** Constructor. */
//	InteractionAmplitudeSet(const HoppingAmplitudeSet *hoppingAmplitudeSet);

	/** Destructor. */
//	~InteractionAmplitudeSet();

	/** Add interaction. */
	void addIA(InteractionAmplitude ia);

	/** Returns the number of interaction amplitudes. */
	unsigned int getNumInteractionAmplitudes() const;

	/** Returns an interaction amplitude. */
	const InteractionAmplitude& getInteractionAmplitude(
		unsigned int n
	) const;
private:
	/** Single-particle HoppingAmplitudeSet. */
//	const HoppingAmplitudeSet *hoppingAmplitudeSet;

	/** Interaction amplitudes. */
	std::vector<InteractionAmplitude> interactionAmplitudes;
};

inline void InteractionAmplitudeSet::addIA(InteractionAmplitude ia){
	interactionAmplitudes.push_back(ia);
}

inline unsigned int InteractionAmplitudeSet::getNumInteractionAmplitudes() const{
	return interactionAmplitudes.size();
}

inline const InteractionAmplitude& InteractionAmplitudeSet::getInteractionAmplitude(
	unsigned int n
) const{
	return interactionAmplitudes.at(n);
}

};	//End of namespace TBTK

#endif
/// @endcond
