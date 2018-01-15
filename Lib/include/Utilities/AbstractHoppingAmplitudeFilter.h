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

/** @package TBTKcalc
 *  @file AbstractHoppingAmplitudeFilter.h
 *  @brief Abstract HoppingAmplitude filter.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ABSTRACT_HOPPING_AMPLITUDE_FILTER
#define COM_DAFER45_TBTK_ABSTRACT_HOPPING_AMPLITUDE_FILTER

#include "HoppingAmplitude.h"

namespace TBTK{

class AbstractHoppingAmplitudeFilter{
public:
	/** Constructor. */
	AbstractHoppingAmplitudeFilter();

	/** Destructor. */
	virtual ~AbstractHoppingAmplitudeFilter();

	/** Clone. */
	virtual AbstractHoppingAmplitudeFilter* clone() const = 0;

	/** Returns true if the filter includes the HoppingAmplitude. */
	virtual bool isIncluded(
		const HoppingAmplitude &hoppingAmplitude
	) const = 0;
private:
};

inline AbstractHoppingAmplitudeFilter::AbstractHoppingAmplitudeFilter(){
}

inline AbstractHoppingAmplitudeFilter::~AbstractHoppingAmplitudeFilter(){
}

}; //End of namesapce TBTK

#endif
