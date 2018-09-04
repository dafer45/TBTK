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

/** @file SpectralFunction.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/SpectralFunction.h"

#include <utility>

using namespace std;

namespace TBTK{
namespace Property{

SpectralFunction::SpectralFunction(
	const std::vector<int> &ranges,
	double lowerBound,
	double upperBound,
	int resolution
) :
	LDOS(ranges, lowerBound, upperBound, resolution)
{
}

SpectralFunction::SpectralFunction(
	const std::vector<int> &ranges,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) :
	LDOS(ranges, lowerBound, upperBound, resolution, data)
{
}

SpectralFunction::SpectralFunction(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution
) : LDOS(indexTree, lowerBound, upperBound, resolution){
}

SpectralFunction::SpectralFunction(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	int resolution,
	const double *data
) : LDOS(indexTree, lowerBound, upperBound, resolution, data){
}

SpectralFunction::SpectralFunction(
	const SpectralFunction &spectralFunction
) :
	LDOS(spectralFunction)
{
}

SpectralFunction::SpectralFunction(
	SpectralFunction &&spectralFunction
) :
	LDOS(std::move(spectralFunction))
{
}

SpectralFunction::~SpectralFunction(){
}

SpectralFunction& SpectralFunction::operator=(const SpectralFunction &rhs){
	if(this != &rhs)
		LDOS::operator=(rhs);

	return *this;
}

SpectralFunction& SpectralFunction::operator=(SpectralFunction &&rhs){
	if(this != &rhs)
		LDOS::operator=(std::move(rhs));

	return *this;
}

string SpectralFunction::serialize(Mode mode) const{
	TBTKNotYetImplemented("SpectralFunction::serialize()");
}

};	//End of namespace Property
};	//End of namespace TBTK
