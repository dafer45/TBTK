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

/** @file SelfEnergy.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{
namespace Property{

SelfEnergy::SelfEnergy() : AbstractProperty(){
}

SelfEnergy::SelfEnergy(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	unsigned int resolution
) :
	AbstractProperty(indexTree, resolution)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

SelfEnergy::SelfEnergy(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	const complex<double> *data
) :
	AbstractProperty(indexTree, resolution, data)
{
	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->resolution = resolution;
}

SelfEnergy::SelfEnergy(
	const SelfEnergy &selfEnergy
) :
	AbstractProperty(selfEnergy)
{
	lowerBound = selfEnergy.lowerBound;
	upperBound = selfEnergy.upperBound;
	resolution = selfEnergy.resolution;
}

SelfEnergy::SelfEnergy(
	SelfEnergy &&selfEnergy
) :
	AbstractProperty(std::move(selfEnergy))
{
	lowerBound = selfEnergy.lowerBound;
	upperBound = selfEnergy.upperBound;
	resolution = selfEnergy.resolution;
}

SelfEnergy::~SelfEnergy(){
}

};	//End of namespace Property
};	//End of namespace TBTK
