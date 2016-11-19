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

/** @file OrthorhombicPrimitive.cpp
 *
 *  @author Kristofer Björnson
 */

#include "OrthorhombicCentered.h"

#include <cmath>

using namespace std;

namespace TBTK{
namespace Lattice{
namespace D2{

OrthorhombicCentered::OrthorhombicCentered(
	double side1Length,
	double side2Length
) :
	OrthorhombicPrimitive(side1Length, side2Length)
{
	vector<vector<double>> additionalSites;
	additionalSites.push_back(vector<double>());
	additionalSites.at(0).push_back(side1Length/2.);
	additionalSites.at(0).push_back(side2Length/2.);

	setAdditionalSites(additionalSites);
}

OrthorhombicCentered::~OrthorhombicCentered(){
}

};	//End of namespace D2
};	//End of namespace Lattice
};	//End of namespace TBTK
