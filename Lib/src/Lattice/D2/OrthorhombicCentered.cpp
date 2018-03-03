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

#include "TBTK/Lattice/D2/OrthorhombicCentered.h"
#include "TBTK/Streams.h"

#include <cmath>

using namespace std;

namespace TBTK{
namespace Lattice{
namespace D2{

OrthorhombicCentered::OrthorhombicCentered(
	double side0Length,
	double side1Length
) :
	OrthorhombicPrimitive(side0Length, side1Length)
{
	vector<vector<double>> additionalSites;
	additionalSites.push_back(vector<double>());
	additionalSites.at(0).push_back(side0Length/2.);
	additionalSites.at(0).push_back(side1Length/2.);

	setAdditionalSites(additionalSites);
}

OrthorhombicCentered::~OrthorhombicCentered(){
}

void OrthorhombicCentered::makePrimitive(){
	const vector<vector<double>> &additionalSites = getAdditionalSites();
	const vector<vector<double>> &latticeVectors = getLatticeVectors();

	vector<vector<double>> newLatticeVectors;
	newLatticeVectors.push_back(vector<double>());
	newLatticeVectors.push_back(vector<double>());

	newLatticeVectors.at(0).push_back(additionalSites.at(0).at(0));
	newLatticeVectors.at(0).push_back(additionalSites.at(0).at(1));

	newLatticeVectors.at(1).push_back(additionalSites.at(0).at(0) - latticeVectors.at(0).at(0));
	newLatticeVectors.at(1).push_back(additionalSites.at(0).at(1) - latticeVectors.at(0).at(1));

	setLatticeVectors(newLatticeVectors);

	setAdditionalSites(vector<vector<double>>());
}

};	//End of namespace D2
};	//End of namespace Lattice
};	//End of namespace TBTK
