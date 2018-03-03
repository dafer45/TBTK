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

/** @file OrthorhombicBaseCentered.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Lattice/D3/OrthorhombicBaseCentered.h"
#include "TBTK/Streams.h"
#include "TBTK/Vector3d.h"

#include <cmath>

using namespace std;

namespace TBTK{
namespace Lattice{
namespace D3{

OrthorhombicBaseCentered::OrthorhombicBaseCentered(
	double side0Length,
	double side1Length,
	double side2Length
) :
	OrthorhombicPrimitive(
		side0Length,
		side1Length,
		side2Length
	)
{
	const vector<vector<double>> &latticeVectors = getLatticeVectors();

	Vector3d v0(latticeVectors.at(0));
	Vector3d v1(latticeVectors.at(1));

	vector<vector<double>> additionalSites;
	additionalSites.push_back(((v0 + v1)/2.).getStdVector());

	setAdditionalSites(additionalSites);
}

OrthorhombicBaseCentered::~OrthorhombicBaseCentered(){
}

void OrthorhombicBaseCentered::makePrimitive(){
	const vector<vector<double>> &additionalSites = getAdditionalSites();
	const vector<vector<double>> &latticeVectors = getLatticeVectors();

	Vector3d v0(latticeVectors.at(0));

	Vector3d newV0(additionalSites.at(0));
	Vector3d newV1 = newV0 - v0;

	vector<vector<double>> newLatticeVectors;
	newLatticeVectors.push_back(newV0.getStdVector());
	newLatticeVectors.push_back(newV1.getStdVector());
	newLatticeVectors.push_back(latticeVectors.at(2));

	setLatticeVectors(newLatticeVectors);
	setAdditionalSites(vector<vector<double>>());
}

};	//End of namespace D3
};	//End of namespace Lattice
};	//End of namespace TBTK
