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

/** @file TriclinicPrimitive.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/TBTKMacros.h"
#include "TBTK/Lattice/D3/TriclinicPrimitive.h"
#include "TBTK/Vector3d.h"

#include <cmath>

using namespace std;

namespace TBTK{
namespace Lattice{
namespace D3{

TriclinicPrimitive::TriclinicPrimitive(
	double side0Length,
	double side1Length,
	double side2Length,
	double angle01,
	double angle02,
	double angle12
){
	vector<vector<double>> latticeVectors;

	latticeVectors.push_back(vector<double>());
	latticeVectors.at(0).push_back(side0Length);
	latticeVectors.at(0).push_back(0.);
	latticeVectors.at(0).push_back(0.);

	latticeVectors.push_back(vector<double>());
	latticeVectors.at(1).push_back(side1Length*cos(angle01));
	latticeVectors.at(1).push_back(side1Length*sin(angle01));
	latticeVectors.at(1).push_back(0.);

	Vector3d comp1 = Vector3d(latticeVectors.at(0)).unit()*cos(angle02);
	Vector3d comp2 = Vector3d(latticeVectors.at(1)).unit()*cos(angle12);
	TBTKAssert(
		(comp1 + comp2).norm() < 1,
		"TriclinicPrimitive::TriclinicPrimitive()",
		"Incompatible lattice angles. It is impossible to simultaneously satisfy the given angles (angle01=" << angle01 << ", angle02=" << angle02 << ", angle12=" << angle12 << ").",
		""
	);
	Vector3d comp3 = side2Length*Vector3d(latticeVectors.at(0)).unit()*Vector3d(latticeVectors.at(1)).unit()*sqrt(1 - pow((comp1+comp2).norm(), 2));

	latticeVectors.push_back((comp1+comp2+comp3).getStdVector());

	setLatticeVectors(latticeVectors);
}

TriclinicPrimitive::~TriclinicPrimitive(){
}

};	//End of namespace D3
};	//End of namespace Lattice
};	//End of namespace TBTK
