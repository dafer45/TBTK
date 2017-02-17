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

/** @file Model.cpp
 *
 *  @author Kristofer Björnson
 */

#include "Geometry.h"
#include "Model.h"
#include "Streams.h"
#include "TBTKMacros.h"

#include <fstream>
#include <math.h>
#include <string>

using namespace std;

namespace TBTK{

Model::Model(){
	temperature = 0.;
	chemicalPotential = 0.;
	isTalkative = true;
}

Model::~Model(){
}

void Model::construct(){
	if(isTalkative)
		Streams::out << "Constructing system\n";

	singleParticleContext.construct();

	int basisSize = getBasisSize();

	if(isTalkative)
		Streams::out << "\tBasis size: " << basisSize << "\n";
}

};	//End of namespace TBTK
