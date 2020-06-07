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

/** @file Solver.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Solver/Solver.h"

using namespace std;

namespace TBTK{
namespace Solver{

DynamicTypeInformation Solver::dynamicTypeInformation(
	"Solver::Solver",
	{}
);

Solver::Solver(){
	model = NULL;
}

Solver::~Solver(){
}

std::string Solver::serialize(Mode mode) const{
	//Need to serialize the Model before this is done. If the Model is
	//added to the Context instead of passed into the Solver directly, it
	//can be serialized in the Context and the name be added as the
	//serialization in the Solver.
	TBTKNotYetImplemented("Solver::Solver::serialize()");

	switch(mode){
	case Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "Solver::Solver";
		//Serialize the Model here. Se comment above and
		//DevelopmentsNotes/Ideas.
		//...
		j["persistentObject"] = PersistentObject::serialize(mode);
		return j.dump();
	}
	default:
		TBTKExit(
			"Solver::Solver::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK
