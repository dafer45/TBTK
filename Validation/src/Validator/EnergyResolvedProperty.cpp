/* Copyright 2019 Kristofer Björnson
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

/** @file EnergyResolvedProperty.cpp
 *  @brief Generates and validates @link EnergyResolvedProperty
 *  EnergyResolvedProperties@endlink.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Property/EnergyResolvedProperty.h"
#include "TBTK/Validation/Validation.h"

#include <iostream>

using namespace std;
using namespace TBTK;

int NUM_TESTS = 4;

int main(int argc, char **argv){
	Mode mode;
	int id;
	init(argc, argv, mode, id, "EnergyResolvedProperty");

	if(mode == NumTests){
		cout << NUM_TESTS;

		return 0;
	}

	Property::EnergyResolvedProperty<int> energyResolvedProperty;
	switch(id){
	case 0:
	{
		CArray<int> data(10);
		for(unsigned int n = 0; n < 10; n++)
			data[n] = n;

		energyResolvedProperty = Property::EnergyResolvedProperty<int>(
			Range(-5, 5, 10),
			data
		);
		energyResolvedProperty.setDefaultValue(137);

		break;
	}
	case 1:
	{
		CArray<int> data(30);
		for(unsigned int n = 0; n < 30; n++)
			data[n] = n;

		IndexTree indexTree;
		indexTree.add({1, 2, 3});
		indexTree.add({1, 2, 4});
		indexTree.add({2, 3});
		indexTree.generateLinearMap();

		energyResolvedProperty = Property::EnergyResolvedProperty<int>(
			indexTree,
			Range(-5, 5, 10),
			data
		);
		energyResolvedProperty.setDefaultValue(137);

		break;
	}
	case 2:
	{
		CArray<int> data(18);
		for(unsigned int n = 0; n < 18; n++)
			data[n] = n;

		IndexTree indexTree;
		indexTree.add({1, 2, 3});
		indexTree.add({1, 2, 4});
		indexTree.add({2, 3});
		indexTree.generateLinearMap();

		energyResolvedProperty = Property::EnergyResolvedProperty<int>(
			Property::EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
			indexTree,
			-5,
			5,
			10,
			data
		);
		energyResolvedProperty.setDefaultValue(137);

		break;
	}
	case 3:
	{
		CArray<int> data(21);
		for(unsigned int n = 0; n < 21; n++)
			data[n] = n;

		IndexTree indexTree;
		indexTree.add({1, 2, 3});
		indexTree.add({1, 2, 4});
		indexTree.add({2, 3});
		indexTree.generateLinearMap();

		energyResolvedProperty = Property::EnergyResolvedProperty<int>(
			Property::EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
			indexTree,
			-6,
			6,
			10,
			data
		);
		energyResolvedProperty.setDefaultValue(137);

		break;
	}
	default:
		TBTKExit(
			"EnergyResolvedProperty",
			"Unknown test id '" << id << "'.",
			""
		);
	}

	execute<Property::EnergyResolvedProperty<int>>(
		energyResolvedProperty,
		mode,
		id,
		"EnergyResolvedProperty"
	);

	return 0;
}
