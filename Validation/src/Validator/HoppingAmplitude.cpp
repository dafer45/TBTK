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

/** @file HoppingAmplitude.cpp
 *  @brief Generates and validates @link HoppingAMplitude
 *  HoppingAmplitudes@endlink.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/HoppingAmplitude.h"
#include "TBTK/Validation/Validation.h"

#include <iostream>

using namespace std;
using namespace TBTK;

int NUM_TESTS = 1;

int main(int argc, char **argv){
	Mode mode;
	int id;
	init(argc, argv, mode, id, "HoppingAmplitude");

	if(mode == NumTests){
		cout << NUM_TESTS;

		return 0;
	}

	HoppingAmplitude hoppingAmplitude;
	switch(id){
	case 0:
		hoppingAmplitude = HoppingAmplitude(137, {1, 2, 3}, {4, 5, 6});
		break;
	default:
		TBTKExit(
			"HoppingAmplitude",
			"Unknown test id '" << id << "'.",
			""
		);
	}

	execute<HoppingAmplitude>(hoppingAmplitude, mode, id, "HoppingAmplitude");

	return 0;
}
