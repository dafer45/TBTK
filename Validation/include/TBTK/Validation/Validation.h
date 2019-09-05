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

/** @file Test.h
 *  @brief Helper function for writing validators.
 *
 *  @author Kristofer Björnson
 */

#include <iostream>
#include <sstream>

using namespace TBTK;

enum Mode {NumTests, Get, Validate};

inline void init(int argc, char **argv, Mode &mode, int &id, std::string testName){
	TBTKAssert(
		argc > 1,
		testName,
		testName << " must be called with exactly one parameter.",
		""
	);

	if(std::string(argv[1]).compare("NumTests") == 0){
		mode = NumTests;
	}
	else if(std::string(argv[1]).compare("Get") == 0){
		mode = Get;
	}
	else if(std::string(argv[1]).compare("Validate") == 0){
		mode = Validate;
	}
	else{
		std::cout << "Invalid argument '" << argv[1] << "'.";

		exit(1);
	}

	switch(mode){
	case NumTests:
		break;
	case Get:
	case Validate:
		TBTKAssert(
			argc == 3,
			testName,
			"Expected two parameters, but found '" << argc << "'.",
			""
		);

		id = atoi(argv[2]);

		break;
	}
}

template<typename Type>
void execute(Type &reference, Mode mode, int id, std::string testName){
	switch(mode){
	case NumTests:
		std::cout << "execute was incorrectly called with the mode NumTests.\n";
		break;
	case Get:
		std::cout << reference.serialize(Serializable::Mode::JSON);
		break;
	case Validate:
		std::stringstream ss;
		ss << std::cin.rdbuf();
		std::string input = ss.str();

		Type inputObject(input, Serializable::Mode::JSON);

		std::string serialization0 = reference.serialize(Serializable::Mode::JSON);
		std::string serialization1 = inputObject.serialize(Serializable::Mode::JSON);

		if(serialization0.compare(serialization1) == 0)
			std::cout << "Pass [" << testName << " " << id << "]\n";
		else
			std::cout << "Fail [" << testName << " " << id << "]\n";

		break;
	}
}
