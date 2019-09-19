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

/** @package TBTKtemp
 *  @file main.cpp
 *  @brief New project
 *
 *  Empty template project.
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <complex>

using namespace std;
using namespace TBTK;

int main(int argc, char **argv){
	if(argc == 1){
		Streams::out << "==================================== ABOUT =====================================\n";
		Streams::out << TBTK_ABOUT_STRING << "\n";
		Streams::out << "\n";
		Streams::out << "============================= FURTHER INFORMATION ==============================\n";
		Streams::out << "For help, type 'TBTK help'.\n";
		Streams::out << "\n";
	}
	else if(string(argv[1]).compare("help") == 0){
		Streams::out << "================================== TBTK HELP ===================================\n";
		Streams::out << "This help is a brief introduction that is aimed at helping new developers\n";
		Streams::out << "getting started with TBTK. More detailed information is found in the\n";
		Streams::out << "documentation and manual pages.\n";
		Streams::out << "\n";
		Streams::out << "================================ DOCUMENTATION =================================\n";
		Streams::out << "The documentation consists of four parts: The instalation instructions, manual,\n";
		Streams::out << "tutorials, and the API. It can be built by executing 'make documentation' in the\n";
		Streams::out << "build folder and then accessed at '/path/to/build/folder/doc/html/index.html'.\n";
		Streams::out << "Alternatively, the manual can be found at http://www.second-quantization.com.\n";
		Streams::out << "\n";
		Streams::out << "==================================== MANUAL ====================================\n";
		Streams::out << "A coherent presentation of TBTK is provided in the manual. This is the place to\n";
		Streams::out << "start if you want to understand the underlying design philosophy of TBTK. The\n";
		Streams::out << "manual is not the quickest way to get started, but reading it is highly\n";
		Streams::out << "recommended in order to make efficient use of TBTK.\n";
		Streams::out << "\n";
		Streams::out << "See 'DOCUMENTATION' for instructions on how to access the manual.\n";
		Streams::out << "\n";
		Streams::out << "================================== TUTORIALS ===================================\n";
		Streams::out << "The quickest way to get started building applications is to follow the\n";
		Streams::out << "tutorials.\n";
		Streams::out << "\n";
		Streams::out << "See 'DOCUMENTATION' for instructions on how to access the tutorials.\n";
		Streams::out << "\n";
		Streams::out << "===================================== API ======================================\n";
		Streams::out << "Detailed information about the C++ library that forms the core of TBTK is found\n";
		Streams::out << "in the application programming interface (API). The API contains information\n";
		Streams::out << "about all of the classes and functions contained in the TBTK library and is the\n";
		Streams::out << "main reference when implementing new applications.\n";
		Streams::out << "\n";
		Streams::out << "See 'DOCUMENTATION' for instructions on how to access the API.\n";
		Streams::out << "\n";
//		Streams::out << "========================================|========================================\n";
//		Streams::out << "\n";
		Streams::out << "================================= MANUAL PAGES =================================\n";
		Streams::out << "In addition to an extensive C++ library for writing applications, TBTK consists\n";
		Streams::out << "of a set of command line tools that aids development. Detailed information about\n";
		Streams::out << "these tools are provided in their respective manual pages. To access the manual\n";
		Streams::out << "page for a given tool, type 'man TBTKTool', where 'TBTKTool' is to be replaced\n";
		Streams::out << "by the name of the corresponding tool.\n";
		Streams::out << "\n";
		Streams::out << "============================ CREATE NEW APPLICATION ============================\n";
		Streams::out << "A new application is created using TBTKCreateApplication. To create, build and\n";
		Streams::out << "run your first application, type:\n";
//		Streams::out << "\n";
		Streams::out << "\tTBTKCreateApplication ApplicationName\n";
		Streams::out << "\tcd ApplicationName\n";
		Streams::out << "\tcmake .\n";
		Streams::out << "\tmake\n";
		Streams::out << "\t./build/Application\n";
		Streams::out << "\n";
		Streams::out << "For more detailed usage information, see the manual page for\n";
		Streams::out << "TBTKCreateApplication.\n";
		Streams::out << "\n";
	}
	else if(string(argv[1]).compare("resource-path") == 0){
		Streams::out << TBTK_RESOURCE_PATH;
	}
	else{
		Streams::out << "Unknown option '" << argv[1] << "'.\n";
	}

	return 0;
}
