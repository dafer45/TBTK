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

#ifndef COM_DAFER45_TBTK_DOCUMENTATION_EXAMPLES_HEADER_AND_FOOTER
#define COM_DAFER45_TBTK_DOCUMENTATION_EXAMPLES_HEADER_AND_FOOTER

#include <iostream>
#include <string>

namespace TBTK{
namespace DocumentationExamples{

/** @brief Adds a header and footer to the output.
 *
 *  This class is meant to be instantiated as a global variable to print the
 *  given string before and after the output written to stdout. This allows for
 *  formating string required by Doxygen to be written without the code for
 *  doing so being included in the example. The intended usage is as follows:
 *
 *  #include "HeaderAndFooter.h"
 *  TBTK::DocumentationExamples::HeaderAndFooter("//! [ExampleName]")
 *
 *  //! [ExampleName]
 *  #include "TBTK/Index.h"
 *  #include "TBTK/Streams.h"
 *
 *  using namespace TBTK;
 *
 *  int main(){
 *    Index index({1, 2, 3});
 *    for(unsigned int n = 0; n < index.getSize(); n++)
 *      Streams::out << index[n] << "\n";
 *  }
 *  //! [ExampleName]
 *
 *  With this structure, doxygen will be able to extract the code snippet
 *  between the two comments. At the same time, the output will have the form
 *  //! [ExampleName]
 *  0
 *  1
 *  2
 *  //! [ExampleName]
 *  where the first and last lines have been added by HeaderAndFooter. This
 *  way, doxygen is able to extract both the output and code used to generate
 *  the output in the same way. */
class HeaderAndFooter{
public:
	HeaderAndFooter(const std::string &value){
		this->value = value;
		std::cout << "//! [" << value << "]\n";
	}
	~HeaderAndFooter(){
		std::cout << "//! [" << value << "]\n";
	}
private:
	std::string value;
};

};	//End of namespace DocumentationExamples
};	//End of namespace TBTK

#endif
