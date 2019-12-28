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

/** @package TBTKcalc
 *  @file Argument.h
 *  @brief Argument to matplotlib.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_ARGUMENT
#define COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_ARGUMENT

#include <string>
#include <map>

namespace TBTK{
namespace Visualization{
namespace MatPlotLib{

/** @brief Argument to matplotlib.
 *
 *  The Argument is a helper class to the Plotter. It allows both strings and
 *  key-value pairs to be passed as arguments to the same function.
 *
 *  For example, consider the following function.
 *  ```cpp
 *    void f(const Argument &argument)
 *  ```
 *  This can be called using either
 *  ```cpp
 *    f("b--");
 *  ```
 *  or
 *  ```cpp
 *    f({
 *      {"linewidth", "2"},
 *      {"color", "blue"},
 *      {"linestyle, "dashed"}
 *    });
 *  ```
 *
 *  # Example
 *  \snippet Visualization/MatPlotLib/Argument.cpp Argument
 *  ## Output
 *  \snippet output/Visualization/MatPlotLib/Argument.txt Argument */
class Argument{
public:
	/** Constructs an uninitialized Argument. */
	Argument();

	/** Constructs an Argument from a zero terminated char*.
	 *
	 *  @param argument String representation fo the argument. */
	Argument(const char *argument);

	/** Constructs an Argument from a string.
	 *
	 *  @param argument String representation fo the argument. */
	Argument(const std::string &argument);

	/** Constructs an Argument from a list of key-value pairs. */
	Argument(
		const std::initializer_list<std::pair<std::string, std::string>> &argument
	);

	/** Get the argument string. */
	const std::string& getArgumentString() const;

	/** Get the argument map. */
	const std::map<std::string, std::string>& getArgumentMap() const;
private:
	std::string argumentString;
	std::map<std::string, std::string> argumentMap;
};

inline Argument::Argument(){
}

inline Argument::Argument(const char *argument) : argumentString(argument){
}

inline Argument::Argument(const std::string &argument){
	argumentString = argument;
}

inline Argument::Argument(
	const std::initializer_list<std::pair<std::string, std::string>> &argument
){
	for(auto element : argument)
		argumentMap[element.first] = element.second;
}

inline const std::string& Argument::getArgumentString() const{
	return argumentString;
}

inline const std::map<std::string, std::string>& Argument::getArgumentMap(
) const{
	return argumentMap;
}

};	//End namespace MatPlotLib
};	//End namespace Visualization
};	//End namespace TBTK

#endif
