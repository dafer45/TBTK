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

/** @package TBTKcalc
 *  @file Coordinate.h
 *  @brief Coordinate
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PLOT_COORDINATE
#define COM_DAFER45_TBTK_PLOT_COORDINATE

#include "TBTK/TBTKMacros.h"

#include <vector>

namespace TBTK{
namespace Plot{

class Coordinate{
public:
	/** Coordinates. */
	double x, y;

	/** Constructor. */
	Coordinate(double x, double y);

	/** Constructor. */
	Coordinate(const std::vector<double> &coordinates);

	/** Destructor. */
	~Coordinate();
private:
};

inline Coordinate::Coordinate(double x, double y){
	this->x = x;
	this->y = y;
}

inline Coordinate::Coordinate(const std::vector<double> &coordinates){
	TBTKAssert(
		coordinates.size() == 2,
		"Coordinate::Coordinate()",
		"Number of coordinates must be 2, but '" << coordinates.size()
		<< "' coordinates supplied.",
		""
	);

	this->x = coordinates[0];
	this->y = coordinates[1];
}

inline Coordinate::~Coordinate(){
}

};	//End namespace Plot
};	//End namespace TBTK

#endif
