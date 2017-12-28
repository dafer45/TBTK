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
 *  @file Path.h
 *  @brief Path
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PATH
#define COM_DAFER45_TBTK_PATH

#include "Coordinate.h"
#include "Decoration.h"

#include <vector>

namespace TBTK{
namespace Plot{

class Path{
public:
	/** Constructor. */
	Path();

	/** Constructor. */
	Path(const std::vector<Coordinate> &coordinates, Decoration &decoration);

	/** Destructor. */
	virtual ~Path();

	/** Add coordinate. */
	void add(const Coordinate &coordinate);

	/** Set decoration. */
	void setDecoration(const Decoration &decoration);

	/** Draw. */
	virtual void draw();
private:
	/** Coordinates. */
	std::vector<Coordinate> coordinates;

	/** Decoration. */
	Decoration decoration;
};

inline Path::Path(){
}

inline Path::Path(
	const std::vector<Coordinate> &coordinates,
	Decoration &decoration
){
	this->coordinates = coordinates;
	this->decoration = decoration;
}

inline Path::~Path(){
}

inline void Path::add(const Coordinate &coordinate){
	coordinates.push_back(coordinate);
}

inline void Path::setDecoration(const Decoration &decoration){
	this->decoration = decoration;
}

};	//End namespace Plot
};	//End namespace TBTK

#endif
