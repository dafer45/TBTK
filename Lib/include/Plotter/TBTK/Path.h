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

#include "TBTK/PlotCanvas.h"
#include "TBTK/Coordinate.h"
#include "TBTK/Decoration.h"
#include "TBTK/Drawable.h"

#include <vector>

#include <opencv2/core/core.hpp>

namespace TBTK{
namespace Plot{

class Path : public Drawable{
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
	virtual void draw(PlotCanvas &canvas);

	/** Implements Drawable::getMinX(). */
	virtual double getMinX() const;

	/** Implements Drawable::getMaxX(). */
	virtual double getMaxX() const;

	/** Implements Drawable::getMinY(). */
	virtual double getMinY() const;

	/** Implements Drawable::getMaxY(). */
	virtual double getMaxY() const;
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

inline double Path::getMinX() const{
	if(coordinates.size() == 0)
		return 0.;

	double min = coordinates[0].x;
	for(unsigned int n = 0; n < coordinates.size(); n++)
		if(coordinates[n].x < min)
			min = coordinates[n].x;

	return min;
}

inline double Path::getMaxX() const{
	if(coordinates.size() == 0)
		return 0.;

	double max = coordinates[0].x;
	for(unsigned int n = 0; n < coordinates.size(); n++)
		if(coordinates[n].x > max)
			max = coordinates[n].x;

	return max;
}

inline double Path::getMinY() const{
	if(coordinates.size() == 0)
		return 0.;

	double min = coordinates[0].y;
	for(unsigned int n = 0; n < coordinates.size(); n++)
		if(coordinates[n].y < min)
			min = coordinates[n].y;

	return min;
}

inline double Path::getMaxY() const{
	if(coordinates.size() == 0)
		return 0.;

	double max = coordinates[0].y;
	for(unsigned int n = 0; n < coordinates.size(); n++)
		if(coordinates[n].y > max)
			max = coordinates[n].y;

	return max;
}

};	//End namespace Plot
};	//End namespace TBTK

#endif
