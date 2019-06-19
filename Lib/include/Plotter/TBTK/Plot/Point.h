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

#ifndef COM_DAFER45_TBTK_POINT
#define COM_DAFER45_TBTK_POINT

#include "TBTK/Plot/PlotCanvas.h"
#include "TBTK/Plot/Coordinate.h"
#include "TBTK/Plot/Decoration.h"
#include "TBTK/Plot/Drawable.h"

namespace TBTK{
namespace Plot{

class Point : public Drawable{
public:
	/** Constructor. */
	Point();

	/** Constructor. */
	Point(const Coordinate &coordinate, const Decoration &decoration);

	/** Destructor. */
	virtual ~Point();

	/** Set coordinate. */
	void setCoordinate(const Coordinate &coordinate);

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
	Coordinate coordinate;

	/** Decoration. */
	Decoration decoration;
};

inline Point::Point() : coordinate(0, 0){
}

inline Point::Point(
	const Coordinate &coordinate,
	const Decoration &decoration
) :
	coordinate(coordinate)
{
	this->decoration = decoration;
}

inline Point::~Point(){
}

inline void Point::setCoordinate(const Coordinate &coordinate){
	this->coordinate = coordinate;
}

inline void Point::setDecoration(const Decoration &decoration){
	this->decoration = decoration;
}

inline double Point::getMinX() const{
	return coordinate.x;
}

inline double Point::getMaxX() const{
	return coordinate.x;
}

inline double Point::getMinY() const{
	return coordinate.y;
}

inline double Point::getMaxY() const{
	return coordinate.y;
}

};	//End namespace Plot
};	//End namespace TBTK

#endif
