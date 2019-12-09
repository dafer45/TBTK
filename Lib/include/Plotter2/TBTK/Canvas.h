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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file Canvas.h
 *  @brief Canvas
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CANVAS
#define COM_DAFER45_TBTK_CANVAS

#include "TBTK/TBTKMacros.h"

#include <string>

namespace TBTK{

class Canvas{
public:
	/** Constructor. */
	Canvas();

	/** Destructor. */
	virtual ~Canvas();

	/** Set width. */
	void setWidth(unsigned int width);

	/** Get width. */
	double getWidth() const;

	/** Set height. */
	void setHeight(unsigned int height);

	/** Get height. */
	double getHeight() const;

	/** Set the title. */
	void setTitle(const std::string &title);

	/** Get the title. */
	const std::string& getTitle() const;

	/** Clear plot. */
	virtual void clear();
private:
	/** Size of the resulting image. */
	double width, height;

	/** Title. */
	std::string title;
};

inline void Canvas::setWidth(unsigned int width){
	this->width = width;
}

inline double Canvas::getWidth() const{
	return width;
}

inline void Canvas::setHeight(unsigned int height){
	this->height = height;
}

inline double Canvas::getHeight() const{
	return height;
}

inline void Canvas::setTitle(const std::string &title){
	this->title = title;
}

inline const std::string& Canvas::getTitle() const{
	return title;
}

inline void Canvas::clear(){
}

};	//End namespace TBTK

#endif
/// @endcond
