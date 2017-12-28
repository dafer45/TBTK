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
 *  @file Drawable.h
 *  @brief Drawable
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DRAWABLE
#define COM_DAFER45_TBTK_DRAWABLE

#include <opencv2/core/core.hpp>

namespace TBTK{
namespace Plot{

class Plotter;

class Drawable{
public:
	/** Constructor. */
	Drawable();

	/** Destructor. */
	virtual ~Drawable();

	/** Draw. */
	virtual void draw(
		cv::Mat &canvas,
		const Plotter &plotter,
		double minX,
		double maxX,
		double minY,
		double maxY
	) = 0;

	/** Get minimum x-coordiante. */
	virtual double getMinX() const = 0;

	/** Get maximum x-coordiante. */
	virtual double getMaxX() const = 0;

	/** Get minimum y-coordiante. */
	virtual double getMinY() const = 0;

	/** Get maximum y-coordiante. */
	virtual double getMaxY() const = 0;
private:
};

inline Drawable::Drawable(){
}

inline Drawable::~Drawable(){
}

};	//End namespace Plot
};	//End namespace TBTK

#endif
