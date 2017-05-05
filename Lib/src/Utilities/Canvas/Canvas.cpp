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

/** @file Canvas.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../../../include/Utilities/Canvas/Canvas.h"

using namespace cv;

namespace TBTK{

template<>
Canvas<unsigned int>::Canvas(unsigned int width, unsigned int height){
	data = Mat::zeros(height, width, CV_8UC3);

	origin[0] = 0;
	origin[1] = 0;

	basisVectors[0][0] = width;
	basisVectors[0][1] = 0;
	basisVectors[1][0] = 0;
	basisVectors[1][1] = height;

	calculateNorms();
}

template<>
Canvas<double>::Canvas(unsigned int width, unsigned int height){
	data = Mat::zeros(height, width, CV_8UC3);

	origin[0] = 0.;
	origin[1] = 0.;

	basisVectors[0][0] = 1.;
	basisVectors[0][1] = 0.;
	basisVectors[1][0] = 0.;
	basisVectors[1][1] = 1.;

	calculateNorms();
}

};	//End of namespace TBTK
