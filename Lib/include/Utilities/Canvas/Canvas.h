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
 *  @file Canvas.h
 *  @brief Canvas
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CANVAS
#define COM_DAFER45_TBTK_CANVAS

#include "Streams.h"
#include "TBTKMacros.h"

#include <initializer_list>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace TBTK{

template<typename CoordinateType>
class Canvas{
public:
	class RGBA{
	public:
		/** Constructor. */
		RGBA(char r, char g, char b, char a = 0);
	private:
		/** Values. */
		char r, g, b, a;

		friend class Canvas;
	};

	class Pixel{
	public:
		/** Constructor. */
		Pixel(unsigned int x, unsigned int y);

		/** Coordinates. */
		unsigned int x, y;
	};

	/** Constructor. */
	Canvas(unsigned int width, unsigned int height);

	/** Destructor. */
	~Canvas();

	/** Set origin. */
	void setOrigin(CoordinateType x, CoordinateType y);

	/** Set basis vectors. */
	void setBasisVectors(
		std::initializer_list<std::initializer_list<CoordinateType>> basisVectors
	);

	/** Draw pixel. */
	void drawPixel(const RGBA &rgba, CoordinateType x, CoordinateType y);

	/** Draw circle. */
	void drawCircle(const RGBA &rgba, CoordinateType x, CoordinateType y, CoordinateType radius);

	/** Save canvas to file. */
	void save(std::string filename) const;
private:
	/** Data. */
	cv::Mat data;

	/** Origin. */
	CoordinateType origin[2];

	/** Basis vectors. */
	CoordinateType basisVectors[2][2];

	/** Basis vector norms. */
	double norms[2];

	/** Get pixel coordinate. */
	Pixel getPixel(CoordinateType x, CoordinateType y);

	/** Calculate norms. */
	void calculateNorms();
};

template<typename CoordinateType>
Canvas<CoordinateType>::~Canvas(){
}

template<typename CoordinateType>
void Canvas<CoordinateType>::setOrigin(CoordinateType x, CoordinateType y){
	origin[0] = x;
	origin[1] = y;
}

template<>
inline void Canvas<unsigned int>::drawPixel(
	const RGBA &rgba,
	unsigned int x,
	unsigned int y
){
	if(x < (unsigned int)data.cols && y < (unsigned int)data.rows){
		data.at<cv::Vec3b>(y, x)[0] = rgba.b;
		data.at<cv::Vec3b>(y, x)[1] = rgba.g;
		data.at<cv::Vec3b>(y, x)[2] = rgba.r;
	}
}

template<>
inline void Canvas<double>::drawPixel(
	const RGBA &rgba,
	double x,
	double y
){
	Pixel p = getPixel(x, y);
	if(p.x >= 0
		&& p.x < (unsigned int)data.cols
		&& p.y >= 0
		&& p.y < (unsigned int)data.rows
	){
		data.at<cv::Vec3b>(p.y, p.x)[0] = rgba.b;
		data.at<cv::Vec3b>(p.y, p.x)[1] = rgba.g;
		data.at<cv::Vec3b>(p.y, p.x)[2] = rgba.r;
	}
}

template<>
inline void Canvas<unsigned int>::drawCircle(
	const RGBA &rgba,
	unsigned int x,
	unsigned int y,
	unsigned int radius
){
//	if(x < (unsigned int)data.cols && y < (unsigned int)data.rows){
		cv::circle(data, cv::Point(x, y), radius, cv::Scalar(rgba.r, rgba.g, rgba.b), CV_FILLED, 8, 0);
//	}
}

template<>
inline void Canvas<double>::drawCircle(
	const RGBA &rgba,
	double x,
	double y,
	double radius
){
	Pixel p = getPixel(x, y);
/*	if(p.x >= 0
		&& p.x < (unsigned int)data.cols
		&& p.y >= 0
		&& p.y < (unsigned int)data.rows
	){*/
		cv::circle(data, cv::Point(p.x, p.y), (int)radius, cv::Scalar(rgba.r, rgba.g, rgba.b));
//	}
}

template<typename CoordinateType>
inline typename Canvas<CoordinateType>::Pixel Canvas<CoordinateType>::getPixel(
	CoordinateType x,
	CoordinateType y
){
	double projections[2];
	for(unsigned int n = 0; n < 2; n++){
		projections[n]
			= ((double)x - (double)origin[0])*((double)basisVectors[n][0])
			+ ((double)y - (double)origin[1])*((double)basisVectors[n][1]);
		projections[n] /= norms[n]*norms[n];
	}

	return Pixel(data.cols*projections[0], data.rows*projections[1]);
}

template<typename CoordinateType>
inline void Canvas<CoordinateType>::setBasisVectors(
	std::initializer_list<std::initializer_list<CoordinateType>> basisVectors
){
	TBTKAssert(
		basisVectors.size() == 2,
		"Canvas::setBasisVectors()",
		"'basisVectors' must contain two vectors.",
		""
	);
	for(unsigned int n = 0; n < 2; n++){
		TBTKAssert(
			(basisVectors.begin() + n)->size() == 2,
			"Canvas::setBasisVectors",
			"Each basis vector must have two components, but basis"
			<< " vector " << n << " has "
			<< (basisVectors.begin() + n)->size()
			<< " components.",
			""
		);
	}

	for(unsigned int n = 0; n < 2; n++)
		for(unsigned int c = 0; c < 2; c++)
			this->basisVectors[n][c] = *((basisVectors.begin() + n)->begin() + c);

	calculateNorms();
}

template<typename CoordinateType>
inline void Canvas<CoordinateType>::save(std::string filename) const{
	imwrite(filename, data);
}

template<typename CoordinateType>
inline Canvas<CoordinateType>::RGBA::RGBA(char r, char g, char b, char a){
	this->r = r;
	this->g = g;
	this->b = b;
	this->a = a;
}

template<typename CoordinateType>
inline Canvas<CoordinateType>::Pixel::Pixel(unsigned int x, unsigned int y){
	this->x = x;
	this->y = y;
}

template<typename CoordinateType>
inline void Canvas<CoordinateType>::calculateNorms(){
	for(unsigned int n = 0; n < 2; n++){
		norms[n] = 0;
		for(unsigned int c = 0; c < 2; c++)
			norms[n] += basisVectors[n][c]*basisVectors[n][c];
		norms[n] = sqrt(norms[n]);
	}
}

};	//End namespace TBTK

#endif
