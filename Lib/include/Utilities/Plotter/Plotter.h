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
 *  @file Plotter.h
 *  @brief Plotter
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PLOTTER
#define COM_DAFER45_TBTK_PLOTTER

#include "DOS.h"
#include "Streams.h"
#include "TBTKMacros.h"

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace TBTK{

class Plotter{
public:
	/** Constructor. */
	Plotter();

	/** Destructor. */
	~Plotter();

	/** Set padding. */
	void setPadding(double x, double y);

	/** Set bounds. */
	void setBounds(double minX, double maxX, double minY, double maxY);

	/** Set canvas. */
	void setCanvas(cv::Mat &canvas);

	/** Plot data. */
	void plot(std::vector<double> data);

	/** Plot density of states. */
	void plot(const Property::DOS &dos);
private:
	/** Paddings. */
	unsigned int paddingX, paddingY;

	/** Bounds. */
	double minX, maxX, minY, maxY;

	/** Converts a coordinate to a cvPoint that can be used as canvas
	 *  /coordinate. */
	cv::Point getCVPoint(double x, double y) const;

	/** Canvas. */
	cv::Mat canvas;
};

inline void Plotter::setPadding(double paddingX, double paddingY){
	this->paddingX = paddingX;
	this->paddingY = paddingY;
}

inline void Plotter::setBounds(
	double minX,
	double maxX,
	double minY,
	double maxY
){
	TBTKAssert(
		minX < maxX,
		"Plotter::setBounds()",
		"minX has to be smaller than maxX",
		""
	);
	TBTKAssert(
		minY < maxY,
		"Plotter::setBounds()",
		"minY has to be smaller than maxY",
		""
	);
	this->minX = minX;
	this->maxX = maxX;
	this->minY = minY;
	this->maxY = maxY;
}

inline void Plotter::setCanvas(cv::Mat &canvas){
	this->canvas = canvas;
}

inline cv::Point Plotter::getCVPoint(double x, double y) const{
	double width = maxX - minX;
	double height = maxY - minY;
	return cv::Point(
		paddingX + (1 - 2*paddingX/(double)canvas.cols)*canvas.cols*(x - minX)/(double)width,
		canvas.rows - 1 - (paddingY + (1 - 2*paddingY/(double)canvas.rows)*canvas.rows*(y - minY)/(double)height)
	);
}

};	//End namespace TBTK

#endif
