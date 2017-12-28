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

#include "TBTKMacros.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace TBTK{
namespace Plot{

class PlotCanvas{
public:
	/** Constructor. */
	PlotCanvas();

	/** Destructor. */
	~PlotCanvas();

	/** Set width. */
	void setWidth(unsigned int width);

	/** Set height. */
	void setHeight(unsigned int height);

	/** Set padding. */
	void setPadding(
		double paddingLeft,
		double paddingRight,
		double paddingBottom,
		double paddingTop
	);

	/** Set bounds. */
	void setBoundsX(double minX, double maxX);

	/** Set bounds. */
	void setBoundsY(double minY, double maxY);

	/** Set bounds. */
	void setBounds(double minX, double maxX, double minY, double maxY);

	/** Get minimum X-value. */
	double getMinX() const;

	/** Get maximum X-value. */
	double getMaxX() const;

	/** Get minimum Y-value. */
	double getMinY() const;

	/** Get maximum Y-value. */
	double getMaxY() const;

	/** Set canvas. */
	void setCanvas(cv::Mat &canvas);

	/** Get canvas. */
	const cv::Mat& getCanvas() const;

	/** Clear plot. */
	void clear();

	/** Set pixel. */
	void setPixel(
		unsigned int x,
		unsigned int y,
		unsigned char red,
		unsigned char green,
		unsigned char blue
	);

	/** Draw line. */
	void drawLine(
		double x0,
		double y0,
		double x1,
		double y1,
		const std::vector<unsigned char> &color,
		unsigned int width
	);

	/** Draw circle. */
	void drawCircle(
		double x,
		double y,
		unsigned int size,
		const std::vector<unsigned char> &color
	);

	/** Save canvas to file. */
	void save(std::string filename) const;
private:
	/** Canvas. */
	cv::Mat canvas;

	/** Size of the resulting image. */
	double width, height;

	/** Paddings. */
	unsigned int paddingLeft, paddingRight, paddingBottom, paddingTop;

	/** Bounds. */
	double minX, maxX, minY, maxY;
public:
	/** Converts a coordinate to a cvPoint that can be used as canvas
	 *  /coordinate. */
	cv::Point getCVPoint(double x, double y) const;

	/** Draw axes. */
	void drawAxes();
};

inline void PlotCanvas::setWidth(unsigned int width){
	this->width = width;
}

inline void PlotCanvas::setHeight(unsigned int height){
	this->height = height;
}

inline void PlotCanvas::setPadding(
	double paddingLeft,
	double paddingRight,
	double paddingBottom,
	double paddingTop
){
	this->paddingLeft = paddingLeft;
	this->paddingRight = paddingRight;
	this->paddingBottom = paddingBottom;
	this->paddingTop = paddingTop;
}

inline void PlotCanvas::setBoundsX(
	double minX,
	double maxX
){
	TBTKAssert(
		minX < maxX,
		"Canvas::setBoundsX()",
		"minX has to be smaller than maxX.",
		""
	);
	this->minX = minX;
	this->maxX = maxX;
}

inline void PlotCanvas::setBoundsY(
	double minY,
	double maxY
){
	TBTKAssert(
		minY < maxY,
		"Canvas::setBoundsY()",
		"minY has to be smaller than maxY.",
		""
	);
	this->minY = minY;
	this->maxY = maxY;
}

inline void PlotCanvas::setBounds(
	double minX,
	double maxX,
	double minY,
	double maxY
){
	setBoundsX(minX, maxX);
	setBoundsY(minY, maxY);
}

inline double PlotCanvas::getMinX() const{
	return minX;
}

inline double PlotCanvas::getMaxX() const{
	return maxX;
}

inline double PlotCanvas::getMinY() const{
	return minY;
}

inline double PlotCanvas::getMaxY() const{
	return maxY;
}

inline void PlotCanvas::setCanvas(cv::Mat &canvas){
	this->canvas = canvas;
}

inline const cv::Mat& PlotCanvas::getCanvas() const{
	return canvas;
}

inline cv::Point PlotCanvas::getCVPoint(double x, double y) const{
	double width = maxX - minX;
	double height = maxY - minY;
	return cv::Point(
		paddingLeft + (1 - (paddingLeft + paddingRight)/(double)canvas.cols)*canvas.cols*(x - minX)/(double)width,
		canvas.rows - 1 - (paddingBottom + (1 - (paddingBottom + paddingTop)/(double)canvas.rows)*canvas.rows*(y - minY)/(double)height)
	);
}

inline void PlotCanvas::clear(){
	canvas = cv::Mat::zeros(height, width, CV_8UC3);
	cv::rectangle(
		canvas,
		cvPoint(0, 0),
		cvPoint(width-1, height-1),
		cv::Scalar(255, 255, 255),
		CV_FILLED,
		8,
		0
	);
}

inline void PlotCanvas::setPixel(
	unsigned int x,
	unsigned int y,
	unsigned char red,
	unsigned char green,
	unsigned char blue
){
	canvas.at<cv::Vec3b>(y, x)[0] = blue;
	canvas.at<cv::Vec3b>(y, x)[1] = green;
	canvas.at<cv::Vec3b>(y, x)[2] = red;
}

inline void PlotCanvas::drawLine(
	double x0,
	double y0,
	double x1,
	double y1,
	const std::vector<unsigned char> &color,
	unsigned int width
){
	//Clip lines
	if(x1 < x0){
		double temp = x0;
		x0 = x1;
		x1 = temp;
		temp = y0;
		y0 = y1;
		y1 = temp;
	}
	if(x0 < minX && x1 < minX)
		return;
	if(x0 > maxX && x1 > maxX)
		return;
	if(x0 < minX){
		if(x1 - x0 != 0)
			y0 += (minX - x0)*(y1 - y0)/(x1 - x0);
		x0 = minX;
	}
	if(x1 > maxX){
		if(x1 - x0 != 0)
			y1 -= (x1 - maxX)*(y1 - y0)/(x1 - x0);
		x1 = maxX;
	}
	if(y0 < minY && y1 < minY)
		return;
	if(y0 > maxY && y1 > maxY)
		return;
	if(y0 < minY){
		if(y1 - y0 != 0)
			x0 += (minY - y0)*(x1 - x0)/(y1 - y0);
		y0 = minY;
	}
	if(y1 > maxY){
		if(y1 - y0 != 0)
			x1 -= (y1 - maxY)*(x1 - x0)/(y1 - y0);
		y1 = maxY;
	}

	//Draw line
	cv::line(
		canvas,
		getCVPoint(x0, y0),
		getCVPoint(x1, y1),
		cv::Scalar(color[2], color[1], color[0]),
		width,
		CV_AA
	);
}

inline void PlotCanvas::drawCircle(
	double x,
	double y,
	unsigned int size,
	const std::vector<unsigned char> &color
){
	//Clip point
	if(x < minX)
		return;
	if(x > maxX)
		return;
	if(y < minY)
		return;
	if(y > maxY)
		return;

	//Draw cirlce
	cv::circle(
		canvas,
		getCVPoint(x, y),
		size,
		cv::Scalar(color[2], color[1], color[0]),
		-1,
		CV_AA
	);
}

inline void PlotCanvas::save(std::string filename) const{
	imwrite(filename, canvas);
}

};	//End namespace Plot
};	//End namespace TBTK

#endif
