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

#include "Array.h"
#include "Decoration.h"
#include "DOS.h"
#include "Path.h"
#include "Streams.h"
#include "TBTKMacros.h"

#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace TBTK{
namespace Plot{

class Plotter{
public:
	/** Constructor. */
	Plotter();

	/** Destructor. */
	~Plotter();

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

	/** Set auto scale. */
	void setAutoScaleX(bool autoScaleX);

	/** Set auto scale. */
	void setAutoScaleY(bool autoScaleY);

	/** Set auto scale. */
	void setAutoScale(bool autoScale);

	/** Set canvas. */
	void setCanvas(cv::Mat &canvas);

	/** Get canvas. */
	const cv::Mat& getCanvas() const;

	/** Class describing how to decorate the data with line style, color,
	 *  etc. */
//	class Decoration{
//	public:
		/** Enum class for describingthe line style. */
//		enum class LineStyle {Line, Point};

		/** Constructor. */
//		Decoration(
//			const std::vector<unsigned char> &color,
//			LineStyle lineStyle,
//			unsigned int size = 1
//		);

		/** Destructor. */
//		~Decoration();
//	private:
		/** Color. */
//		std::vector<unsigned char> color;

		/** Line style. */
//		LineStyle lineStyle;

		/** Width. */
//		unsigned int size;

		/** The Plotter have direct access to Decoration members. */
//		friend class Plotter;
//	};

	/** Plot data. */
	void plot(
		const std::vector<double> &axis,
		const std::vector<double> &data,
		const Decoration &decoration = Decoration(
			{0, 0, 0}, Decoration::LineStyle::Line
		)
	);

	/** Plot data. */
	void plot(
		const std::vector<double> &data,
		const Decoration &decoration = Decoration(
			{0, 0, 0}, Decoration::LineStyle::Line
		)
	);

	/** Plot density of states. */
	void plot(
		const Property::DOS &dos,
		double sigma = 0,
		unsigned int windowSize = 51
	);

	/** Plot 2D data. */
	void plot(
		const std::vector<std::vector<double>> &data
	);

	/** Plot data. */
	void plot(
		const Array<double> &data,
		const Decoration &decoration = Decoration(
			{0, 0, 0}, Decoration::LineStyle::Line
		)
	);

	/** Set whether ot not data is plotted on top of old data. */
	void setHold(bool hold);

	/** Clear plot. */
	void clear();

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

	/** Flags indicating whether to auto scale along x and y direction. */
	bool autoScaleX, autoScaleY;

	/** Flag indicating whether data is ploted on top of previous data or
	 *  not. */
	bool hold;

	/** Storage for ploted data. Used if holde is true. */
	std::vector<
		std::tuple<std::vector<double>, std::vector<double>, Decoration>
	> dataStorage;

	/** Converts a coordinate to a cvPoint that can be used as canvas
	 *  /coordinate. */
	cv::Point getCVPoint(double x, double y) const;

	/** Draw axes. */
	void drawAxes();
};

inline void Plotter::setWidth(unsigned int width){
	this->width = width;
}

inline void Plotter::setHeight(unsigned int height){
	this->height = height;
}

inline void Plotter::setPadding(
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

inline void Plotter::setBoundsX(
	double minX,
	double maxX
){
	TBTKAssert(
		minX < maxX,
		"Plotter::setBoundsX()",
		"minX has to be smaller than maxX",
		""
	);
	this->minX = minX;
	this->maxX = maxX;
	this->autoScaleX = false;
}

inline void Plotter::setBoundsY(
	double minY,
	double maxY
){
	TBTKAssert(
		minY < maxY,
		"Plotter::setBoundsY()",
		"minY has to be smaller than maxY",
		""
	);
	this->minY = minY;
	this->maxY = maxY;
	this->autoScaleY = false;
}

inline void Plotter::setBounds(
	double minX,
	double maxX,
	double minY,
	double maxY
){
	setBoundsX(minX, maxX);
	setBoundsY(minY, maxY);
}

inline void Plotter::setAutoScaleX(bool autoScaleX){
	this->autoScaleX = autoScaleX;
}

inline void Plotter::setAutoScaleY(bool autoScaleY){
	this->autoScaleY = autoScaleY;
}

inline void Plotter::setAutoScale(bool autoScale){
	setAutoScaleX(autoScale);
	setAutoScaleY(autoScale);
}

inline void Plotter::setCanvas(cv::Mat &canvas){
	this->canvas = canvas;
}

inline const cv::Mat& Plotter::getCanvas() const{
	return canvas;
}

inline cv::Point Plotter::getCVPoint(double x, double y) const{
	double width = maxX - minX;
	double height = maxY - minY;
	return cv::Point(
		paddingLeft + (1 - (paddingLeft + paddingRight)/(double)canvas.cols)*canvas.cols*(x - minX)/(double)width,
		canvas.rows - 1 - (paddingBottom + (1 - (paddingBottom + paddingTop)/(double)canvas.rows)*canvas.rows*(y - minY)/(double)height)
	);
}

inline void Plotter::setHold(bool hold){
	this->hold = hold;
}

inline void Plotter::clear(){
	this->dataStorage.clear();
}

inline void Plotter::save(std::string filename) const{
	imwrite(filename, canvas);
}

/*inline Plotter::Decoration::Decoration(
	const std::vector<unsigned char> &color,
	LineStyle lineStyle,
	unsigned int size
){
	this->color = color;
	this->lineStyle = lineStyle;
	this->size = size;
}

inline Plotter::Decoration::~Decoration(){
}*/

};	//End namespace Plot
};	//End namespace TBTK

#endif
