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
#include "PlotCanvas.h"
#include "Decoration.h"
#include "DOS.h"
#include "Path.h"
#include "Point.h"
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

	/** Set x-label. */
	void setLabelX(const std::string &labelX);

	/** Set y-label. */
	void setLabelY(const std::string &labelY);

	/** Set canvas. */
	void setCanvas(cv::Mat &canvas);

	/** Get canvas. */
	const cv::Mat& getCanvas();

	/** Plot point. */
	void plot(
		double x,
		double y,
		const Decoration &decoration = Decoration(
			{0, 0, 0}, Decoration::LineStyle::Point
		)
	);

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

	/** Plot data with color coded intensity. */
	void plot(
		const std::vector<std::vector<double>> &data,
		const std::vector<std::vector<double>> &intensity,
		const Decoration &decoration = Decoration(
			{0, 0, 0}, Decoration::LineStyle::Point
		)
	);

	/** Plot data with color coded intensity. */
	void plot(
		const Array<double> &data,
		const Array<double> &intensity,
		const Decoration &decoration = Decoration(
			{0, 0, 0}, Decoration::LineStyle::Point
		)
	);

	/** Set whether ot not data is plotted on top of old data. */
	void setHold(bool hold);

	/** Clear plot. */
	void clear();

	/** Save canvas to file. */
	void save(std::string filename);
private:
	/** Canvas. */
	PlotCanvas canvas;

	/** Flags indicating whether to auto scale along x and y direction. */
	bool autoScaleX, autoScaleY;

	/** Flag indicating whether data is ploted on top of previous data or
	 *  not. */
	bool hold;

	/** Data storage. */
	std::vector<Drawable*> dataStorage;

	/** Draw data storage. */
	void drawDataStorage();

	/** Clear data storage. */
	void clearDataStorage();
};

inline void Plotter::setWidth(unsigned int width){
	canvas.setWidth(width);
}

inline void Plotter::setHeight(unsigned int height){
	canvas.setHeight(height);
}

inline void Plotter::setPadding(
	double paddingLeft,
	double paddingRight,
	double paddingBottom,
	double paddingTop
){
	canvas.setPadding(
		paddingLeft,
		paddingRight,
		paddingBottom,
		paddingTop
	);
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
	this->autoScaleX = false;
	canvas.setBoundsX(minX, maxX);
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
	this->autoScaleY = false;
	canvas.setBoundsY(minY, maxY);
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

inline void Plotter::setLabelX(const std::string &labelX){
	canvas.setLabelX(labelX);
}

inline void Plotter::setLabelY(const std::string &labelY){
	canvas.setLabelY(labelY);
}

inline void Plotter::setCanvas(cv::Mat &canvas){
	this->canvas.setCanvas(canvas);
}

inline const cv::Mat& Plotter::getCanvas(){
	if(dataStorage.size() != 0){
		drawDataStorage();
		canvas.drawAxes();
	}

	return canvas.getCanvas();
}

inline void Plotter::setHold(bool hold){
	this->hold = hold;
}

inline void Plotter::clear(){
	clearDataStorage();
	canvas.clear();
}

inline void Plotter::save(std::string filename){
	if(dataStorage.size() != 0){
		drawDataStorage();
		canvas.drawAxes();
	}

	canvas.save(filename);
}

inline void Plotter::clearDataStorage(){
	for(unsigned int n = 0; n < dataStorage.size(); n++)
		delete dataStorage[n];
	dataStorage.clear();
}

};	//End namespace Plot
};	//End namespace TBTK

#endif
