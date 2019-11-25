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

/** @package TBTKcalc
 *  @file Plotter.h
 *  @brief Plotter
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_PLOTTER
#define COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_PLOTTER

#include "TBTK/Array.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <string>
#include <tuple>
#include <vector>

#include "TBTK/Visualization/MatPlotLib/matplotlibcpp.h"

namespace TBTK{
namespace Visualization{
namespace MatPlotLib{

class Plotter{
public:
	/** Set size. */
	void setSize(unsigned int width, unsigned int height);

	/** Set bounds. */
	void setBoundsX(double minX, double maxX);

	/** Set bounds. */
	void setBoundsY(double minY, double maxY);

	/** Set bounds. */
	void setBounds(double minX, double maxX, double minY, double maxY);

	/** Set auto scale. */
//	void setAutoScaleX(bool autoScaleX);

	/** Set auto scale. */
//	void setAutoScaleY(bool autoScaleY);

	/** Set auto scale. */
//	void setAutoScale(bool autoScale);

	/** Set x-label. */
	void setLabelX(const std::string &labelX);

	/** Set y-label. */
	void setLabelY(const std::string &labelY);

	/** Plot point. */
//	void plot(double x, double y, const std::string &arguments);

	/** Plot data. */
	void plot(
		const std::vector<double> &x,
		const std::vector<double> &y,
		const std::string &arguments = ""
	);

	/** Plot data. */
	void plot(
		const std::vector<double> &y,
		const std::string &arguments = ""
	);

	/** Plot density of states. */
	void plot(
		const Property::DOS &dos,
		double sigma = 0,
		unsigned int windowSize = 51
	);

	/** Plot eigenvalues. */
//	void plot(const Property::EigenValues &eigenValues);

	/** Plot 2D data. */
	void plot(
		const std::vector<std::vector<double>> &data
	);

	/** Plot data. */
	void plot(
		const Array<double> &data,
		const std::string &arguments = ""
	);

	/** Plot data with color coded intensity. */
/*	void plot(
		const std::vector<std::vector<double>> &data,
		const std::vector<std::vector<double>> &intensity,
		const std::string &arguments
	);*/

	/** Plot data with color coded intensity. */
/*	void plot(
		const Array<double> &data,
		const Array<double> &intensity,
		const std::string &arguments
	);*/

	/** Set whether ot not data is plotted on top of old data. */
//	void setHold(bool hold);

	/** Clear plot. */
	void clear();

	/** Show the plot. */
	void show() const;

	/** Save canvas to file. */
	void save(const std::string &filename) const;
};

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
	matplotlibcpp::xlim(minX, maxX);
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
	matplotlibcpp::ylim(minY, maxY);
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

/*inline void Plotter::setAutoScaleX(bool autoScaleX){
	this->autoScaleX = autoScaleX;
}

inline void Plotter::setAutoScaleY(bool autoScaleY){
	this->autoScaleY = autoScaleY;
}

inline void Plotter::setAutoScale(bool autoScale){
	setAutoScaleX(autoScale);
	setAutoScaleY(autoScale);
}*/

inline void Plotter::setLabelX(const std::string &labelX){
	matplotlibcpp::xlabel(labelX);
}

inline void Plotter::setLabelY(const std::string &labelY){
	matplotlibcpp::ylabel(labelY);
}

/*inline void Plotter::setHold(bool hold){
	this->hold = hold;
}*/

inline void Plotter::clear(){
	matplotlibcpp::clf();
}

inline void Plotter::show() const{
	matplotlibcpp::show();
}

inline void Plotter::save(const std::string &filename) const{
	matplotlibcpp::save(filename);
}

};	//End namespace MatPlotLib
};	//End namespace Visualization
};	//End namespace TBTK

#endif
