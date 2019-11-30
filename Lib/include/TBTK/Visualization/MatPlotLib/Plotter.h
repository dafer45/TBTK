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
#include "TBTK/Property/Density.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/Visualization/MatPlotLib/Argument.h"
#include "TBTK/Visualization/MatPlotLib/ContourfParameters.h"
#include "TBTK/Visualization/MatPlotLib/matplotlibcpp.h"
#include "TBTK/Visualization/MatPlotLib/PlotParameters.h"
#include "TBTK/Visualization/MatPlotLib/PlotSurfaceParameters.h"

#include <string>
#include <tuple>
#include <vector>


namespace TBTK{
namespace Visualization{
namespace MatPlotLib{

class Plotter{
public:
	/** Default constructor. */
	Plotter();

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

	/** Set title. */
	void setTitle(const std::string &title);

	/** Set x-label. */
	void setLabelX(const std::string &labelX);

	/** Set y-label. */
	void setLabelY(const std::string &labelY);

	/** Set z-label. */
	void setLabelZ(const std::string &labelZ);

	/** Plot point. */
//	void plot(double x, double y, const std::string &arguments);

	/** Plot density. */
	void plot(
		const Property::Density &density,
		const Argument &argument = ""
	);

	/** Plot density. */
	void plot(
		const Index &pattern,
		const Property::Density &density,
		const Argument &argument = ""
	);

	/** Plot density of states. */
	void plot(
		const Property::DOS &dos,
		double sigma = 0,
		unsigned int windowSize = 51,
		const Argument &argument = ""
	);

	/** Plot eigenvalues. */
	void plot(
		const Property::EigenValues &eigenValues,
		const Argument &argument = "black"
	);

	/** Plot LDOS. */
	void plot(const Property::LDOS &ldos, const Argument &argument = "");

	/** Plot LDOS. */
	void plot(
		const Index &pattern,
		const Property::LDOS &ldos,
		const Argument &argument = ""
	);

	/** Plot data. */
	void plot(
		const Array<double> &data,
		const Argument &argument = ""
	);

	/** Plot data. */
	void plot(
		const std::initializer_list<double> &data,
		const Argument &argument = ""
	){
		plot(std::vector<double>(data), argument);
	}

	/** Plot data. */
	void plot(
		const Array<double> &x,
		const Array<double> &y,
		const Argument &argument = ""
	);

	/** Plot data. */
	void plot(
		const std::initializer_list<double> &x,
		const Array<double> &y,
		const Argument &argument = ""
	){
		plot(std::vector<double>(x), y, argument);
	}

	/** Plot data. */
	void plot(
		const Array<double> &x,
		const std::initializer_list<double> &y,
		const Argument &argument = ""
	){
		plot(x, std::vector<double>(y), argument);
	}

	/** Plot data. */
	void plot(
		const std::initializer_list<double> &x,
		const std::initializer_list<double> &y,
		const Argument &argument = ""
	){
		plot(
			std::vector<double>(x),
			std::vector<double>(y),
			argument
		);
	}

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

	/** Set plot method to use for 3D data. */
	void setPlotMethod3D(const std::string &plotMethod3D);

	/** Set rotation angels. */
	void setRotation(int elevation, int azimuthal);

	/** Set whether ot not data is plotted on top of old data. */
//	void setHold(bool hold);

	/** Clear plot. */
	void clear();

	/** Show the plot. */
	void show() const;

	/** Save canvas to file. */
	void save(const std::string &filename) const;
private:
	/** Enum class for keeping track of the current type of plot. */
	enum class CurrentPlotType{None, Plot1D, PlotSurface, Contourf};

	/** Current plot type. */
	CurrentPlotType currentPlotType;

	/** Enum class for keeping track of the plot method to use for 3D data.
	 */
	enum class PlotMethod3D{PlotSurface, Contourf};

	/** The plot method to use for 3D data. */
	PlotMethod3D plotMethod3D;

	/** Parameters for plots using plot. */
	PlotParameters plotParameters;

	/** Parameters for plots using plot_surface. */
	PlotSurfaceParameters plotSurfaceParameters;

	/** Parameters for plots using contourf. */
	ContourfParameters contourfParameters;

	/** Plot data. */
	void plot1D(
		const std::vector<double> &y,
		const Argument &argument = ""
	);

	/** Plot data. */
	void plot1D(
		const std::vector<double> &x,
		const std::vector<double> &y,
		const Argument &argument = ""
	);

	/** Plot 2D data. */
	void plot2D(
		const std::vector<std::vector<double>> &data,
		const Argument &argument = ""
	);
};

inline Plotter::Plotter(){
	currentPlotType = CurrentPlotType::None;
	plotMethod3D = PlotMethod3D::PlotSurface;
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

inline void Plotter::setTitle(const std::string &title){
	plotParameters.setTitle(title);
	plotSurfaceParameters.setTitle(title);
	contourfParameters.setTitle(title);
	matplotlibcpp::title(title);
}

inline void Plotter::setLabelX(const std::string &labelX){
	plotParameters.setLabelX(labelX);
	plotSurfaceParameters.setLabelX(labelX);
	contourfParameters.setLabelX(labelX);
	matplotlibcpp::xlabel(labelX);
}

inline void Plotter::setLabelY(const std::string &labelY){
	plotParameters.setLabelY(labelY);
	plotSurfaceParameters.setLabelY(labelY);
	contourfParameters.setLabelY(labelY);
	matplotlibcpp::ylabel(labelY);
}

inline void Plotter::setLabelZ(const std::string &labelZ){
	plotSurfaceParameters.setLabelZ(labelZ);
}

inline void Plotter::setPlotMethod3D(const std::string &plotMethod3D){
	if(plotMethod3D.compare("plot_surface") == 0){
		this->plotMethod3D = PlotMethod3D::PlotSurface;
	}
	else if(plotMethod3D.compare("contourf") == 0){
		this->plotMethod3D = PlotMethod3D::Contourf;
	}
	else{
		TBTKExit(
			"Plotter::setPlotMethod3D()",
			"Unknown plot method.",
			"Must be 'plot_surface' or 'contourf'."
		);
	}
}

inline void Plotter::setRotation(int elevation, int azimuthal){
	plotSurfaceParameters.setRotation(elevation, azimuthal);
	switch(currentPlotType){
	case CurrentPlotType::PlotSurface:
		plotSurfaceParameters.flush();
		break;
	default:
		break;
	}
/*	this->elevation = elevation;
	this->azimuthal = azimuthal;
	switch(currentPlotType){
	case CurrentPlotType::PlotSurface:
		matplotlibcpp::view_init({
			{"elev", std::to_string(elevation)},
			{"azim", std::to_string(azimuthal)}
		});
	default:
		break;
	}*/
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
