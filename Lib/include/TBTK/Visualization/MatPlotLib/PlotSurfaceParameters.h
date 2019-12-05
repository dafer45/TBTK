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
 *  @file PlotSurfaceParameters.h
 *  @brief Parameter container for plotting using the matplotlib function
 *  plot_surface.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_PLOT_SURFACE_PARAMETERS
#define COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_PLOT_SURFACE_PARAMETERS

#include "TBTK/Visualization/MatPlotLib/matplotlibcpp.h"

#include <string>
#include <utility>

namespace TBTK{
namespace Visualization{
namespace MatPlotLib{

class PlotSurfaceParameters{
public:
	/** Default constructor. */
	PlotSurfaceParameters();

	/** Set title. */
	void setTitle(const std::string &title, bool overwrite = true);

	/** Set x-label. */
	void setLabelX(const std::string &labelX, bool overwrite = true);

	/** Set y-label. */
	void setLabelY(const std::string &labelY, bool overwrite = true);

	/** Set z-label. */
	void setLabelZ(const std::string &labelZ, bool overwrite = true);

	/** Set the bounds for the x-axis. */
	void setBoundsX(double minX, double maxX, bool overwrite = true);

	/** Set the bounds for the y-axis. */
	void setBoundsY(double minY, double maxY, bool overwrite = true);

	/** Set rotation. */
	void setRotation(int elevation, int azimuthal, bool overwrite = true);

	/** Flush parameters to matplotlib. */
	void flush() const;

	/** Clear parameters. */
	void clear();
private:
	std::pair<bool, std::string> title;
	std::pair<bool, std::string> labelX;
	std::pair<bool, std::string> labelY;
	std::pair<bool, std::string> labelZ;
	std::pair<bool, std::string> elevation;
	std::pair<bool, std::string> azimuthal;
	std::pair<bool, double> minX;
	std::pair<bool, double> maxX;
	std::pair<bool, double> minY;
	std::pair<bool, double> maxY;
};

inline PlotSurfaceParameters::PlotSurfaceParameters(
) :
	title(false, ""),
	labelX(false, ""),
	labelY(false, ""),
	labelZ(false, ""),
	elevation(false, "30"),
	azimuthal(false, "-60"),
	minX(false, 0),
	maxX(false, 1),
	minY(false, 0),
	maxY(false, 1)
{
}

inline void PlotSurfaceParameters::setTitle(
	const std::string &title,
	bool overwrite
){
	if(overwrite || !this->title.first)
		this->title = {true, title};
}

inline void PlotSurfaceParameters::setLabelX(
	const std::string &labelX,
	bool overwrite
){
	if(overwrite || !this->labelX.first)
		this->labelX = {true, labelX};
}

inline void PlotSurfaceParameters::setLabelY(
	const std::string &labelY,
	bool overwrite
){
	if(overwrite || !this->labelY.first)
		this->labelY = {true, labelY};
}

inline void PlotSurfaceParameters::setLabelZ(
	const std::string &labelZ,
	bool overwrite
){
	if(overwrite || !this->labelZ.first)
		this->labelZ = {true, labelZ};
}

inline void PlotSurfaceParameters::setRotation(
	int elevation,
	int azimuthal,
	bool overwrite
){
	if(overwrite || !this->elevation.first){
		this->elevation = {true, std::to_string(elevation)};
		this->azimuthal = {true, std::to_string(azimuthal)};
	}
}

inline void PlotSurfaceParameters::setBoundsX(
	double minX,
	double maxX,
	bool overwrite
){
	if(overwrite || !this->minX.first)
		this->minX = {true, minX};
	if(overwrite || !this->maxX.first)
		this->maxX = {true, maxX};
}

inline void PlotSurfaceParameters::setBoundsY(
	double minY,
	double maxY,
	bool overwrite
){
	if(overwrite || !this->minY.first)
		this->minY = {true, minY};
	if(overwrite || !this->maxY.first)
		this->maxY = {true, maxY};
}

inline void PlotSurfaceParameters::flush() const{
	if(title.first)
		matplotlibcpp::title(title.second);
	if(labelX.first)
		matplotlibcpp::xlabel(labelX.second);
	if(labelY.first)
		matplotlibcpp::ylabel(labelY.second);
	if(elevation.first){
		matplotlibcpp::view_init({
			{"elev", elevation.second},
			{"azim", azimuthal.second}
		});
	}
	if(minX.first)
		matplotlibcpp::xlim(minX.second, maxX.second);
	if(minY.first)
		matplotlibcpp::ylim(minY.second, maxY.second);
}

inline void PlotSurfaceParameters::clear(){
	title = {false, ""};
	labelX = {false, ""};
	labelY = {false, ""};
	labelZ = {false, ""};
	elevation = {false, ""};
	azimuthal = {false, ""};
}

};	//End namespace MatPlotLib
};	//End namespace Visualization
};	//End namespace TBTK

#endif
