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
 *  @file ContourfParameters.h
 *  @brief Parameter container for plotting using the matplotlib function
 *  contourf.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_CONTOURF_PARAMETERS
#define COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_CONTOURF_PARAMETERS

#include "TBTK/Visualization/MatPlotLib/matplotlibcpp.h"

#include <string>
#include <utility>

namespace TBTK{
namespace Visualization{
namespace MatPlotLib{

class ContourfParameters{
public:
	/** Default constructor. */
	ContourfParameters();

	/** Set title. */
	void setTitle(const std::string &title);

	/** Set x-label. */
	void setLabelX(const std::string &labelX);

	/** Set y-label. */
	void setLabelY(const std::string &labelY);

	/** Flush parameters to matplotlib. */
	void flush() const;
private:
	std::pair<bool, std::string> title;
	std::pair<bool, std::string> labelX;
	std::pair<bool, std::string> labelY;
};

inline ContourfParameters::ContourfParameters(
) :
	title(false, ""),
	labelX(false, ""),
	labelY(false, "")
{
}

inline void ContourfParameters::setTitle(const std::string &title){
	this->title = {true, title};
}

inline void ContourfParameters::setLabelX(const std::string &labelX){
	this->labelX = {true, labelX};
}

inline void ContourfParameters::setLabelY(const std::string &labelY){
	this->labelY = {true, labelY};
}

inline void ContourfParameters::flush() const{
	if(title.first)
		matplotlibcpp::title(title.second);
	if(labelX.first)
		matplotlibcpp::xlabel(labelX.second);
	if(labelY.first)
		matplotlibcpp::ylabel(labelY.second);
}

};	//End namespace MatPlotLib
};	//End namespace Visualization
};	//End namespace TBTK

#endif
