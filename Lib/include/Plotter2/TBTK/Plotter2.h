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
 *  @file Plotter2.h
 *  @brief Plotter2
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PLOTTER_2
#define COM_DAFER45_TBTK_PLOTTER_2

#include "TBTK/Array.h"
#include "TBTK/Canvas2D.h"
#include "TBTK/Canvas3D.h"
#include "TBTK/PNGCanvas2D.h"
#include "TBTK/PNGCanvas3D.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

#include <sstream>

namespace TBTK{

class Plotter2{
public:
	/** Constructor. */
	Plotter2();

	/** Destructor. */
	~Plotter2();

	/** Set width. */
	void setWidth(unsigned int width);

	/** Set height. */
	void setHeight(unsigned int height);

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

	/** Set title. */
	void setTitle(const std::string &title);

	/** Set x-label. */
	void setLabelX(const std::string &labelX);

	/** Set y-label. */
	void setLabelY(const std::string &labelY);

	/** Set z-label. */
	void setLabelZ(const std::string &labelZ);

	/** Plot data. */
	void plot(
		const std::vector<double> &x,
		const std::vector<double> &y,
		const std::string &title = "",
		const std::vector<unsigned char> &color = {0, 0, 0},
		unsigned int size = 1
	);

	/** Plot data. */
	void plot(
		const std::vector<double> &data,
		const std::string &title = "",
		const std::vector<unsigned char> &color = {0, 0, 0},
		unsigned int size = 1
	);

	/** Plot density of states. */
	void plot(
		const Property::DOS &dos,
		double sigma = 0,
		unsigned int windowSize = 51
	);

	/** Plot eigenvalues. */
	void plot(const Property::EigenValues &eigenValues);

	/** Plot 2D data. */
	void plot(
		const std::vector<std::vector<double>> &data,
		const std::string &title = ""
	);

	/** Plot data. */
	void plot(
		const Array<double> &data,
		const std::string &title = "",
		const std::vector<unsigned char> &color = {0, 0, 0},
		unsigned int size = 1
	);

	/** Plot data with color coded intensity. */
/*	void plot(
		const std::vector<std::vector<double>> &data,
		const std::vector<std::vector<double>> &intensity
	);*/

	/** Plot data with color coded intensity. */
/*	void plot(
		const Array<double> &data,
		const Array<double> &intensity
	);*/

	/** Set whether ot not data is plotted on top of old data. */
	void setHold(bool hold);

	/** Set whether the 3D plot should be displayed using a top view or
	 *  not. */
	void setTopView(bool topView);

	/** Clear plot. */
	void clear();

	/** Save canvas to file. */
	void save(std::string filename);
private:
	/** 2D canvas. */
	Canvas2D canvas2D;

	/** 3D canvas. */
	Canvas3D canvas3D;

	/** Pointer to the current canvas. */
	Canvas *currentCanvas;

	/** Flags indicating whether to auto scale along x and y direction. */
	bool autoScaleX, autoScaleY;

	/** Set the current canvas. */
	void setCurrentCanvas(Canvas &canvas);
};

inline void Plotter2::setWidth(unsigned int width){
	canvas2D.setWidth(width);
}

inline void Plotter2::setHeight(unsigned int height){
	canvas2D.setHeight(height);
}

inline void Plotter2::setBoundsX(double minX, double maxX){
	TBTKAssert(
		minX < maxX,
		"Plotter2::setBoundsX()",
		"minX has to be smaller than maxX",
		""
	);
	this->autoScaleX = false;
	canvas2D.setBoundsX(minX, maxX);
}

inline void Plotter2::setBoundsY(double minY, double maxY){
	TBTKAssert(
		minY < maxY,
		"Plotter2::setBoundsY()",
		"minY has to be smaller than maxY",
		""
	);
	this->autoScaleY = false;
	canvas2D.setBoundsY(minY, maxY);
}

inline void Plotter2::setBounds(
	double minX,
	double maxX,
	double minY,
	double maxY
){
	setBoundsX(minX, maxX);
	setBoundsY(minY, maxY);
}

inline void Plotter2::setAutoScaleX(bool autoScaleX){
	this->autoScaleX = autoScaleX;
}

inline void Plotter2::setAutoScaleY(bool autoScaleY){
	this->autoScaleY = autoScaleY;
}

inline void Plotter2::setAutoScale(bool autoScale){
	setAutoScaleX(autoScale);
	setAutoScaleY(autoScale);
}

inline void Plotter2::setTitle(const std::string &title){
	canvas2D.setTitle(title);
	canvas3D.setTitle(title);
}

inline void Plotter2::setLabelX(const std::string &labelX){
	canvas2D.setLabelX(labelX);
	canvas3D.setLabelX(labelX);
}

inline void Plotter2::setLabelY(const std::string &labelY){
	canvas2D.setLabelY(labelY);
	canvas3D.setLabelY(labelY);
}

inline void Plotter2::setLabelZ(const std::string &labelZ){
	canvas3D.setLabelZ(labelZ);
}

inline void Plotter2::setHold(bool hold){
	canvas2D.setHold(hold);
	canvas3D.setHold(hold);
}

inline void Plotter2::setTopView(bool topView){
	setCurrentCanvas(canvas3D);
	canvas3D.setTopView(topView);
}

inline void Plotter2::clear(){
	canvas2D.clear();
}

inline void Plotter2::save(std::string filename){
	std::vector<std::string> tokens;
	std::stringstream ss(filename);
	std::string token;
	while(std::getline(ss, token, '.'))
		tokens.push_back(token);

	TBTKAssert(
		tokens.size() != 0,
		"Plotter2::save()",
		"Invalid filename '" << filename << "'.",
		""
	);
	if(
		tokens.back().compare("png") == 0
		|| tokens.back().compare("PNG") == 0
	){
		if(currentCanvas == &canvas2D){
			PNGCanvas2D canvas(canvas2D);
			canvas.flush(filename);
		}
		else if(currentCanvas == &canvas3D){
			PNGCanvas3D canvas(canvas3D);
			canvas.flush(filename);
		}
	}
	else{
		TBTKExit(
			"Plotter2::save()",
			"Unknown file type '" << tokens.back() << "'.",
			""
		);
	}
}

};	//End namespace TBTK

#endif
