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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file Canvas2D.h
 *  @brief Canvas2D
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CANVAS_2D
#define COM_DAFER45_TBTK_CANVAS_2D

#include "TBTK/Canvas.h"
#include "TBTK/TBTKMacros.h"

#include <string>
#include <tuple>

namespace TBTK{

class Canvas2D : public Canvas{
public:
	/** Constructor. */
	Canvas2D();

	/** Destructor. */
	virtual ~Canvas2D();

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

	/** Set x-label. */
	void setLabelX(const std::string &labelX);

	/** Set y-label. */
	void setLabelY(const std::string &labelY);

	/** Get x-label. */
	const std::string& getLabelX() const;

	/** Get y-label. */
	const std::string& getLabelY() const;

	/** Set hold. */
	void setHold(bool hold);

	/** Plot data. */
	void plot(
		const std::vector<double> &y,
		const std::string &title = "",
		const std::vector<unsigned char> &color = {0, 0, 0},
		unsigned int size = 1
	);

	/** Plot data. */
	void plot(
		const std::vector<double> &x,
		const std::vector<double> &y,
		const std::string &title = "",
		const std::vector<unsigned char> &color = {0, 0, 0},
		unsigned int size = 1
	);

	/** Clear plot. */
	virtual void clear();
protected:
	/** Get the number of contained data sets.
	 *
	 * @return The number of contained data sets. */
	unsigned int getNumDataSets() const;

	/*** Get the x values for the given data set.
	 *
	 *  @param dataSet The data set to get the x-values for.
	 *
	 *  @return The x-values for the given data set. */
	const std::vector<double>& getX(unsigned int dataSet) const;

	/** Get the y-values for the given data set.
	 *
	 *  @param dataSet The data set to get the y-values for.
	 *
	 *  @return The y-values for the given data set. */
	const std::vector<double>& getY(unsigned int dataSet) const;

	/*** Get the title for the given data set.
	 *
	 *  @param dataSet The data set to get the title for.
	 *
	 *  @return The title for the given data set. */
	const std::string& getTitle(unsigned int dataSet) const;

	/** Get the color for the given data set.
	 *
	 *  @param dataSet The data set to get the color for.
	 *
	 *  @return The color for the given data set. */
	const std::vector<unsigned char>& getColor(unsigned int dataSet) const;

	/** Get the size for the given data set.
	 *
	 *  @param dataSet The data set to get the size for.
	 *
	 *  @return The size for the data set. */
	unsigned int getSize(unsigned int dataSet) const;
private:
	/** Bounds. */
	double minX, maxX, minY, maxY;

	/** Labels. */
	std::string labelX, labelY;

	/** Flag indicating whether or not to keep old data when plotting new
	 *  data. */
	bool hold;

	/** Data. */
	std::vector<
		std::tuple<
			std::vector<double>,		//x-values
			std::vector<double>,		//y-values
			std::string,			//title
			std::vector<unsigned char>,	//color
			unsigned int			//size
		>
	> dataSets;
};

inline void Canvas2D::setBoundsX(
	double minX,
	double maxX
){
	TBTKAssert(
		minX <= maxX,
		"Canvas2D::setBoundsX()",
		"minX has to be smaller than maxX.",
		""
	);

	if(minX == maxX){
		this->minX = minX - 1e-10;
		this->maxX = maxX + 1e-10;
	}
	else{
		this->minX = minX;
		this->maxX = maxX;
	}
}

inline void Canvas2D::setBoundsY(
	double minY,
	double maxY
){
	TBTKAssert(
		minY <= maxY,
		"Canvas2D::setBoundsY()",
		"minY has to be smaller than maxY.",
		""
	);

	if(minY == maxY){
		this->minY = minY - 1e-10;
		this->maxY = maxY + 1e-10;
	}
	else{
		this->minY = minY;
		this->maxY = maxY;
	}
}

inline void Canvas2D::setBounds(
	double minX,
	double maxX,
	double minY,
	double maxY
){
	setBoundsX(minX, maxX);
	setBoundsY(minY, maxY);
}

inline double Canvas2D::getMinX() const{
	return minX;
}

inline double Canvas2D::getMaxX() const{
	return maxX;
}

inline double Canvas2D::getMinY() const{
	return minY;
}

inline double Canvas2D::getMaxY() const{
	return maxY;
}

inline void Canvas2D::setLabelX(const std::string &labelX){
	this->labelX = labelX;
}

inline void Canvas2D::setLabelY(const std::string &labelY){
	this->labelY = labelY;
}

inline const std::string& Canvas2D::getLabelX() const{
	return labelX;
}

inline const std::string& Canvas2D::getLabelY() const{
	return labelY;
}

inline void Canvas2D::setHold(bool hold){
	this->hold = hold;
}

inline void Canvas2D::plot(
	const std::vector<double> &y,
	const std::string &title,
	const std::vector<unsigned char> &color,
	unsigned int size
){
	std::vector<double> x;
	for(unsigned int n = 0; n < y.size(); n++)
		x.push_back(n);

	plot(x, y, title, color, size);
}

inline void Canvas2D::plot(
	const std::vector<double> &x,
	const std::vector<double> &y,
	const std::string &title,
	const std::vector<unsigned char> &color,
	unsigned int size
){
	TBTKAssert(
		x.size() == y.size(),
		"Canvas2D::plot()",
		"'x' and 'y' must have the same size, but x has size '"
		<< x.size() << "' while y has size'" << y.size() << "'.",
		""
	);
	TBTKAssert(
		color.size() == 3,
		"Canvas2D::plot()",
		"The color must have three components but have '"
		<< color.size() << "'.",
		""
	);

	if(!hold)
		dataSets.clear();

	dataSets.push_back(std::make_tuple(x, y, title, color, size));
}

inline void Canvas2D::clear(){
	dataSets.clear();
}

inline unsigned int Canvas2D::getNumDataSets() const{
	return dataSets.size();
}

inline const std::vector<double>& Canvas2D::getX(unsigned int dataSet) const{
	return std::get<0>(dataSets[dataSet]);
}

inline const std::vector<double>& Canvas2D::getY(unsigned int dataSet) const{
	return std::get<1>(dataSets[dataSet]);
}

inline const std::string& Canvas2D::getTitle(unsigned int dataSet) const{
	return std::get<2>(dataSets[dataSet]);
}

inline const std::vector<unsigned char>& Canvas2D::getColor(
	unsigned int dataSet
) const{
	return std::get<3>(dataSets[dataSet]);
}

inline unsigned int Canvas2D::getSize(unsigned int dataSet) const{
	return std::get<4>(dataSets[dataSet]);
}

};	//End namespace TBTK

#endif
/// @endcond
