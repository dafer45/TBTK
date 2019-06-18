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
 *  @file Canvas3D.h
 *  @brief Canvas3D
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CANVAS_3D
#define COM_DAFER45_TBTK_CANVAS_3D

#include "TBTK/Canvas.h"
#include "TBTK/TBTKMacros.h"

#include <string>
#include <tuple>

namespace TBTK{

class Canvas3D : public Canvas{
public:
	/** Constructor. */
	Canvas3D();

	/** Destructor. */
	virtual ~Canvas3D();

	/** Set bounds. */
	void setBoundsX(double minX, double maxX);

	/** Set bounds. */
	void setBoundsY(double minY, double maxY);

	/** Set bounds. */
	void setBoundsZ(double minZ, double maxZ);

	/** Set bounds. */
	void setBounds(
		double minX,
		double maxX,
		double minY,
		double maxY,
		double minZ,
		double maxZ
	);

	/** Get minimum X-value. */
	double getMinX() const;

	/** Get maximum X-value. */
	double getMaxX() const;

	/** Get minimum Y-value. */
	double getMinY() const;

	/** Get maximum Y-value. */
	double getMaxY() const;

	/** Get minimum Z-value. */
	double getMinZ() const;

	/** Get maximum Z-value. */
	double getMaxZ() const;

	/** Set x-label. */
	void setLabelX(const std::string &labelX);

	/** Set y-label. */
	void setLabelY(const std::string &labelY);

	/** Set z-label. */
	void setLabelZ(const std::string &labelZ);

	/** Get x-label. */
	const std::string& getLabelX() const;

	/** Get y-label. */
	const std::string& getLabelY() const;

	/** Get z-label. */
	const std::string& getLabelZ() const;

	/** Set hold. */
	void setHold(bool hold);

	/** Set whether to use a top view or not. */
	void setTopView(bool topView);

	/** Get whether to use a top view or not. */
	bool getTopView() const;

	/** Plot data. */
	void plot(
		const std::vector<std::vector<double>> &z,
		const std::string &title = "",
		const std::vector<unsigned char> &color = {0, 0, 0}
	);

	/** Plot data. */
	void plot(
		const std::vector<double> &x,
		const std::vector<double> &y,
		const std::vector<std::vector<double>> &values,
		const std::string &title = "",
		const std::vector<unsigned char> &color = {0, 0, 0}
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

	/*** Get the y values for the given data set.
	 *
	 *  @param dataSet The data set to get the y-values for.
	 *
	 *  @return The y-values for the given data set. */
	const std::vector<double>& getY(unsigned int dataSet) const;

	/** Get the z-values for the given data set.
	 *
	 *  @param dataSet The data set to get the z-values for.
	 *
	 *  @return The z-values for the given data set. */
	const std::vector<std::vector<double>>& getZ(
		unsigned int dataSet
	) const;

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
private:
	/** Bounds. */
	double minX, maxX, minY, maxY, minZ, maxZ;

	/** Labels. */
	std::string labelX, labelY, labelZ;

	/** Flag indicating whether or not to keep old data when plotting new
	 *  data. */
	bool hold;

	/** Flag indicating whether to use top view or not. */
	bool topView;

	/** Data. */
	std::vector<
		std::tuple<
			std::vector<double>,			//x-values
			std::vector<double>,			//y-values
			std::vector<std::vector<double>>,	//z-values
			std::string,				//title
			std::vector<unsigned char>		//color
		>
	> dataSets;
};

inline void Canvas3D::setBoundsX(
	double minX,
	double maxX
){
	TBTKAssert(
		minX <= maxX,
		"Canvas3D::setBoundsX()",
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

inline void Canvas3D::setBoundsY(
	double minY,
	double maxY
){
	TBTKAssert(
		minY <= maxY,
		"Canvas3D::setBoundsY()",
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

inline void Canvas3D::setBoundsZ(
	double minZ,
	double maxZ
){
	TBTKAssert(
		minZ <= maxZ,
		"Canvas3D::setBoundsZ()",
		"minZ has to be smaller than maxZ.",
		""
	);

	if(minZ == maxZ){
		this->minZ = minZ - 1e-10;
		this->maxZ = maxZ + 1e-10;
	}
	else{
		this->minZ = minZ;
		this->maxZ = maxZ;
	}
}

inline void Canvas3D::setBounds(
	double minX,
	double maxX,
	double minY,
	double maxY,
	double minZ,
	double maxZ
){
	setBoundsX(minX, maxX);
	setBoundsY(minY, maxY);
	setBoundsY(minZ, maxZ);
}

inline double Canvas3D::getMinX() const{
	return minX;
}

inline double Canvas3D::getMaxX() const{
	return maxX;
}

inline double Canvas3D::getMinY() const{
	return minY;
}

inline double Canvas3D::getMaxY() const{
	return maxY;
}

inline double Canvas3D::getMinZ() const{
	return minZ;
}

inline double Canvas3D::getMaxZ() const{
	return maxZ;
}

inline void Canvas3D::setLabelX(const std::string &labelX){
	this->labelX = labelX;
}

inline void Canvas3D::setLabelY(const std::string &labelY){
	this->labelY = labelY;
}

inline void Canvas3D::setLabelZ(const std::string &labelZ){
	this->labelZ = labelZ;
}

inline const std::string& Canvas3D::getLabelX() const{
	return labelX;
}

inline const std::string& Canvas3D::getLabelY() const{
	return labelY;
}

inline const std::string& Canvas3D::getLabelZ() const{
	return labelZ;
}

inline void Canvas3D::setHold(bool hold){
	this->hold = hold;
}

inline void Canvas3D::setTopView(bool topView){
	this->topView = topView;
}

inline bool Canvas3D::getTopView() const{
	return topView;
}

inline void Canvas3D::plot(
	const std::vector<std::vector<double>> &z,
	const std::string &title,
	const std::vector<unsigned char> &color
){
	TBTKAssert(
		z.size() > 0,
		"Canvas3D::plot()",
		"z is empty.",
		""
	);

	std::vector<double> x;
	for(unsigned int n = 0; n < z.size(); n++)
		x.push_back(n);

	std::vector<double> y;
	for(unsigned int n = 0; n < z[0].size(); n++)
		y.push_back(n);

	plot(x, y, z, title, color);
}

inline void Canvas3D::plot(
	const std::vector<double> &x,
	const std::vector<double> &y,
	const std::vector<std::vector<double>> &z,
	const std::string &title,
	const std::vector<unsigned char> &color
){
	TBTKAssert(
		x.size() == z.size(),
		"Canvas3D::plot()",
		"'x' and 'z' must have the same size, but z has size '"
		<< x.size() << "' while z has size'" << z.size() << "'.",
		""
	);
	for(unsigned int n = 0; n < z.size(); n++){
		TBTKAssert(
			y.size() == z[n].size(),
			"Canvas3D::plot()",
			"'y' and 'z[ << n << ]' must have the same size, but y"
			<< " has size'" << y.size() << "' while z[" << n
			<< "]' has size '" << z[n].size() << "'.",
			""
		);
	}
	TBTKAssert(
		color.size() == 3,
		"Canvas3D::plot()",
		"The color must have three components but have '"
		<< color.size() << "'.",
		""
	);

	if(!hold)
		dataSets.clear();

	dataSets.push_back(std::make_tuple(x, y, z, title, color));
}

inline void Canvas3D::clear(){
	dataSets.clear();
}

inline unsigned int Canvas3D::getNumDataSets() const{
	return dataSets.size();
}

inline const std::vector<double>& Canvas3D::getX(unsigned int dataSet) const{
	return std::get<0>(dataSets[dataSet]);
}

inline const std::vector<double>& Canvas3D::getY(unsigned int dataSet) const{
	return std::get<1>(dataSets[dataSet]);
}

inline const std::vector<std::vector<double>>& Canvas3D::getZ(
	unsigned int dataSet
) const{
	return std::get<2>(dataSets[dataSet]);
}

inline const std::string& Canvas3D::getTitle(unsigned int dataSet) const{
	return std::get<3>(dataSets[dataSet]);
}

inline const std::vector<unsigned char>& Canvas3D::getColor(
	unsigned int dataSet
) const{
	return std::get<4>(dataSets[dataSet]);
}

};	//End namespace TBTK

#endif
