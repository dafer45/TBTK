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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file Decoration.h
 *  @brief Decoration
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PLOT_DECORATION
#define COM_DAFER45_TBTK_PLOT_DECORATION

#include <vector>

namespace TBTK{
namespace Plot{

class Decoration{
public:
	/** Enum class for describingthe line style. */
	enum class LineStyle {Line, Point};

	/** Constructor. */
	Decoration();

	/** Constructor. */
	Decoration(
		const std::vector<unsigned char> &color,
		LineStyle lineStyle,
		unsigned int size = 1
	);

	/** Destructor. */
	~Decoration();

	/** Set color. */
	void setColor(const std::vector<unsigned char> &color);

	/** Get color. */
	const std::vector<unsigned char>& getColor() const;

	/** Get line style. */
	const LineStyle& getLineStyle() const;

	/** Get size. */
	unsigned int getSize() const;
private:
	/** Color. */
	std::vector<unsigned char> color;

	/** Line style. */
	LineStyle lineStyle;

	/** Width. */
	unsigned int size;
};

inline Decoration::Decoration(){
	color = {0, 0, 0};
	lineStyle = LineStyle::Point;
	size = 1;
}

inline Decoration::Decoration(
	const std::vector<unsigned char> &color,
	LineStyle lineStyle,
	unsigned int size
){
	this->color = color;
	this->lineStyle = lineStyle;
	this->size = size;
}

inline Decoration::~Decoration(){
}

inline void Decoration::setColor(const std::vector<unsigned char> &color){
	TBTKAssert(
		color.size() == 3,
		"Decoration::setColor()",
		"Color must have three components but '" << color.size() << "'"
		<< " components given.",
		""
	);
	this->color = color;
}

inline const std::vector<unsigned char>& Decoration::getColor() const{
	return color;
}

inline const Decoration::LineStyle& Decoration::getLineStyle() const{
	return lineStyle;
}

inline unsigned int Decoration::getSize() const{
	return size;
}

};	//End namespace Plot
};	//End namespace TBTK

#endif
/// @endcond
