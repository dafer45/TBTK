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
 *  @file PNGCanvas3D.h
 *  @brief PNGCanvas3D
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PNG_CANVAS_3D
#define COM_DAFER45_TBTK_PNG_CANVAS_3D

#include "TBTK/Canvas3D.h"
#include "TBTK/TBTKMacros.h"

#include <string>
#include <tuple>

namespace TBTK{

class PNGCanvas3D : public Canvas3D{
public:
	/** Constructor. */
	PNGCanvas3D(const Canvas3D &canvas);

	/** Destructor. */
	virtual ~PNGCanvas3D();

	/** Flush the canvas content to the output file.
	 *
	 *  @param filename The file to write the result to. */
	void flush(const std::string &filename);
private:
	/** Convert a color to it's hexadecimal string representation. */
	static std::string convertColorToHex(
		const std::vector<unsigned char> &color
	);
};

};	//End namespace TBTK

#endif
