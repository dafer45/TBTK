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
 *  @file PNGCanvas2D.h
 *  @brief PNGCanvas2D
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PNG_CANVAS_2D
#define COM_DAFER45_TBTK_PNG_CANVAS_2D

#include "TBTK/Canvas2D.h"
#include "TBTK/TBTKMacros.h"

#include <string>
#include <tuple>

namespace TBTK{

class PNGCanvas2D : public Canvas2D{
public:
	/** Constructor. */
	PNGCanvas2D(const Canvas2D &canvas);

	/** Destructor. */
	virtual ~PNGCanvas2D();

	/** Flush the canvas content to the output file.
	 *
	 *  @param filename The file to write the result to. */
	void flush(const std::string &filename);
private:
};

};	//End namespace TBTK

#endif
