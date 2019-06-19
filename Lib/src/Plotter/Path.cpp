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

/** @file Path.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Plot/Path.h"
#include "TBTK/Smooth.h"
#include "TBTK/Streams.h"

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace TBTK{
namespace Plot{

void Path::draw(PlotCanvas &canvas){
	Scalar color(
		decoration.getColor()[2],
		decoration.getColor()[1],
		decoration.getColor()[0]
	);

	switch(decoration.getLineStyle()){
	case Decoration::LineStyle::Line:
		for(unsigned int c = 1; c < coordinates.size(); c++){
			double x0 = coordinates[c-1].x;
			double y0 = coordinates[c-1].y;
			double x1 = coordinates[c].x;
			double y1 = coordinates[c].y;

			//Draw line
			canvas.drawLine(
				x0,
				y0,
				x1,
				y1,
				decoration.getColor(),
				decoration.getSize()
			);
		}
		break;
	case Decoration::LineStyle::Point:
		for(unsigned int c = 1; c < coordinates.size(); c++){
			double x = coordinates[c-1].x;
			double y = coordinates[c-1].y;

			//Draw line
			canvas.drawCircle(
				x,
				y,
				decoration.getSize(),
				decoration.getColor()
			);
		}
		break;
	default:
		TBTKExit(
			"Path::draw()",
			"Unknown mode.",
			"This should never happen, contact the developer."
		);
	}
}

};	//End of namespace Plot
};	//End of namespace TBTK
