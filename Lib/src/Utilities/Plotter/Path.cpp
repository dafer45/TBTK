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

#include "../../../include/Utilities/Plotter/Path.h"
#include "Plotter/Plotter.h"
#include "Smooth.h"
#include "Streams.h"

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace TBTK{
namespace Plot{

void Path::draw(
	Mat &canvas,
	const Plotter &plotter,
	double minX,
	double maxX,
	double minY,
	double maxY
){
//	const std::vector<double> axis = get<0>(dataStorage.at(n));
//	const std::vector<double> data = get<1>(dataStorage.at(n));
//	Decoration &decoration = get<2>(dataStorage.at(n));
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

			//Clip lines
			if(x1 < x0){
				double temp = x0;
				x0 = x1;
				x1 = temp;
				temp = y0;
				y0 = y1;
				y1 = temp;
			}
			if(x0 < minX && x1 < minX)
				continue;
			if(x0 > maxX && x1 > maxX)
				continue;
			if(x0 < minX){
				if(x1 - x0 != 0)
					y0 += (minX - x0)*(y1 - y0)/(x1 - x0);
				x0 = minX;
			}
			if(x1 > maxX){
				if(x1 - x0 != 0)
					y1 -= (x1 - maxX)*(y1 - y0)/(x1 - x0);
				x1 = maxX;
			}
			if(y0 < minY && y1 < minY)
				continue;
			if(y0 > maxY && y1 > maxY)
				continue;
			if(y0 < minY){
				if(y1 - y0 != 0)
					x0 += (minY - y0)*(x1 - x0)/(y1 - y0);
				y0 = minY;
			}
			if(y1 > maxY){
				if(y1 - y0 != 0)
					x1 -= (y1 - maxY)*(x1 - x0)/(y1 - y0);
				y1 = maxY;
			}

			Point p1 = plotter.getCVPoint(x0, y0);
			Point p2 = plotter.getCVPoint(x1, y1);

			Streams::out << x0 << "\t" << y0 << "\t" << x1 << "\t" << y1 << "\n";
			Streams::out << color[0] << "\t" << color[1] << "\t" << color[2] << "\t" << decoration.getSize() << "\n";
			Streams::out << p1.x << "\t" << p1.y << "\t" << p2.x << "\t" << p2.y << "\n";
			//Draw line
			line(
				canvas,
				plotter.getCVPoint(x0, y0),
				plotter.getCVPoint(x1, y1),
				color,
				decoration.getSize(),
				CV_AA
			);
		}
		break;
	case Decoration::LineStyle::Point:
		for(unsigned int c = 1; c < coordinates.size(); c++){
			double x = coordinates[c-1].x;
			double y = coordinates[c-1].y;

			//Clip points
			if(x < minX)
				continue;
			if(x > maxX)
				continue;
			if(y < minY)
				continue;
			if(y > maxY)
				continue;

			//Draw line
			circle(
				canvas,
				plotter.getCVPoint(x, y),
				decoration.getSize(),
				color,
				-1,
				CV_AA
			);
		}
		break;
	default:
		TBTKExit(
			"Plotter::plot()",
			"Unknown mode.",
			"This should never happen, contact the developer."
		);
	}
}

};	//End of namespace Plot
};	//End of namespace TBTK
