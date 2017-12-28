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

/** @file Plotter.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../../../include/Utilities/Plotter/PlotCanvas.h"
#include "Smooth.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace TBTK{
namespace Plot{

PlotCanvas::PlotCanvas(){
	width = 600;
	height = 400;

	paddingLeft = 100;
	paddingRight = 40;
	paddingBottom = 30;
	paddingTop = 20;
}

PlotCanvas::~PlotCanvas(){
}

void PlotCanvas::drawAxes(){
	line(
		canvas,
		getCVPoint(minX, minY),
		getCVPoint(maxX, minY),
		Scalar(0, 0, 0),
		2,
		CV_AA
	);
	line(
		canvas,
		getCVPoint(minX, minY),
		getCVPoint(minX, maxY),
		Scalar(0, 0, 0),
		2,
		CV_AA
	);
	stringstream ss;
	ss.precision(1);
	ss << scientific << minX;
	string minXString = ss.str();
	ss.str("");
	ss << scientific << maxX;
	string maxXString = ss.str();
	ss.str("");
	ss << scientific << minY;
	string minYString = ss.str();
	ss.str("");
	ss << scientific << maxY;
	string maxYString = ss.str();
	int minXStringBaseLine;
	int maxXStringBaseLine;
	int minYStringBaseLine;
	int maxYStringBaseLine;
	Size minXStringSize = getTextSize(
		minXString,
		FONT_HERSHEY_SIMPLEX,
		0.5,
		1,
		&minXStringBaseLine
	);
	Size maxXStringSize = getTextSize(
		maxXString,
		FONT_HERSHEY_SIMPLEX,
		0.5,
		1,
		&maxXStringBaseLine
	);
	Size minYStringSize = getTextSize(
		minYString,
		FONT_HERSHEY_SIMPLEX,
		0.5,
		1,
		&minYStringBaseLine
	);
	Size maxYStringSize = getTextSize(
		maxYString,
		FONT_HERSHEY_SIMPLEX,
		0.5,
		1,
		&maxYStringBaseLine
	);

	putText(
		canvas,
		minXString,
		Point(
			paddingLeft - minXStringSize.width/2,
			canvas.rows - (paddingBottom - 1.5*minXStringSize.height)
		),
		FONT_HERSHEY_SIMPLEX,
		0.5,
		Scalar(0, 0, 0),
		2,
		false
	);
	putText(
		canvas,
		maxXString,
		Point(
			canvas.cols - (paddingRight + maxXStringSize.width/2),
			canvas.rows - (paddingBottom - 1.5*maxXStringSize.height)
		),
		FONT_HERSHEY_SIMPLEX,
		0.5,
		Scalar(0, 0, 0),
		2,
		false
	);
	putText(
		canvas,
		minYString,
		Point(
			paddingLeft - minYStringSize.width - 10,
			canvas.rows - paddingBottom
		),
		FONT_HERSHEY_SIMPLEX,
		0.5,
		Scalar(0, 0, 0),
		2,
		false
	);
	putText(
		canvas,
		maxYString,
		Point(
			paddingLeft - maxYStringSize.width - 10,
			paddingTop + maxYStringSize.height
		),
		FONT_HERSHEY_SIMPLEX,
		0.5,
		Scalar(0, 0, 0),
		2,
		false
	);
}

};	//End of namespace Plot
};	//End of namespace TBTK
