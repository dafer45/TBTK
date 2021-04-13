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

#include "TBTK/Plot/PlotCanvas.h"
#include "TBTK/Smooth.h"

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

	showColorBox = false;
	minColor = 0;
	maxColor = 0;
}

PlotCanvas::~PlotCanvas(){
}

void PlotCanvas::drawAxes(){
	//Axes
	line(
		canvas,
		getCVPoint(minX, minY),
		getCVPoint(maxX, minY),
		Scalar(0, 0, 0),
		2,
#if CV_MAJOR_VERSION > 3
		cv::LINE_AA
#else
		CV_AA
#endif
	);
	line(
		canvas,
		getCVPoint(minX, minY),
		getCVPoint(minX, maxY),
		Scalar(0, 0, 0),
		2,
#if CV_MAJOR_VERSION > 3
		cv::LINE_AA
#else
		CV_AA
#endif
	);

	//Axes values
	stringstream ss;
	ss.precision(1);
	if((abs(minX) > 1 && abs(minX) < 1000) || minX == 0)
		ss << fixed << minX;
	else
		ss << scientific << minX;
	string minXString = ss.str();
	ss.str("");
	if((abs(maxX) > 1 && abs(maxX) < 1000) || maxX == 0)
		ss << fixed << maxX;
	else
		ss << scientific << maxX;
	string maxXString = ss.str();
	ss.str("");
	if((abs(minY) > 1 && abs(minY) < 1000) || minY == 0)
		ss << fixed << minY;
	else
		ss << scientific << minY;
	string minYString = ss.str();
	ss.str("");
	if((abs(maxY) > 1 && abs(maxY) < 1000) || maxY == 0)
		ss << fixed << maxY;
	else
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
			canvas.cols - (paddingRight + showColorBox*COLOR_BOX_WINDOW_WIDTH + maxXStringSize.width/2),
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

	//Labels
	int labelXStringBaseLine;
	int labelYStringBaseLine;
	Size labelXStringSize = getTextSize(
		labelX,
		FONT_HERSHEY_SIMPLEX,
		0.5,
		1,
		&labelXStringBaseLine
	);
	Size labelYStringSize = getTextSize(
		labelY,
		FONT_HERSHEY_SIMPLEX,
		0.5,
		1,
		&labelYStringBaseLine
	);
	putText(
		canvas,
		labelX,
		Point(
			paddingLeft + (canvas.cols - paddingLeft - paddingRight - showColorBox*COLOR_BOX_WINDOW_WIDTH)/2 - labelXStringSize.width/2,
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
		labelY,
		Point(
			paddingLeft - labelYStringSize.width - 10,
			paddingBottom + (canvas.rows - paddingBottom - paddingTop)/2 - labelYStringSize.height/2
		),
		FONT_HERSHEY_SIMPLEX,
		0.5,
		Scalar(0, 0, 0),
		2,
		false
	);

	if(showColorBox)
		drawColorBox();
}

void PlotCanvas::drawColorBox(){
	stringstream ss;
	ss.precision(1);
	ss << scientific << maxColor;
	string maxColorString = ss.str();
	int maxColorStringBaseLine;
	Size maxColorStringSize = getTextSize(
		maxColorString,
		FONT_HERSHEY_SIMPLEX,
		0.5,
		1,
		&maxColorStringBaseLine
	);
	putText(
		canvas,
		maxColorString,
		Point(
			canvas.cols - COLOR_BOX_WINDOW_WIDTH/2 - maxColorStringSize.width/2,
			paddingTop + 1.5*maxColorStringSize.height
		),
		FONT_HERSHEY_SIMPLEX,
		0.5,
		Scalar(0, 0, 0),
		2,
		false
	);

	ss.str("");
	ss << scientific << minColor;
	string minColorString = ss.str();
	int minColorStringBaseLine;
	Size minColorStringSize = getTextSize(
		minColorString,
		FONT_HERSHEY_SIMPLEX,
		0.5,
		1,
		&minColorStringBaseLine
	);
	putText(
		canvas,
		minColorString,
		Point(
			canvas.cols - COLOR_BOX_WINDOW_WIDTH/2 - minColorStringSize.width/2,
			canvas.rows - (paddingBottom - 1.5*minColorStringSize.height)
		),
		FONT_HERSHEY_SIMPLEX,
		0.5,
		Scalar(0, 0, 0),
		2,
		false
	);

	double minX = canvas.cols - 3*COLOR_BOX_WINDOW_WIDTH/4;
	double maxX = canvas.cols - COLOR_BOX_WINDOW_WIDTH/4;
	double minY = paddingTop + 2.5*maxColorStringSize.height;
	double maxY = canvas.rows - paddingBottom;
	for(
		unsigned int y = minY;
		y < maxY;
		y++
	){
		for(
			unsigned int x = minX;
			x < maxX;
			x++
		){
			double value = minColor + (maxColor - minColor)*(maxY - y)/(maxY - minY);

			canvas.at<Vec3b>(y, x)[0] = 255;
			canvas.at<Vec3b>(y, x)[1] = (255 - 255*(value - minColor)/(maxColor - minColor));
			canvas.at<Vec3b>(y, x)[2] = (255 - 255*(value - minColor)/(maxColor - minColor));
		}
	}
}

};	//End of namespace Plot
};	//End of namespace TBTK

