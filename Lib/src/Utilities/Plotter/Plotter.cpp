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

#include "../../../include/Utilities/Plotter/Plotter.h"
#include "Smooth.h"
#include "Streams.h"

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace TBTK{

Plotter::Plotter(){
	width = 600;
	height = 400;

	paddingLeft = 100;
	paddingRight = 40;
	paddingBottom = 30;
	paddingTop = 20;

	autoScaleX = true;
	autoScaleY = true;

	hold = false;
}

Plotter::~Plotter(){
}

void Plotter::plot(
	const vector<double> &axis,
	const vector<double> &data,
	const Decoration &decoration
){
	TBTKAssert(
		axis.size() == data.size(),
		"Plotter::plot()",
		"Incompatible axis and data. Axis size is " << axis.size()
			<< " while data size is " << data.size() << ".",
		""
	);

	if(!hold)
		dataStorage.clear();

	Decoration modifiedDecoration = decoration;
	if(decoration.color.size() != 3)
		modifiedDecoration.color = {0, 0, 0};

	dataStorage.push_back(make_tuple(axis, data, modifiedDecoration));

	if(autoScaleX){
		minX = get<0>(dataStorage.at(0)).at(0);
		maxX = get<0>(dataStorage.at(0)).at(0);
		for(unsigned int n = 0; n < dataStorage.size(); n++){
			const std::vector<double> axis = get<0>(dataStorage.at(n));
			const std::vector<double> data = get<1>(dataStorage.at(n));
			for(unsigned int c = 0; c < data.size(); c++){
				if(axis.at(c) < minX)
					minX = axis.at(c);
				if(axis.at(c) > maxX)
					maxX = axis.at(c);
			}
		}
	}
	if(autoScaleY){
		minY = get<1>(dataStorage.at(0)).at(0);
		maxY = get<1>(dataStorage.at(0)).at(0);
		for(unsigned int n = 0; n < dataStorage.size(); n++){
			const std::vector<double> axis = get<0>(dataStorage.at(n));
			const std::vector<double> data = get<1>(dataStorage.at(n));
			for(unsigned int c = 0; c < data.size(); c++){
				if(data.at(c) < minY)
					minY = data.at(c);
				if(data.at(c) > maxY)
					maxY = data.at(c);
			}
		}
	}

	canvas = Mat::zeros(height, width, CV_8UC3);
	rectangle(
		canvas,
		cvPoint(0, 0),
		cvPoint(width-1, height-1),
		Scalar(255, 255, 255),
		CV_FILLED,
		8,
		0
	);

	for(unsigned int n = 0; n < dataStorage.size(); n++){
		const std::vector<double> axis = get<0>(dataStorage.at(n));
		const std::vector<double> data = get<1>(dataStorage.at(n));
		Decoration &decoration = get<2>(dataStorage.at(n));
		Scalar color(
			decoration.color[2],
			decoration.color[1],
			decoration.color[0]
		);

		switch(decoration.lineStyle){
		case Decoration::LineStyle::Line:
			for(unsigned int c = 1; c < data.size(); c++){
				double x0 = axis.at(c-1);
				double y0 = data.at(c-1);
				double x1 = axis.at(c);
				double y1 = data.at(c);

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

				//Draw line
				line(
					canvas,
					getCVPoint(x0, y0),
					getCVPoint(x1, y1),
					color,
					1,
					CV_AA
				);
			}
			break;
		case Decoration::LineStyle::Point:
			for(unsigned int c = 1; c < data.size(); c++){
				double x = axis.at(c-1);
				double y = data.at(c-1);

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
					getCVPoint(x, y),
					5.,
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

	drawAxes();
}

void Plotter::plot(
	const vector<double> &data,
	const Decoration &decoration
){
	vector<double> axis;
	for(unsigned int n = 0; n < data.size(); n++)
		axis.push_back(n);

	plot(axis, data, decoration);
}

void Plotter::plot(
	const Property::DOS &dos,
	double sigma,
	unsigned int windowSize
){
	vector<double> data;
	vector<double> axis;
	double dE = (dos.getUpperBound() - dos.getLowerBound())/dos.getResolution();
	for(unsigned int n = 0; n < dos.getSize(); n++){
		axis.push_back(dos.getLowerBound() + n*dE);
		data.push_back(dos(n));
	}

	if(sigma != 0){
		double scaledSigma = sigma/(dos.getUpperBound() - dos.getLowerBound())*dos.getResolution();
		data = Smooth::gaussian(data, scaledSigma, windowSize);
	}

	plot(axis, data);
}

void Plotter::plot(
	const vector<vector<double>> &data
){
	if(data.size() == 0)
		return;
	if(data[0].size() == 0)
		return;

	unsigned int sizeY = data[0].size();
	for(unsigned int x = 1; x < data.size(); x++){
		TBTKAssert(
			data[x].size() == sizeY,
			"Plotter:plot()",
			"Incompatible array dimensions. 'data[0]' has "
				<< sizeY << " elements, while 'data[" << x
				<< "]' has " << data[x].size() << " elements.",
			""
		);
	}
	minX = 0;
	maxX = data.size()-1;
	minY = 0;
	maxY = sizeY-1;

	canvas = Mat::zeros(height, width, CV_8UC3);
	rectangle(
		canvas,
		cvPoint(0, 0),
		cvPoint(width-1, height-1),
		Scalar(255, 255, 255),
		CV_FILLED,
		8,
		0
	);

	double minValue = data[0][0];
	double maxValue = data[0][0];
	for(unsigned int x = 0; x < data.size(); x++){
		for(unsigned int y = 0; y < data[x].size(); y++){
			if(data[x][y] < minValue)
				minValue = data[x][y];
			if(data[x][y] > maxValue)
				maxValue = data[x][y];
		}
	}

	for(unsigned int x = 0; x < data.size()-1; x++){
		for(unsigned int y = 0; y < sizeY-1; y++){
			double value00 = data[x][y];
			double value01 = data[x][y+1];
			double value10 = data[x+1][y];
			double value11 = data[x+1][y+1];

			Point p00 = getCVPoint(x, y);
			Point p01 = getCVPoint(x, y+1);
			Point p10 = getCVPoint(x+1, y);
			for(int x = p00.x; x <= p10.x; x++){
				for(int y = p00.y; y >= p01.y; y--){
					double distanceX = (x-p00.x)/(double)(p10.x - p00.x);
					double distanceY = (y-p00.y)/(double)(p01.y - p00.y);
					double value0 = value00*(1 - distanceX) + value10*distanceX;
					double value1 = value01*(1 - distanceX) + value11*distanceX;
					double averagedValue = value0*(1 - distanceY) + value1*distanceY;
					canvas.at<Vec3b>(y, x)[0] = 255;
					canvas.at<Vec3b>(y, x)[1] = (255 - 255*(averagedValue - minValue)/(maxValue - minValue));
					canvas.at<Vec3b>(y, x)[2] = (255 - 255*(averagedValue - minValue)/(maxValue - minValue));
				}
			}
		}
	}

	drawAxes();
}

void Plotter::drawAxes(){
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

};	//End of namespace TBTK
