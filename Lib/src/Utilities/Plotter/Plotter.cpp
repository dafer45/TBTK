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
#include "Streams.h"

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace TBTK{

Plotter::Plotter(){
	paddingLeft = 100;
	paddingRight = 40;
	paddingBottom = 30;
	paddingTop = 20;
}

Plotter::~Plotter(){
}

void Plotter::plot(const vector<double> &axis, const vector<double> &data){
	const unsigned int WIDTH = 600;
	const unsigned int HEIGHT = 400;

	double minX = axis.at(0);
	double maxX = axis.at(0);
	double minY = data.at(0);
	double maxY = data.at(0);
	for(unsigned int n = 0; n < data.size(); n++){
		if(axis.at(n) < minX)
			minX = axis.at(n);
		if(axis.at(n) > maxX)
			maxX = axis.at(n);
		if(data.at(n) < minY)
			minY = data.at(n);
		if(data.at(n) > maxY)
			maxY = data.at(n);
	}

	setBounds(minX, maxX, minY, maxY);

	canvas = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	rectangle(
		canvas,
		cvPoint(0, 0),
		cvPoint(WIDTH-1, HEIGHT-1),
		Scalar(255, 255, 255),
		CV_FILLED,
		8,
		0
	);
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

	for(unsigned int n = 1; n < data.size(); n++){
		line(
			canvas,
			getCVPoint(axis.at(n-1), data.at(n-1)),
			getCVPoint(axis.at(n), data.at(n)),
			Scalar(0, 0, 0),
			1,
			CV_AA
		);
	}
}

void Plotter::plot(const vector<double> &data){
	vector<double> axis;
	for(unsigned int n = 0; n < data.size(); n++)
		axis.push_back(n);

	plot(axis, data);
}

void Plotter::plot(const Property::DOS &dos){
	vector<double> data;
	for(unsigned int n = 0; n < dos.getSize(); n++)
		data.push_back(dos(n));

	plot(data);
}

};	//End of namespace TBTK
