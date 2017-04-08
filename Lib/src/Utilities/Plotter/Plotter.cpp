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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace TBTK{

Plotter::Plotter(){
}

Plotter::~Plotter(){
}

void Plotter::plot(vector<double> data){
	const unsigned int WIDTH = 600;
	const unsigned int HEIGHT = 400;

	double minX = 0;
	double maxX = data.size();
	double minY = data.at(0);
	double maxY = data.at(0);
	for(unsigned int n = 0; n < data.size(); n++){
		if(data.at(n) < minY)
			minY = data.at(n);
		if(data.at(n) > maxY)
			maxY = data.at(n);
	}

	setBounds(minX, maxX, minY, maxY);
	setPadding(20, 20);

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
		1,
		CV_AA
	);
	line(
		canvas,
		getCVPoint(minX, minY),
		getCVPoint(minX, maxY),
		Scalar(0, 0, 0),
		1,
		CV_AA
	);

	for(unsigned int n = 1; n < data.size(); n++){
		line(
			canvas,
			getCVPoint(n-1, data.at(n-1)),
			getCVPoint(n, data.at(n)),
			Scalar(0, 0, 0),
			1,
			CV_AA
		);
	}

	imwrite("figures/plot.png", canvas);
}

void Plotter::plot(const Property::DOS &dos){
	vector<double> data;
	for(unsigned int n = 0; n < dos.getSize(); n++)
		data.push_back(dos(n));

	plot(data);
}

};	//End of namespace TBTK
