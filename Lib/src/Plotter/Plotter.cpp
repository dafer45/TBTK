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

#include "TBTK/Plot/Plotter.h"
#include "TBTK/Smooth.h"
#include "TBTK/Streams.h"

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace TBTK{
namespace Plot{

Plotter::Plotter(){
	autoScaleX = true;
	autoScaleY = true;
	hold = false;
}

Plotter::~Plotter(){
	clearDataStorage();
}

void Plotter::plot(
	double x,
	double y,
	const Decoration &decoration
){
	if(!hold)
		clearDataStorage();

	Decoration modifiedDecoration = decoration;
	if(decoration.getColor().size() != 3)
		modifiedDecoration.setColor({0, 0, 0});

	Point *point = new Point();
	point->setDecoration(modifiedDecoration);
	point->setCoordinate({x, y});
	dataStorage.push_back(point);
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
		clearDataStorage();

	Decoration modifiedDecoration = decoration;
	if(decoration.getColor().size() != 3)
		modifiedDecoration.setColor({0, 0, 0});

	Path *path = new Path();
	path->setDecoration(modifiedDecoration);
	for(unsigned int n = 0; n < axis.size(); n++)
		path->add({axis[n], data[n]});
	dataStorage.push_back(path);
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

void Plotter::plot(const Property::EigenValues &eigenValues){
	vector<double> data;
	for(unsigned int n = 0; n < eigenValues.getSize(); n++)
		data.push_back(eigenValues(n));

	plot(
		data,
		Decoration(
			{0, 0, 0},
			Decoration::LineStyle::Point
		)
	);
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
	canvas.setBounds(0, data.size()-1, 0, sizeY-1);

	clearDataStorage();
	canvas.clear();

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

	bool tempShowColorBox = canvas.getShowColorBox();
	canvas.setShowColorBox(true);
	canvas.setBoundsColor(minValue, maxValue);

	for(unsigned int x = 0; x < data.size()-1; x++){
		for(unsigned int y = 0; y < sizeY-1; y++){
			double value00 = data[x][y];
			double value01 = data[x][y+1];
			double value10 = data[x+1][y];
			double value11 = data[x+1][y+1];

			cv::Point p00 = canvas.getCVPoint(x, y);
			cv::Point p01 = canvas.getCVPoint(x, y+1);
			cv::Point p10 = canvas.getCVPoint(x+1, y);
			for(int x = p00.x; x <= p10.x; x++){
				for(int y = p00.y; y >= p01.y; y--){
					double distanceX = (x-p00.x)/(double)(p10.x - p00.x);
					double distanceY = (y-p00.y)/(double)(p01.y - p00.y);
					double value0 = value00*(1 - distanceX) + value10*distanceX;
					double value1 = value01*(1 - distanceX) + value11*distanceX;
					double averagedValue = value0*(1 - distanceY) + value1*distanceY;
					canvas.setPixel(
						x,
						y,
						(255 - 255*(averagedValue - minValue)/(maxValue - minValue)),
						(255 - 255*(averagedValue - minValue)/(maxValue - minValue)),
						255
					);
				}
			}
		}
	}

	canvas.drawAxes();

	canvas.setShowColorBox(tempShowColorBox);
}

void Plotter::plot(
	const Array<double> &data,
	const Decoration &decoration
){
	const vector<unsigned int> &ranges = data.getRanges();
	switch(ranges.size()){
	case 1:
	{
		vector<double> d;
		for(unsigned int n = 0; n < ranges[0]; n++)
			d.push_back(data[{n}]);
		plot(d, decoration);

		break;
	}
	case 2:
	{
		vector<vector<double>> d;
		for(unsigned int m = 0; m < ranges[0]; m++){
			d.push_back(vector<double>());
			for(unsigned int n = 0; n < ranges[1]; n++)
				d[m].push_back(data[{m, n}]);
		}
		plot(d);

		break;
	}
	default:
		TBTKExit(
			"Plotter:plot()",
			"Array size not supported.",
			"Only arrays with one or two dimensions can be"
			<< " plotter."
		);
	}
}

void Plotter::plot(
	const vector<vector<double>> &data,
	const vector<vector<double>> &intensities,
	const Decoration &decoration
){
	TBTKAssert(
		data.size() == intensities.size(),
		"Plotter::plot()",
		"The dimensions of 'data' and 'intensities' do not agree."
		<< " 'data' has size '" << data.size() << "', while"
		<< " 'intensities' have size '" << intensities.size() << "'.",
		""
	);

	bool tempHold = hold;
	if(!hold){
		clearDataStorage();
		hold = true;
	}

	bool isInitialized = false;
	double min = 0;
	double max = 1;
	for(unsigned int n = 0; n < data.size(); n++){
		TBTKAssert(
			data[n].size() == intensities[n].size(),
			"Plotter::plot()",
			"The dimensions of 'data[" << n << "]' and"
			<< " 'intensities[" << n << "]' do not agree. 'data["
			<< n << "]'" << " has size '" << data[n].size() << "',"
			<< " while 'intensities[" << n << "]' has size '"
			<< intensities.size() << "'.",
			""
		);

		for(unsigned int c = 0; c < data[n].size(); c++){
			if(!isInitialized){
				min = intensities[n][c];
				max = intensities[n][c];
				isInitialized = true;
			}

			if(intensities[n][c] < min)
				min = intensities[n][c];
			if(intensities[n][c] > max)
				max = intensities[n][c];
		}
	}
	if(min == max)
		min = max -1;

	for(unsigned int n = 0; n < data.size(); n++){
		for(unsigned int c = 0; c < data[n].size(); c++){
			plot(
				n,
				data[n][c],
				Decoration(
					{
						(unsigned char)(255 - 255*(intensities[n][c] - min)/(max - min)),
						0,
						(unsigned char)(255*(intensities[n][c] - min)/(max-min))
					},
					Decoration::LineStyle::Point,
					decoration.getSize()
				)
			);
		}
	}

	hold = tempHold;
}

void Plotter::plot(
	const Array<double> &data,
	const Array<double> &intensities,
	const Decoration &decoration
){
	const vector<unsigned int> &ranges = data.getRanges();
	switch(ranges.size()){
/*	case 1:
	{
		vector<double> d;
		vector<double> i;
		for(unsigned int n = 0; n < ranges[0]; n++){
			d.push_back(data[{n}]);
			i.push_back(intensities[{n}]);
		}
		plot(d, i, decoration);

		break;
	}*/
	case 2:
	{
		vector<vector<double>> d;
		vector<vector<double>> i;
		for(unsigned int m = 0; m < ranges[0]; m++){
			d.push_back(vector<double>());
			i.push_back(vector<double>());
			for(unsigned int n = 0; n < ranges[1]; n++){
				d[m].push_back(data[{m, n}]);
				i[m].push_back(intensities[{m, n}]);
			}
		}
		plot(d, i, decoration);

		break;
	}
	default:
		TBTKExit(
			"Plotter:plot()",
			"Array size not supported.",
			"Only arrays with one or two dimensions can be"
			<< " plotter."
		);
	}
}

void Plotter::drawDataStorage(){
	if(dataStorage.size() == 0)
		return;

	if(autoScaleX){
		double minX = dataStorage[0]->getMinX();
		double maxX = dataStorage[0]->getMaxX();
		for(unsigned int n = 1; n < dataStorage.size(); n++){
			double min = dataStorage[n]->getMinX();
			double max = dataStorage[n]->getMaxX();
			if(min < minX)
				minX = min;
			if(max > maxX)
				maxX = max;
		}
		canvas.setBoundsX(minX, maxX);
	}
	if(autoScaleY){
		double minY = dataStorage[0]->getMinY();
		double maxY = dataStorage[0]->getMaxY();
		for(unsigned int n = 1; n < dataStorage.size(); n++){
			double min = dataStorage[n]->getMinY();
			double max = dataStorage[n]->getMaxY();
			if(min < minY)
				minY = min;
			if(max > maxY)
				maxY = max;
		}
		canvas.setBoundsY(minY, maxY);
	}

	canvas.clear();

	for(unsigned int n = 0; n < dataStorage.size(); n++)
		dataStorage[n]->draw(canvas);
}

};	//End of namespace Plot
};	//End of namespace TBTK
