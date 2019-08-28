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

/** @file Plotter2.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Plotter2.h"
#include "TBTK/Smooth.h"
#include "TBTK/Streams.h"

#include <sstream>

using namespace std;

namespace TBTK{

Plotter2::Plotter2(){
	autoScaleX = true;
	autoScaleY = true;
	currentCanvas = nullptr;
}

Plotter2::~Plotter2(){
}

void Plotter2::plot(
	const vector<double> &x,
	const vector<double> &y,
	const string &title,
	const vector<unsigned char> &color,
	unsigned int size
){
	setCurrentCanvas(canvas2D);

	canvas2D.plot(x, y, title, color, size);
}

void Plotter2::plot(
	const vector<double> &y,
	const string &title,
	const vector<unsigned char> &color,
	unsigned int size
){
	setCurrentCanvas(canvas2D);
	canvas2D.plot(y, title, color, size);
}

void Plotter2::plot(
	const Property::DOS &dos,
	double sigma,
	unsigned int windowSize
){
	vector<double> x;
	vector<double> y;
	double dE = (dos.getUpperBound() - dos.getLowerBound())/dos.getResolution();
	for(unsigned int n = 0; n < dos.getSize(); n++){
		x.push_back(dos.getLowerBound() + n*dE);
		y.push_back(dos(n));
	}

	if(sigma != 0){
		double scaledSigma = sigma/(dos.getUpperBound() - dos.getLowerBound())*dos.getResolution();
		y = Smooth::gaussian(y, scaledSigma, windowSize);
	}

	setCurrentCanvas(canvas2D);
	canvas2D.plot(x, y);
}

void Plotter2::plot(
	const Property::LDOS &ldos,
	double sigma,
	unsigned int windowSize
){
	TBTKAssert(
		ldos.getIndexDescriptor().getFormat()
			== IndexDescriptor::Format::Custom,
		"Plotter2::plot()",
		"Format not supported. The LDOS needs to be of the format"
		<< " IndexDescriptor::Format::Custom.",
		"Use the syntax PropertyExtractor::calculateLDOS({{...}}) to"
		" calculate the LDOS on the custom format."
	);
	TBTKAssert(
		ldos.getIndexDescriptor().getSize() == 1,
		"Plotter2::plot()",
		"The LDOS must contain data for exactly one Index, but this"
		<< " LDOS contains data for '"
		<< ldos.getIndexDescriptor().getSize() << "' Indices."
,		""
	);

	vector<double> x;
	vector<double> y;
	double dE = (ldos.getUpperBound() - ldos.getLowerBound())/ldos.getResolution();
	for(unsigned int n = 0; n < ldos.getSize(); n++){
		x.push_back(ldos.getLowerBound() + n*dE);
		y.push_back(ldos(n));
	}

	if(sigma != 0){
		double scaledSigma = sigma/(ldos.getUpperBound() - ldos.getLowerBound())*ldos.getResolution();
		y = Smooth::gaussian(y, scaledSigma, windowSize);
	}

	setCurrentCanvas(canvas2D);
	canvas2D.plot(x, y);
}

void Plotter2::plot(const Property::EigenValues &eigenValues){
	vector<double> y;
	for(unsigned int n = 0; n < eigenValues.getSize(); n++)
		y.push_back(eigenValues(n));

	setCurrentCanvas(canvas2D);
	canvas2D.plot(y);
}

void Plotter2::plot(const vector<vector<double>> &z, const string &title){
	setCurrentCanvas(canvas3D);
	canvas3D.plot(z, title);
}

void Plotter2::plot(
	const Array<double> &data,
	const string &title,
	const vector<unsigned char> &color,
	unsigned int size
){
	const vector<unsigned int> &ranges = data.getRanges();
	switch(ranges.size()){
	case 1:
	{
		vector<double> d;
		for(unsigned int n = 0; n < ranges[0]; n++)
			d.push_back(data[{n}]);
		plot(d, title, color, size);

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

/*void Plotter::plot(
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
}*/

/*void Plotter::plot(
	const Array<double> &data,
	const Array<double> &intensities
){
	const vector<unsigned int> &ranges = data.getRanges();
	switch(ranges.size()){
//	case 1:
//	{
//		vector<double> d;
//		vector<double> i;
//		for(unsigned int n = 0; n < ranges[0]; n++){
//			d.push_back(data[{n}]);
//			i.push_back(intensities[{n}]);
//		}
//		plot(d, i, decoration);
//
//		break;
//	}
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
}*/

void Plotter2::setCurrentCanvas(Canvas &canvas){
	if(currentCanvas != &canvas){
		if(currentCanvas != nullptr)
			currentCanvas->clear();

		currentCanvas = &canvas;
	}
}

};	//End of namespace TBTK
