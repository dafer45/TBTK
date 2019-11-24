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

/** @file Plotter.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Visualization/MatPlotLib/Plotter.h"
#include "TBTK/Smooth.h"
#include "TBTK/Streams.h"

#include <sstream>

using namespace std;

namespace TBTK{
namespace Visualization{
namespace MatPlotLib{

void Plotter::setSize(unsigned int width, unsigned int height){
	matplotlibcpp::figure_size(width, height);
}

/*void Plotter::plot(
	double x,
	double y,
	const string &arguments
){
}*/

void Plotter::plot(
	const vector<double> &x,
	const vector<double> &y,
	const string &arguments
){
	TBTKAssert(
		x.size() == y.size(),
		"Plotter::plot()",
		"Incompatible 'x' and 'y'. 'x' has size " << x.size()
		<< " while 'y' has size " << y.size() << ".",
		""
	);

	matplotlibcpp::plot(x, y, arguments);
}

void Plotter::plot(
	const vector<double> &y,
	const string &arguments
){
	matplotlibcpp::plot(y, arguments);
}

void Plotter::plot(
	const Property::DOS &dos,
	double sigma,
	unsigned int windowSize
){
	vector<double> y;
	vector<double> x;
//	double dE = (dos.getUpperBound() - dos.getLowerBound())/dos.getResolution();
	double dE = dos.getDeltaE();
	for(unsigned int n = 0; n < dos.getSize(); n++){
		x.push_back(dos.getLowerBound() + n*dE);
		y.push_back(dos(n));
	}

	if(sigma != 0){
		double scaledSigma = sigma/(dos.getUpperBound() - dos.getLowerBound())*dos.getResolution();
		y = Smooth::gaussian(y, scaledSigma, windowSize);
	}

	plot(x, y);
}

/*void Plotter::plot(const Property::EigenValues &eigenValues){
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
}*/

void Plotter::plot(
	const vector<vector<double>> &data,
	const std::string &arguments
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

	vector<vector<double>> x , y;
	for(unsigned int X = 0; X < data.size(); X++){
		x.push_back(vector<double>());
		y.push_back(vector<double>());
		for(unsigned int Y = 0; Y < data[X].size(); Y++){
			x[X].push_back(X);
			y[X].push_back(Y);
		}
	}
	matplotlibcpp::plot_surface(x, y, data);
}

void Plotter::plot(
	const Array<double> &data,
	const string &arguments
){
	const vector<unsigned int> &ranges = data.getRanges();
	switch(ranges.size()){
	case 1:
	{
		vector<double> d;
		for(unsigned int n = 0; n < ranges[0]; n++)
			d.push_back(data[{n}]);
		plot(d, arguments);

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
		plot(d, arguments);

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
}

void Plotter::plot(
	const Array<double> &data,
	const Array<double> &intensities,
	const Decoration &decoration
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

};	//End of namespace MatPlotLib
};	//End of namespace Visualization
};	//End of namespace TBTK
