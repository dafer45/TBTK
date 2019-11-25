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

void Plotter::plot(const Property::Density &density){
	TBTKAssert(
		density.getIndexDescriptor().getFormat()
			== IndexDescriptor::Format::Ranges,
		"Plotter::plot()",
		"Invalid format. The density must be on the format"
		<< " IndexDescriptor::Format::Ranges.",
		"See the documentation for the PropertyExtractors to see how"
		<< " to extract the density on this format."
	);

	const std::vector<int> ranges = density.getRanges();
	switch(ranges.size()){
	case 1:
	{
		const std::vector<double> &data = density.getData();
		Array<double> dataArray({(unsigned int)ranges[0]});
		for(unsigned int x = 0; x < (unsigned int)ranges[0]; x++)
			dataArray[{x}] = data[x];
		plot(dataArray);
		break;
	}
	case 2:
	{
		const std::vector<double> &data = density.getData();
		Array<double> dataArray(
			{(unsigned int)ranges[0], (unsigned int)ranges[1]}
		);
		for(unsigned int x = 0; x < (unsigned int)ranges[0]; x++){
			for(
				unsigned int y = 0;
				y < (unsigned int)ranges[1];
				y++
			){
				dataArray[{x, y}] = data[ranges[1]*x + y];
			}
		}
		plot(dataArray);
		break;
	}
	default:
		TBTKExit(
			"Plotter::plot()",
			"Unsupported number of ranges. Only one- and"
			<< " two-dimensional data supported.",
			""
		);
	}
}

void Plotter::plot(const Property::Density &density, const Index &pattern){
	TBTKAssert(
		density.getIndexDescriptor().getFormat()
			== IndexDescriptor::Format::Custom,
		"Plotter::plot()",
		"Invalid format. The density must be on the format"
		<< " IndexDescriptor::Format::Custom.",
		"See the documentation for the PropertyExtractors to see how"
		<< " to extract the density on this format."
	);

	std::vector<unsigned int> wildcardPositions;
	for(unsigned int n = 0; n < pattern.getSize(); n++)
		if(pattern[n] == IDX_ALL)
			wildcardPositions.push_back(n);

	const IndexTree &indexTree
		= density.getIndexDescriptor().getIndexTree();
	std::vector<Index> indexList = indexTree.getIndexList(pattern);

	switch(wildcardPositions.size()){
	case 1:
	{
		Subindex minX = indexList[0][wildcardPositions[0]];
		Subindex maxX = indexList[0][wildcardPositions[0]];
		for(unsigned int n = 1; n < indexList.size(); n++){
			Subindex x = indexList[n][wildcardPositions[0]];
			if(minX > x)
				minX = x;
			if(maxX < x)
				maxX = x;
		}
		Array<double> X({(unsigned int)(maxX - minX + 1)});
		Array<double> Y({(unsigned int)(maxX - minX + 1)}, 0);
		for(
			unsigned int n = 0;
			n < (unsigned int)(maxX - minX + 1);
			n++
		){
			X[{n}] = minX + n;
		}
		for(unsigned int n = 0; n < indexList.size(); n++){
			Subindex x = indexList[n][wildcardPositions[0]];
			Y[{(unsigned int)(x - minX)}] = density(indexList[n]);
		}

		plot(X, Y);
		break;
	}
	case 2:
	{
		Subindex minX = indexList[0][wildcardPositions[0]];
		Subindex maxX = indexList[0][wildcardPositions[0]];
		Subindex minY = indexList[0][wildcardPositions[1]];
		Subindex maxY = indexList[0][wildcardPositions[1]];
		for(unsigned int n = 1; n < indexList.size(); n++){
			Subindex x = indexList[n][wildcardPositions[0]];
			Subindex y = indexList[n][wildcardPositions[1]];
			if(minX > x)
				minX = x;
			if(maxX < x)
				maxX = x;
			if(minY > y)
				minY = y;
			if(maxY < y)
				maxY = y;
		}
		Array<double> X({
			(unsigned int)(maxX - minX + 1),
			(unsigned int)(maxY - minY + 1)
		});
		Array<double> Y({
			(unsigned int)(maxX - minX + 1),
			(unsigned int)(maxY - minY + 1)
		});
		Array<double> Z(
			{
				(unsigned int)(maxX - minX + 1),
				(unsigned int)(maxY - minY + 1)
			},
			0
		);
		for(
			unsigned int x = 0;
			x < (unsigned int)(maxX - minX + 1);
			x++
		){
			for(
				unsigned int y = 0;
				y < (unsigned int)(maxY - minY + 1);
				y++
			){
				X[{x, y}] = minX + x;
				Y[{x, y}] = minY + y;
			}
		}
		for(unsigned int n = 0; n < indexList.size(); n++){
			Subindex x = indexList[n][wildcardPositions[0]];
			Subindex y = indexList[n][wildcardPositions[1]];
			Z[{(unsigned int)(x - minX), (unsigned int)(y - minY)}]
				= density(indexList[n]);
		}

//		plot(X, Y);
		plot(Z);
		break;
	}
	default:
		TBTKExit(
			"Plotter::plot()",
			"Unsupported number of wildcards. Only one or two"
			<< " wildcards are supported but found '"
			<< wildcardPositions.size() << "'.",
			""
		);
	}
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

	plot1D(x, y);
}

void Plotter::plot(const Property::EigenValues &eigenValues){
	for(unsigned int n = 0; n < eigenValues.getSize(); n++)
		plot1D({eigenValues(n), eigenValues(n)}, "black");
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
		plot1D(d, arguments);

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
		plot2D(d);

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
	const Array<double> &x,
	const Array<double> &y,
	const string &arguments
){
	const vector<unsigned int> &xRanges = x.getRanges();
	const vector<unsigned int> &yRanges = y.getRanges();
	TBTKAssert(
		xRanges.size() == yRanges.size(),
		"Plotter::plot()",
		"Incompatible ranges. 'x' and 'y' must have the same ranges.",
		""
	);
	for(unsigned int n = 0; n < xRanges.size(); n++){
		TBTKAssert(
			xRanges[n] == yRanges[n],
			"Plotter::plot()",
			"Incompatible ranges. 'x' and 'y' must have the same"
			<< " ranges.",
			""
		);
	}

	switch(xRanges.size()){
	case 1:
	{
		vector<double> X, Y;
		for(unsigned int n = 0; n < xRanges[0]; n++){
			X.push_back(x[{n}]);
			Y.push_back(y[{n}]);
		}
		plot1D(X, Y, arguments);

		break;
	}
	default:
		TBTKExit(
			"Plotter:plot()",
			"Array size not supported.",
			"This function can only be used with one-dimensional"
			<< " arrays."
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

void Plotter::plot1D(
	const vector<double> &x,
	const vector<double> &y,
	const string &arguments
){
	TBTKAssert(
		x.size() == y.size(),
		"Plotter::plot1D()",
		"Incompatible 'x' and 'y'. 'x' has size " << x.size()
		<< " while 'y' has size " << y.size() << ".",
		""
	);

	matplotlibcpp::plot(x, y, arguments);
}

void Plotter::plot1D(
	const vector<double> &y,
	const string &arguments
){
	matplotlibcpp::plot(y, arguments);
}

void Plotter::plot2D(const vector<vector<double>> &data){
	if(data.size() == 0)
		return;
	if(data[0].size() == 0)
		return;

	unsigned int sizeY = data[0].size();
	for(unsigned int x = 1; x < data.size(); x++){
		TBTKAssert(
			data[x].size() == sizeY,
			"Plotter:plot2D()",
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

};	//End of namespace MatPlotLib
};	//End of namespace Visualization
};	//End of namespace TBTK
