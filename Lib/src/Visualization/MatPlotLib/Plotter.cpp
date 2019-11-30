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

#include "TBTK/AnnotatedArray.h"
#include "TBTK/PropertyConverter.h"
#include "TBTK/Smooth.h"
#include "TBTK/Streams.h"
#include "TBTK/Visualization/MatPlotLib/Plotter.h"

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

void Plotter::plot(const Property::Density &density, const Argument &argument){
	AnnotatedArray<double, Subindex> annotatedArray
		= PropertyConverter::convert(density);

	setTitle("Density", false);
	switch(annotatedArray.getRanges().size()){
	case 1:
		setLabelX("x", false);
		setLabelY("Density", false);
		break;
	default:
		setLabelX("x", false);
		setLabelY("y", false);
		break;
	}

	plot(annotatedArray, argument);
}

void Plotter::plot(
	const Index &pattern,
	const Property::Density &density,
	const Argument &argument
){
	AnnotatedArray<double, Subindex> annotatedArray
		= PropertyConverter::convert(density, pattern);

	setTitle("Density", false);
	switch(annotatedArray.getRanges().size()){
	case 1:
		setLabelX("x", false);
		setLabelY("Density", false);
		break;
	default:
		setLabelX("x", false);
		setLabelY("y", false);
		break;
	}

	plot(annotatedArray, argument);
}

void Plotter::plot(
	const Property::DOS &dos,
	double sigma,
	unsigned int windowSize,
	const Argument &argument
){
	setTitle("DOS", false);
	setLabelX("Energy", false);
	setLabelY("DOS", false);

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

	plot1D(x, y, argument);
}

void Plotter::plot(
	const Property::EigenValues &eigenValues,
	const Argument &argument
){
	setTitle("Eigenvalues", false);
	setLabelX("", false);
	setLabelY("Energy", false);
	for(unsigned int n = 0; n < eigenValues.getSize(); n++)
		plot1D({eigenValues(n), eigenValues(n)}, argument);
}

void Plotter::plot(const Property::LDOS &ldos, const Argument &argument){
	AnnotatedArray<double, Subindex> annotatedArray
		= PropertyConverter::convert(ldos);

	setTitle("LDOS", false);
	switch(annotatedArray.getRanges().size()){
	case 2:
		setLabelX("x", false);
		setLabelY("Energy", false);
		break;
	default:
		break;
	}

	plot(annotatedArray, argument);
}

void Plotter::plot(
	const Index &pattern,
	const Property::LDOS &ldos,
	const Argument &argument
){
	AnnotatedArray<double, Subindex> annotatedArray
		= PropertyConverter::convert(ldos, pattern);

	setTitle("LDOS", false);
	switch(annotatedArray.getRanges().size()){
	case 2:
		setLabelX("x", false);
		setLabelY("Energy", false);
		break;
	default:
		break;
	}

	plot(annotatedArray, argument);
}

void Plotter::plot(
	const AnnotatedArray<double, double> &data,
	const Argument &argument
){
	const vector<unsigned int> &ranges = data.getRanges();
	const vector<vector<double>> &axes = data.getAxes();
	switch(ranges.size()){
	case 1:
	{
		Array<double> X = Array<double>::create(ranges);
		Array<double> Y = Array<double>::create(ranges);
		for(unsigned int x = 0; x < ranges[0]; x++){
			X[{x}] = axes[0][x];
			Y[{x}] = data[x];
		}
		plot(X, Y, argument);
		break;
	}
	case 2:
	{
		Array<double> X = Array<double>::create(ranges);
		Array<double> Y = Array<double>::create(ranges);
		Array<double> Z = Array<double>::create(ranges);
		for(unsigned int x = 0; x < ranges[0]; x++){
			for(unsigned int y = 0; y < ranges[1]; y++){
				X[{x, y}] = axes[0][x];
				Y[{x, y}] = axes[1][y];
				Z[{x, y}] = data[{x, y}];
			}
		}
		plot(X, Y, Z, argument);
		break;
	}
	default:
		TBTKExit(
			"Plotter:plot()",
			"Array size not supported.",
			"Only arrays with one and two dimensions can be"
			<< " plotted."
		);
	}
}

void Plotter::plot(
	const AnnotatedArray<double, Subindex> &data,
	const Argument &argument
){
	const vector<vector<Subindex>> &axes = data.getAxes();;
	vector<vector<double>> newAxes(axes.size());
	for(unsigned int n = 0; n < axes.size(); n++)
		for(unsigned int c = 0; c < axes[n].size(); c++)
			newAxes[n].push_back(axes[n][c]);

	plot(AnnotatedArray<double, double>(data, newAxes), argument);
}

void Plotter::plot(const Array<double> &data, const Argument &argument){
	const vector<unsigned int> &ranges = data.getRanges();
	switch(ranges.size()){
	case 1:
	{
		vector<double> d;
		for(unsigned int n = 0; n < ranges[0]; n++)
			d.push_back(data[{n}]);
		plot1D(d, argument);

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
		plot2D(d, argument);

		break;
	}
	default:
		TBTKExit(
			"Plotter:plot()",
			"Array size not supported.",
			"Only arrays with one or two dimensions can be"
			<< " plotted."
		);
	}
}

void Plotter::plot(
	const Array<double> &x,
	const Array<double> &y,
	const Argument &argument
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
		plot1D(X, Y, argument);

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

void Plotter::plot(
	const Array<double> &x,
	const Array<double> &y,
	const Array<double> &z,
	const Argument &argument
){
	const vector<unsigned int> &xRanges = x.getRanges();
	const vector<unsigned int> &yRanges = y.getRanges();
	const vector<unsigned int> &zRanges = z.getRanges();
	TBTKAssert(
		xRanges.size() == yRanges.size()
		&& xRanges.size() == zRanges.size(),
		"Plotter::plot()",
		"Incompatible ranges. 'x', 'y', and 'z' must have the same"
		<< " ranges.",
		""
	);
	for(unsigned int n = 0; n < xRanges.size(); n++){
		if(xRanges[n] == 0)
			return;

		TBTKAssert(
			xRanges[n] == yRanges[n] && xRanges[n] == zRanges[n],
			"Plotter::plot()",
			"Incompatible ranges. 'x', 'y', and 'z' must have the"
			<< " same ranges.",
			""
		);
	}

	switch(xRanges.size()){
	case 2:
	{
		vector<vector<double>> X(
			xRanges[0],
			vector<double>(xRanges[1])
		);
		vector<vector<double>> Y(
			xRanges[0],
			vector<double>(xRanges[1])
		);
		vector<vector<double>> Z(
			xRanges[0],
			vector<double>(xRanges[1])
		);
		for(unsigned int i = 0; i < xRanges[0]; i++){
			for(unsigned int j = 0; j < xRanges[1]; j++){
				X[i][j] = x[{i, j}];
				Y[i][j] = y[{i, j}];
				Z[i][j] = z[{i, j}];
			}
		}
		plot2D(X, Y, Z, argument);
		break;
	}
	default:
		TBTKExit(
			"Plotter:plot()",
			"Array size not supported.",
			"This function can only be used with two-dimensional"
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
	const Argument &argument
){
	TBTKAssert(
		x.size() == y.size(),
		"Plotter::plot1D()",
		"Incompatible 'x' and 'y'. 'x' has size " << x.size()
		<< " while 'y' has size " << y.size() << ".",
		""
	);

	if(argument.getArgumentMap().size() == 0)
		matplotlibcpp::plot(x, y, argument.getArgumentString());
	else
		matplotlibcpp::plot(x, y, argument.getArgumentMap());
	plotParameters.flush();

	currentPlotType = CurrentPlotType::Plot1D;
}

void Plotter::plot1D(
	const vector<double> &y,
	const Argument &argument
){
	if(argument.getArgumentMap().size() == 0)
		matplotlibcpp::plot(y, argument.getArgumentString());
	else
		matplotlibcpp::plot(y, argument.getArgumentMap());
	plotParameters.flush();

	currentPlotType = CurrentPlotType::Plot1D;
}

void Plotter::plot2D(
	const vector<vector<double>> &z,
	const Argument &argument
){
	if(z.size() == 0)
		return;
	if(z[0].size() == 0)
		return;

	unsigned int sizeY = z[0].size();
	for(unsigned int x = 1; x < z.size(); x++){
		TBTKAssert(
			z[x].size() == sizeY,
			"Plotter:plot2D()",
			"Incompatible array dimensions. 'z[0]' has " << sizeY
			<< " elements, while 'z[" << x << "]' has "
			<< z[x].size() << " elements.",
			""
		);
	}

	vector<vector<double>> x, y;
	for(unsigned int X = 0; X < z.size(); X++){
		x.push_back(vector<double>());
		y.push_back(vector<double>());
		for(unsigned int Y = 0; Y < z[X].size(); Y++){
			x[X].push_back(X);
			y[X].push_back(Y);
		}
	}

	plot2D(x, y, z, argument);
}

void Plotter::plot2D(
	const vector<vector<double>> &x,
	const vector<vector<double>> &y,
	const vector<vector<double>> &z,
	const Argument &argument
){
	TBTKAssert(
		x.size() == y.size() && x.size() == z.size(),
		"Plotter::plot2D()",
		"Incompatible array dimensions. 'x' has '" << x.size() << "'"
		<< " elements, 'y' has '" << y.size() << "' elements, and 'z'"
		<< " has '" << z.size() << "' elements.",
		""
	);

	unsigned int size[3];
	size[0] = z.size();
	if(size[0] == 0)
		return;
	size[1] = z[0].size();
	for(unsigned int X = 0; X < z.size(); X++){
		TBTKAssert(
			x[X].size() == size[1],
			"Plotter::plot2D()",
			"Incompatible array dimensions. 'x[" << X << "]' has '"
			<< x[X].size() << "' elements, but 'z[0]' has '"
			<< size[1] << "' elements.",
			""
		);
		TBTKAssert(
			y[X].size() == size[1],
			"Plotter::plot2D()",
			"Incompatible array dimensions. 'y[" << X << "]' has '"
			<< y[X].size() << "' elements, but 'z[0]' has '"
			<< size[1] << "' elements.",
			""
		);
		TBTKAssert(
			z[X].size() == size[1],
			"Plotter::plot2D()",
			"Incompatible array dimensions. 'z[" << X << "]' has '"
			<< z[X].size() << "' elements, but 'z[0]' has '"
			<< size[1] << "' elements.",
			""
		);
	}
	if(size[1] == 0)
		return;

	switch(plotMethod3D){
	case PlotMethod3D::PlotSurface:
		matplotlibcpp::plot_surface(
			x,
			y,
			z,
			argument.getArgumentMap()
		);
		plotSurfaceParameters.flush();
		currentPlotType = CurrentPlotType::PlotSurface;
		break;
	case PlotMethod3D::Contourf:
		matplotlibcpp::contourf(x, y, z, argument.getArgumentMap());
		contourfParameters.flush();
		currentPlotType = CurrentPlotType::Contourf;
		break;
	default:
		TBTKExit(
			"Plotter::plot2D()",
			"Unkown plot method.",
			"This should never happen, contact the developer."
		);
	}
}

};	//End of namespace MatPlotLib
};	//End of namespace Visualization
};	//End of namespace TBTK
