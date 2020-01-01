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

//Python is explicitly included before all other libraries to avoid warning
//that _POSIX_C_SOURCE is redefined.
#include "Python.h"

#include "TBTK/AnnotatedArray.h"
#include "TBTK/MultiCounter.h"
#include "TBTK/PropertyConverter.h"
#include "TBTK/Range.h"
#include "TBTK/Smooth.h"
#include "TBTK/Streams.h"
#include "TBTK/Visualization/MatPlotLib/ColorMap.h"
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
	const Argument &argument
){
	setTitle("DOS", false);
	setLabelX("Energy", false);
	setLabelY("DOS", false);

	Array<double> x({dos.getSize()});
	Array<double> y({dos.getSize()});
	double dE = dos.getDeltaE();
	for(unsigned int n = 0; n < dos.getSize(); n++){
		x[{n}] = dos.getLowerBound() + n*dE;
		y[{n}] = dos(n);
	}

	plot(x, y, argument);
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
	AnnotatedArray<double, double> annotatedArrayWithDoubleAxes
		= convertAxes(
			annotatedArray,
			{
				{
					annotatedArray.getAxes().size()-1,
					{
						ldos.getLowerBound(),
						ldos.getUpperBound()
					}
				}
			}
		);

	setTitle("LDOS", false);
	switch(annotatedArray.getRanges().size()){
	case 1:
		setLabelX("Energy", false);
		break;
	case 2:
		setLabelX("x", false);
		setLabelY("Energy", false);
		break;
	default:
		break;
	}

	plot(annotatedArrayWithDoubleAxes, argument);
}

void Plotter::plot(
	const Index &pattern,
	const Property::LDOS &ldos,
	const Argument &argument
){
	AnnotatedArray<double, Subindex> annotatedArray
		= PropertyConverter::convert(ldos, pattern);
	AnnotatedArray<double, double> annotatedArrayWithDoubleAxes
		= convertAxes(
			annotatedArray,
			{
				{
					annotatedArray.getAxes().size()-1,
					{
						ldos.getLowerBound(),
						ldos.getUpperBound()
					}
				}
			}
		);

	setTitle("LDOS", false);
	switch(annotatedArray.getRanges().size()){
	case 1:
		setLabelX("Energy");
		break;
	case 2:
		setLabelX("x", false);
		setLabelY("Energy", false);
		break;
	default:
		break;
	}

	plot(annotatedArrayWithDoubleAxes, argument);
}

void Plotter::plot(
	const Vector3d &direction,
	const Property::Magnetization &magnetization,
	const Argument &argument
){
	AnnotatedArray<SpinMatrix, Subindex> annotatedArray
		= PropertyConverter::convert(magnetization);
	AnnotatedArray<double, Subindex> annotatedArrayWithDoubleValues
		= convertSpinMatrixToDouble(annotatedArray, direction);

	setTitle("Magnetization", false);
	switch(annotatedArrayWithDoubleValues.getRanges().size()){
	case 1:
		setLabelX("x", false);
		break;
	case 2:
		setLabelX("x", false);
		setLabelY("y", false);
		break;
	default:
		break;
	}

	plot(annotatedArrayWithDoubleValues, argument);
}

void Plotter::plot(
	const Index &pattern,
	const Vector3d &direction,
	const Property::Magnetization &magnetization,
	const Argument &argument
){
	AnnotatedArray<SpinMatrix, Subindex> annotatedArray
		= PropertyConverter::convert(magnetization, pattern);
	AnnotatedArray<double, Subindex> annotatedArrayWithDoubleValues
		= convertSpinMatrixToDouble(annotatedArray, direction);

	setTitle("Magnetization", false);
	switch(annotatedArrayWithDoubleValues.getRanges().size()){
	case 1:
		setLabelX("x");
		break;
	case 2:
		setLabelX("x", false);
		setLabelY("y", false);
		break;
	default:
		break;
	}

	plot(annotatedArrayWithDoubleValues, argument);
}

void Plotter::plot(
	const Vector3d &direction,
	const Property::SpinPolarizedLDOS &spinPolarizedLDOS,
	const Argument &argument
){
	AnnotatedArray<SpinMatrix, Subindex> annotatedArray
		= PropertyConverter::convert(spinPolarizedLDOS);
	AnnotatedArray<double, Subindex> annotatedArrayWithDoubleValues
		= convertSpinMatrixToDouble(annotatedArray, direction);
	AnnotatedArray<double, double> annotatedArrayWithDoubleAxes
		= convertAxes(
			annotatedArrayWithDoubleValues,
			{
				{
					annotatedArrayWithDoubleValues.getAxes(
					).size()-1,
					{
						spinPolarizedLDOS.getLowerBound(),
						spinPolarizedLDOS.getUpperBound()
					}
				}
			}
		);

	setTitle("Spin-polarized LDOS", false);
	switch(annotatedArrayWithDoubleAxes.getRanges().size()){
	case 1:
		setLabelX("Energy", false);
		break;
	case 2:
		setLabelX("x", false);
		setLabelY("Energy", false);
		break;
	default:
		break;
	}

	plot(annotatedArrayWithDoubleAxes, argument);
}

void Plotter::plot(
	const Index &pattern,
	const Vector3d &direction,
	const Property::SpinPolarizedLDOS &spinPolarizedLDOS,
	const Argument &argument
){
	AnnotatedArray<SpinMatrix, Subindex> annotatedArray
		= PropertyConverter::convert(spinPolarizedLDOS, pattern);
	AnnotatedArray<double, Subindex> annotatedArrayWithDoubleValues
		= convertSpinMatrixToDouble(annotatedArray, direction);
	AnnotatedArray<double, double> annotatedArrayWithDoubleAxes
		= convertAxes(
			annotatedArrayWithDoubleValues,
			{
				{
					annotatedArrayWithDoubleValues.getAxes(
					).size()-1,
					{
						spinPolarizedLDOS.getLowerBound(),
						spinPolarizedLDOS.getUpperBound()
					}
				}
			}
		);

	setTitle("Spin-polarized LDOS", false);
	switch(annotatedArrayWithDoubleAxes.getRanges().size()){
	case 1:
		setLabelX("Energy");
		break;
	case 2:
		setLabelX("x", false);
		setLabelY("Energy", false);
		break;
	default:
		break;
	}

	plot(annotatedArrayWithDoubleAxes, argument);
}

void Plotter::plot(
	unsigned int state,
	const Property::WaveFunctions &waveFunctions,
	const Argument &argument
){
	const vector<unsigned int> &states = waveFunctions.getStates();
	unsigned int stateID;
	bool foundState = false;
	for(unsigned int n = 0; n < states.size(); n++){
		if(states[n] == state){
			stateID = n;
			foundState = true;
			break;
		}
	}
	TBTKAssert(
		foundState,
		"Plotter::plot()",
		"State '" << state << "' is not included in 'waveFunctions'.",
		""
	);

	AnnotatedArray<std::complex<double>, Subindex> annotatedArray
		= PropertyConverter::convert(waveFunctions);

	vector<Subindex> slicePattern(annotatedArray.getRanges().size());
	for(unsigned int n = 0; n < slicePattern.size()-1; n++)
		slicePattern[n] = IDX_ALL;
	slicePattern.back() = stateID;

	vector<vector<Subindex>> axes = annotatedArray.getAxes();
	axes.pop_back();

	annotatedArray = AnnotatedArray<complex<double>, Subindex>(
		annotatedArray.getSlice(slicePattern),
		axes
	);

	setTitle("Wave function (state = " + to_string(state) +  ")", false);
	switch(annotatedArray.getRanges().size()){
	case 1:
	{
		AnnotatedArray<double, Subindex> annotatedArrayReal(
			Array<double>::create(annotatedArray.getRanges()),
			annotatedArray.getAxes()
		);
		AnnotatedArray<double, Subindex> annotatedArrayImaginary(
			Array<double>::create(annotatedArray.getRanges()),
			annotatedArray.getAxes()
		);
		for(unsigned int n = 0; n < annotatedArray.getSize(); n++){
			annotatedArrayReal[n] = real(annotatedArray[n]);
			annotatedArrayImaginary[n] = imag(annotatedArray[n]);
		}

		setLabelX("x", false);
		setLabelY("Amplitude (real and imaginary)", false);
		plot(annotatedArrayReal, argument);
		plot(annotatedArrayImaginary, argument);
		break;
	}
	case 2:
	{
		AnnotatedArray<double, Subindex> annotatedArrayDensity(
			Array<double>::create(annotatedArray.getRanges()),
			annotatedArray.getAxes()
		);
		for(unsigned int n = 0; n < annotatedArray.getSize(); n++)
			annotatedArrayDensity[n] = pow(abs(annotatedArray[n]), 2);

		setLabelX("x", false);
		setLabelX("y", false);
		setLabelY("Amplitude (absolute squared)", false);
		plot(annotatedArrayDensity, argument);
		break;
	}
	default:
		break;
	}
}

void Plotter::plot(
	const Index &pattern,
	unsigned int state,
	const Property::WaveFunctions &waveFunctions,
	const Argument &argument
){
	const vector<unsigned int> &states = waveFunctions.getStates();
	unsigned int stateID;
	bool foundState = false;
	for(unsigned int n = 0; n < states.size(); n++){
		if(states[n] == state){
			stateID = n;
			foundState = true;
			break;
		}
	}
	TBTKAssert(
		foundState,
		"Plotter::plot()",
		"State '" << state << "' is not included in 'waveFunctions'.",
		""
	);

	AnnotatedArray<complex<double>, Subindex> annotatedArray
		= PropertyConverter::convert(waveFunctions, pattern);

	vector<Subindex> slicePattern(annotatedArray.getRanges().size());
	for(unsigned int n = 0; n < slicePattern.size()-1; n++)
		slicePattern[n] = IDX_ALL;
	slicePattern.back() = stateID;

	vector<vector<Subindex>> axes = annotatedArray.getAxes();
	axes.pop_back();

	annotatedArray = AnnotatedArray<complex<double>, Subindex>(
		annotatedArray.getSlice(slicePattern),
		axes
	);

	setTitle("Wave function (state = " + to_string(state) + ")", false);
	switch(annotatedArray.getRanges().size()){
	case 1:
	{
		AnnotatedArray<double, Subindex> annotatedArrayReal(
			Array<double>::create(annotatedArray.getRanges()),
			annotatedArray.getAxes()
		);
		AnnotatedArray<double, Subindex> annotatedArrayImaginary(
			Array<double>::create(annotatedArray.getRanges()),
			annotatedArray.getAxes()
		);
		for(unsigned int n = 0; n < annotatedArray.getSize(); n++){
			annotatedArrayReal[n] = real(annotatedArray[n]);
			annotatedArrayImaginary[n] = imag(annotatedArray[n]);
		}


		setLabelX("x");
		setLabelY("Amplitude (real and imaginary)");
		plot(annotatedArrayReal, argument);
		plot(annotatedArrayImaginary, argument);
		break;
	}
	case 2:
	{
		AnnotatedArray<double, Subindex> annotatedArrayDensity(
			Array<double>::create(annotatedArray.getRanges()),
			annotatedArray.getAxes()
		);
		for(unsigned int n = 0; n < annotatedArray.getSize(); n++)
			annotatedArrayDensity[n] = pow(abs(annotatedArray[n]), 2);

		setLabelX("x", false);
		setLabelX("y", false);
		setLabelY("Amplitude (absolute squared)", false);
		plot(annotatedArrayDensity, argument);
		break;
	}
	default:
		break;
	}
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
		Array<double> x({ranges[0]});
		for(unsigned int n = 0; n < ranges[0]; n++)
			x[{n}] = n;
		x = getNonDefaultAxis(x, 0);
		plot(x, data, argument);

		break;
	}
	case 2:
	{
		Array<double> x({ranges[0], ranges[1]});
		Array<double> y({ranges[0], ranges[1]});
		for(unsigned int X = 0; X < ranges[0]; X++){
			for(unsigned int Y = 0; Y < ranges[1]; Y++){
				x[{X, Y}] = X;
				y[{X, Y}] = Y;
			}
		}

		plot(x, y, data, argument);

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
	Array<double> x,
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
	x = getNonDefaultAxis(x, 0);

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
	Array<double> x,
	Array<double> y,
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
	x = getNonDefaultAxis(x, 0);
	y = getNonDefaultAxis(y, 1);

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

string Plotter::colorToHex(const Array<double> &color) const{
	const vector<unsigned int> &ranges = color.getRanges();
	TBTKAssert(
		ranges.size() == 1,
		"Plotter::colorToHex()",
		"Invalid color. The number of ranges must be 1, but is '"
		<< ranges.size() << "'.",
		""
	);
	TBTKAssert(
		ranges[0] == 3,
		"Plotter::colorToHex()",
		"Invalid color. The number of colors must be 3, but is '"
		<< ranges[0] << "'.",
		""
	);
	string hexString = "#";
	for(unsigned int n = 0; n < ranges[0]; n++)
		hexString += doubleToHex(color[{n}]);

	return hexString;
}

string Plotter::doubleToHex(double value) const{
	if(value < 0)
		return "00";
	if(value >= 1)
		return "FF";

	char digits[] = "0123456789ABCDEF";
	int x = 256*value;
	string result;
	result += digits[(x/16)];
	result += digits[x%16];

	return result;
}

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

	if(argument.getArgumentString().compare("") != 0){
		matplotlibcpp::plot(x, y, argument.getArgumentString());
	}
	else{
		std::map<string, string> argumentMap
			= argument.getArgumentMap();
		if(argumentMap.find("color") == argumentMap.end()){
			unsigned int colorID = (ColorMap::inferno.getRanges()[0]*11/15.)*numLines;
			colorID %= ColorMap::inferno.getRanges()[0];
			argumentMap.insert({
				"color",
				colorToHex(
					ColorMap::inferno.getSlice(
						{colorID, IDX_ALL}
					)
				)
			});
		}
		if(argumentMap.find("linestyle") == argumentMap.end()){
			unsigned int lineStyleID = (numLines/6)%3;
			string lineStyle;
			switch(lineStyleID){
			case 0:
				lineStyle = "-";
				break;
			case 1:
				lineStyle = "--";
				break;
			case 2:
				lineStyle = "-.";
				break;
			}
			argumentMap.insert({"linestyle", lineStyle});
		}
		if(argumentMap.find("linewidth") == argumentMap.end())
			argumentMap.insert({"linewidth", "2"});
		matplotlibcpp::plot(x, y, argumentMap);

		if(argumentMap.find("label") != argumentMap.end())
			matplotlibcpp::legend();
	}
	plotParameters.flush();

	currentPlotType = CurrentPlotType::Plot1D;
	numLines++;
}

void Plotter::plot1D(
	const vector<double> &y,
	const Argument &argument
){
	vector<double> x(y.size());
	for(unsigned int n = 0; n < x.size(); n++)
		x[n] = n;
	plot1D(x, y, argument);
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
	{
		std::map<string, string> argumentMap
			= argument.getArgumentMap();
		if(argumentMap.find("cmap") == argumentMap.end())
			argumentMap.insert({"cmap", "inferno"});
		matplotlibcpp::plot_surface(x, y, z, argumentMap);
		plotSurfaceParameters.flush();
		currentPlotType = CurrentPlotType::PlotSurface;
		break;
	}
	case PlotMethod3D::Contourf:
	{
		std::map<string, string> argumentMap
			= argument.getArgumentMap();
		if(argumentMap.find("cmap") == argumentMap.end())
			argumentMap.insert({"cmap", "inferno"});
		matplotlibcpp::contourf(x, y, z, numContours, argumentMap);
		contourfParameters.flush();
		currentPlotType = CurrentPlotType::Contourf;
		break;
	}
	default:
		TBTKExit(
			"Plotter::plot2D()",
			"Unkown plot method.",
			"This should never happen, contact the developer."
		);
	}
}

AnnotatedArray<double, double> Plotter::convertAxes(
	const AnnotatedArray<double, Subindex> &annotatedArray,
	const initializer_list<
		pair<unsigned int, pair<double, double>>
	> &axisReplacement
){
	const vector<vector<Subindex>> &axes = annotatedArray.getAxes();
	vector<vector<double>> newAxes(axes.size());
	for(unsigned int n = 0; n < axes.size(); n++)
		for(unsigned int c = 0; c < axes[n].size(); c++)
			newAxes[n].push_back(axes[n][c]);

	for(
		initializer_list<
			pair<unsigned int, pair<double, double>>
		>::const_iterator iterator = axisReplacement.begin();
		iterator != axisReplacement.end();
		++iterator
	){
		unsigned int axisID = iterator->first;
		TBTKAssert(
			axisID < axes.size(),
			"Plotter::convertAxes()",
			"'axisID' cannot be larger than the number of axes,"
			<< " but 'axisID' is '" << axisID << "', while the"
			<< " number of axes are '" << axes.size() << "'.",
			""
		);
		double lowerBound = iterator->second.first;
		double upperBound = iterator->second.second;
		if(newAxes[axisID].size() == 1){
			newAxes[axisID][0] = lowerBound;
		}
		else{
			unsigned int numSteps = newAxes[axisID].size();
			double stepLength
				= (upperBound - lowerBound)/(numSteps - 1);
			for(unsigned int n = 0; n < numSteps; n++)
				newAxes[axisID][n] = lowerBound + n*stepLength;
		}
	}

	return AnnotatedArray<double, double>(annotatedArray, newAxes);
}

AnnotatedArray<double, Subindex> Plotter::convertSpinMatrixToDouble(
	const AnnotatedArray<SpinMatrix, Subindex> &annotatedArray,
	const Vector3d &direction
) const{
	Array<double> result
		= Array<double>::create(annotatedArray.getRanges());
	for(unsigned int n = 0; n < result.getSize(); n++){
		result[n] = Vector3d::dotProduct(
			direction,
			annotatedArray[n].getSpinVector()
		);
	}

	return AnnotatedArray<double, Subindex>(
		result,
		annotatedArray.getAxes()
	);
}

Array<double> Plotter::getNonDefaultAxis(
	const Array<double> &axis,
	unsigned int axisID
) const{
	const vector<unsigned int> &ranges = axis.getRanges();
	TBTKAssert(
		axisID < ranges.size(),
		"Plotter::getNonDefaultAxis()",
		"Unable to calculate non-default axis because 'axisID="
		<< axisID << "', but the data only has '"
		<< ranges.size() << "' mutable axes.",
		""
	);
	Array<double> result = axis;
	for(unsigned int n = 0; n < axes.size(); n++){
		if(axisID != axes[n].first)
			continue;

		vector<unsigned int> begin;
		vector<unsigned int> end;
		vector<unsigned int> increment;
		for(unsigned int c = 0; c < ranges.size(); c++){
			begin.push_back(0);
			end.push_back(ranges[c]);
			increment.push_back(1);
		}
		MultiCounter<unsigned int> counter(begin, end, increment);
		switch(axes[n].second.size()){
		case 2:
		{
			const vector<double> &bounds = axes[n].second;
			Range ticks(
				bounds[0],
				bounds[1],
				ranges[axisID],
				true,
				true
			);
			for(counter.reset(); !counter.done(); ++counter)
				result[counter] = ticks[counter[axisID]];
			break;
		}
		default:
		{
			const vector<double> &ticks = axes[n].second;
			TBTKAssert(
				ticks.size() == ranges[axisID],
				"Plotter::getNonDefaultAxis()",
				"Incompatible ticks and axis data. The number"
				" of ticks for the axis with 'axisID="
				<< axisID << "' is " << ticks.size() << ", but"
				<< " the supplied axis runs over '"
				<< ranges[axisID] << "' values.",
				""
			);
			for(counter.reset(); !counter.done(); ++counter)
				result[counter] = ticks[counter[axisID]];
			break;
		}
		}
	}

	return result;
}

};	//End of namespace MatPlotLib
};	//End of namespace Visualization
};	//End of namespace TBTK
