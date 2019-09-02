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

/** @file PNGCanvas2D.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/PNGCanvas2D.h"
#include "TBTK/Smooth.h"

#include "TBTK/External/Boost/gnuplot-iostream.h"

using namespace std;

namespace TBTK{

PNGCanvas2D::PNGCanvas2D(const Canvas2D &canvas) : Canvas2D(canvas){
}

PNGCanvas2D::~PNGCanvas2D(){
}

void PNGCanvas2D::flush(const string &filename){
	TBTKAssert(
		getNumDataSets() > 0,
		"PNGCanvas2D::flush()",
		"Unable to flush the data to file since the canvas is empty.",
		""
	);

	Gnuplot gnuplot;
	gnuplot << "set terminal png size " << getWidth() <<  "," << getHeight() << "\n";
	gnuplot << "set output '" << filename << "'\n";
	gnuplot << "set title '" << Canvas::getTitle() << "'\n";
	gnuplot << "set xlabel '" << getLabelX() << "'\n";
	gnuplot << "set ylabel '" << getLabelY() << "'\n";
	gnuplot << "plot";
	for(unsigned int n = 0; n < getNumDataSets(); n++){
		if(n != 0)
			gnuplot << ",";
		gnuplot << " '-' using 1:2"
			<< " title '" << getTitle(n) << "'"
			<< " lw " << getSize(n)
			<< " lt rgb '" << convertColorToHex(getColor(n)) << "'"
			<< " with lines";
	}
	gnuplot << "\n";
	for(unsigned int n = 0; n < getNumDataSets(); n++)
		gnuplot.send1d(boost::make_tuple(getX(n), getY(n)));
}

string PNGCanvas2D::convertColorToHex(const std::vector<unsigned char> &color){
	const vector<char> hexadecimals = {
		'0', '1', '2', '3', '4', '5', '6', '7',
		'8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
	};

	string result = "#";
	for(unsigned int n = 0; n < 3; n++){
		result += hexadecimals[color[n]/16];
		result += hexadecimals[color[n]%16];
	}

	return result;
}

};	//End of namespace TBTK
