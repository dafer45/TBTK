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

/** @package TBTKcalc
 *  @file Smooth.h
 *  @brief Collection of functions for smoothing data.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SMOOTH
#define COM_DAFER45_TBTK_SMOOTH

#include "TBTKMacros.h"

#include <cmath>
#include <vector>

namespace TBTK{

class Smooth{
public:
	/** Gaussian smoothing of custom data. */
	static std::vector<double> gaussian(
		const std::vector<double> &data,
		double sigma,
		int windowSize
	);

	/** Gaussian smoothing of custom data. */
	static std::vector<double> gaussian(
		const double *data,
		unsigned int size,
		double sigma,
		int windowSize
	);

	/** Gaussian smoothing of DOS. */
	static Property::DOS gaussian(
		const Property::DOS &dos,
		double sigma,
		int windowSize
	);
private:
};

inline std::vector<double> Smooth::gaussian(
	const std::vector<double> &data,
	double sigma,
	int windowSize
){
	TBTKAssert(
		windowSize > 0,
		"Smooth::gaussian()",
		"'windowSize' must be larger than zero.",
		""
	);
	TBTKAssert(
		windowSize%2 == 1,
		"Smooth::gaussian()",
		"'windowSize' must be odd.",
		""
	);

	double normalization = 0;
	for(int n = -windowSize/2; n <= windowSize/2; n++){
		Streams::out << "Normalization:\t" << normalization << "\n";
		normalization += exp(-n*n/(2*sigma*sigma));
	}
	Streams::out << "Normalization:\t" << normalization << "\n";
	normalization = 1/normalization;

	std::vector<double> result;
	for(int n = 0; n < data.size(); n++){
		result.push_back(0);
		for(
			int c = std::max(0, n - (int)windowSize/2);
			c < std::min(n + (int)windowSize/2 + 1, (int)data.size());
			c++
		){
			result.at(n) += data.at(c)*exp(-(c-n)*(c-n)/(2*sigma*sigma));
		}
		result.at(n) *= normalization;
	}

	return result;
}

inline std::vector<double> Smooth::gaussian(
	const double *data,
	unsigned int size,
	double sigma,
	int windowSize
){
	std::vector<double> dataVector;
	for(unsigned int n = 0; n < size; n++)
		dataVector.push_back(data[n]);

	return gaussian(
		dataVector,
		sigma,
		windowSize
	);
}

inline Property::DOS Smooth::gaussian(
	const Property::DOS &dos,
	double sigma,
	int windowSize
){
	const double *data = dos.getData();
	std::vector<double> dataVector;
	for(unsigned int n = 0; n < dos.getSize(); n++)
		dataVector.push_back(data[n]);

	double lowerBound = dos.getLowerBound();
	double upperBound = dos.getUpperBound();
	int resolution = dos.getResolution();
	double scaledSigma = sigma/(upperBound - lowerBound)*resolution;

	std::vector<double> smoothedData = gaussian(
		dataVector,
		scaledSigma,
		windowSize
	);

	return Property::DOS(
		lowerBound,
		upperBound,
		resolution,
		smoothedData.data()
	);
}

};	//End of namespace TBTK

#endif
