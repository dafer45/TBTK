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

#include "TBTK/Array.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/TBTKMacros.h"

#include <cmath>
#include <vector>

namespace TBTK{

class Smooth{
public:
	/** Gaussian smoothing of custom data. */
	static Array<double> gaussian(
		const Array<double> &data,
		double sigma,
		int windowSize
	);

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

	/** Gaussian smoothing of LDOS. */
	static Property::LDOS gaussian(
		const Property::LDOS &dos,
		double sigma,
		int windowSize
	);
private:
};

inline Array<double> Smooth::gaussian(
	const Array<double> &data,
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
	TBTKAssert(
		data.getRanges().size() == 1,
		"Smooth::gaussian()",
		"Array must have rank 1, but the rank is "
		<< data.getRanges().size() << ".",
		""
	);

	double normalization = 0;
	for(int n = -windowSize/2; n <= windowSize/2; n++){
		normalization += exp(-n*n/(2*sigma*sigma));
	}
	normalization = 1/normalization;

	Array<double> result({data.getRanges()[0]}, 0);
	for(int n = 0; n < (int)data.getRanges()[0]; n++){
		for(
			int c = std::max(0, (int)n - (int)windowSize/2);
			c < std::min(
				(int)n + (int)windowSize/2 + 1,
				(int)data.getRanges()[0]
			);
			c++
		){
			result[{(unsigned int)n}]
				+= data[
					{(unsigned int)c}
				]*exp(-(c-n)*(c-n)/(2*sigma*sigma));
		}
		result[{(unsigned int)n}] *= normalization;
	}

	return result;
}

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
		normalization += exp(-n*n/(2*sigma*sigma));
	}
	normalization = 1/normalization;

	std::vector<double> result;
	for(int n = 0; n < (int)data.size(); n++){
		result.push_back(0);
		for(
			int c = std::max(0, (int)n - (int)windowSize/2);
			c < std::min(
				(int)n + (int)windowSize/2 + 1,
				(int)data.size()
			);
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
	const std::vector<double> &data = dos.getData();
	std::vector<double> dataVector;
	for(unsigned int n = 0; n < data.size(); n++)
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

inline Property::LDOS Smooth::gaussian(
	const Property::LDOS &ldos,
	double sigma,
	int windowSize
){
	Property::LDOS newLdos = ldos;
	std::vector<double> &newData = newLdos.getDataRW();

	const std::vector<double> &data = ldos.getData();
	unsigned int blockSize = ldos.getBlockSize();
	unsigned int numBlocks = ldos.getSize()/blockSize;
	double lowerBound = ldos.getLowerBound();
	double upperBound = ldos.getUpperBound();
	int resolution = ldos.getResolution();
	double scaledSigma = sigma/(upperBound - lowerBound)*resolution;
	for(unsigned int block = 0; block < numBlocks; block++){
		std::vector<double> blockData(blockSize);
		for(unsigned int n = 0; n < blockSize; n++)
			blockData[n] = data[block*blockSize + n];


		std::vector<double> smoothedData = gaussian(
			blockData,
			scaledSigma,
			windowSize
		);

		for(unsigned int n = 0; n < blockSize; n++)
			newData[block*blockSize + n] = smoothedData[n];
	}

	return newLdos;
}

};	//End of namespace TBTK

#endif
