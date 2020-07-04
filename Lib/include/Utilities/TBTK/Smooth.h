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
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/TBTKMacros.h"

#include <cmath>
#include <vector>

namespace TBTK{

class Smooth{
public:
	/** Gaussian smoothing of custom data. */
	template<typename DataType>
	static Array<DataType> gaussian(
		const Array<DataType> &data,
		double sigma,
		int windowSize
	);

	/** Gaussian smoothing of custom data. */
	template<typename DataType>
	static std::vector<DataType> gaussian(
		const std::vector<DataType> &data,
		double sigma,
		int windowSize
	);

	/** Gaussian smoothing of custom data. */
	template<typename DataType>
	static CArray<DataType> gaussian(
		const CArray<DataType> &data,
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
		const Property::LDOS &ldos,
		double sigma,
		int windowSize
	);

	/** Gaussian smoothing of LDOS. */
	static Property::SpinPolarizedLDOS gaussian(
		const Property::SpinPolarizedLDOS &ldos,
		double sigma,
		int windowSize
	);
private:
};

template<typename DataType>
inline Array<DataType> Smooth::gaussian(
	const Array<DataType> &data,
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

	DataType normalization = 0;
	for(int n = -windowSize/2; n <= windowSize/2; n++){
		normalization += exp(-n*n/(2*sigma*sigma));
	}
	normalization = DataType(1)/normalization;

	Array<DataType> result({data.getRanges()[0]}, 0);
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

template<typename DataType>
inline std::vector<DataType> Smooth::gaussian(
	const std::vector<DataType> &data,
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

	DataType normalization = 0;
	for(int n = -windowSize/2; n <= windowSize/2; n++){
		normalization += exp(-n*n/(2*sigma*sigma));
	}
	normalization = DataType(1)/normalization;

	std::vector<DataType> result;
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

template<typename DataType>
inline CArray<DataType> Smooth::gaussian(
	const CArray<DataType> &data,
	double sigma,
	int windowSize
){
	std::vector<DataType> dataVector;
	for(unsigned int n = 0; n < data.getSize(); n++)
		dataVector.push_back(data[n]);

	std::vector<DataType> resultVector = gaussian(
		dataVector,
		sigma,
		windowSize
	);

	CArray<DataType> result(resultVector.size());
	for(unsigned int n = 0; n < resultVector.size(); n++)
		result[n] = resultVector[n];

	return result;
}

inline Property::DOS Smooth::gaussian(
	const Property::DOS &dos,
	double sigma,
	int windowSize
){
	const std::vector<double> &data = dos.getData();
	CArray<double> dataCArray(data.size());
	for(unsigned int n = 0; n < data.size(); n++)
		dataCArray[n] = data[n];

	double lowerBound = dos.getLowerBound();
	double upperBound = dos.getUpperBound();
	int resolution = dos.getResolution();
	double scaledSigma = sigma/(upperBound - lowerBound)*resolution;

	CArray<double> smoothedData = gaussian(
		dataCArray,
		scaledSigma,
		windowSize
	);

	return Property::DOS(
		Range(
			lowerBound,
			upperBound,
			resolution
		),
		smoothedData
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

inline Property::SpinPolarizedLDOS Smooth::gaussian(
	const Property::SpinPolarizedLDOS &spinPolarizedLDOS,
	double sigma,
	int windowSize
){
	Property::SpinPolarizedLDOS newSpinPolarizedLDOS = spinPolarizedLDOS;
	std::vector<SpinMatrix> &newData = newSpinPolarizedLDOS.getDataRW();

	const std::vector<SpinMatrix> &data = spinPolarizedLDOS.getData();
	unsigned int blockSize = spinPolarizedLDOS.getBlockSize();
	unsigned int numBlocks = spinPolarizedLDOS.getSize()/blockSize;
	double lowerBound = spinPolarizedLDOS.getLowerBound();
	double upperBound = spinPolarizedLDOS.getUpperBound();
	int resolution = spinPolarizedLDOS.getResolution();
	double scaledSigma = sigma/(upperBound - lowerBound)*resolution;
	for(unsigned int block = 0; block < numBlocks; block++){
		std::vector<SpinMatrix> blockData(blockSize);
		for(unsigned int n = 0; n < blockSize; n++)
			blockData[n] = data[block*blockSize + n];

		std::vector<
			std::vector<std::complex<double>>
		> spinMatrixComponents(
			4,
			std::vector<std::complex<double>>(blockSize)
		);
		for(unsigned int n = 0; n < blockSize; n++){
			spinMatrixComponents[0][n] = blockData[n].at(0, 0);
			spinMatrixComponents[1][n] = blockData[n].at(0, 1);
			spinMatrixComponents[2][n] = blockData[n].at(1, 0);
			spinMatrixComponents[3][n] = blockData[n].at(1, 1);
		}

		for(unsigned int n = 0; n < 4; n++){
			spinMatrixComponents[n] = gaussian(
				spinMatrixComponents[n],
				scaledSigma,
				windowSize
			);
		}

		for(unsigned int n = 0; n < blockSize; n++){
			newData[block*blockSize + n].at(0, 0)
				= spinMatrixComponents[0][n];
			newData[block*blockSize + n].at(0, 1)
				= spinMatrixComponents[1][n];
			newData[block*blockSize + n].at(1, 0)
				= spinMatrixComponents[2][n];
			newData[block*blockSize + n].at(1, 1)
				= spinMatrixComponents[3][n];
		}
	}

	return newSpinPolarizedLDOS;
}

};	//End of namespace TBTK

#endif
