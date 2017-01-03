/* Copyright 2016 Kristofer Björnson
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
 *  @file FourierTransform.h
 *  @brief Fourier transform
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_FOURIER_TRANSFORM
#define COM_DAFER45_TBTK_FOURIER_TRANSFORM

#include "Index.h"

#include <fftw3.h>

#include <complex>

namespace TBTK{

class FourierTransform{
public:
	/** One-dimensional real Fourier transform. */
	static void transform(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sign
	);

	/** Two-dimensional complex Fourier transform. */
	static void transform(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY,
		int sign
	);

	/** Three-dimensional complex Fourier transform. */
	static void transform(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY,
		int sizeZ,
		int sign
	);

	/** One-dimensional complex forward Fourier transform. */
	static void forward(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX
	);

	/** Two-dimensional complex forward Fourier transform. */
	static void forward(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY
	);

	/** Three-dimensional complex forward Fourier transform. */
	static void forward(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY,
		int sizeZ
	);

	/** One-dimensional complex inverse Fourier transform. */
	static void inverse(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX
	);

	/** Two-dimensional complex inverse Fourier transform. */
	static void inverse(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY
	);

	/** Three-dimensional complex inverse Fourier transform. */
	static void inverse(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY,
		int sizeZ
	);
private:
};

inline void FourierTransform::forward(
	std::complex<double> *in,
	std::complex<double> *out,
	int sizeX
){
	transform(in, out, sizeX, -1);
}

inline void FourierTransform::forward(
	std::complex<double> *in,
	std::complex<double> *out,
	int sizeX,
	int sizeY
){
	transform(in, out, sizeX, sizeY, -1);
}

inline void FourierTransform::forward(
	std::complex<double> *in,
	std::complex<double> *out,
	int sizeX,
	int sizeY,
	int sizeZ
){
	transform(in, out, sizeX, sizeY, sizeZ, -1);
}

inline void FourierTransform::inverse(
	std::complex<double> *in,
	std::complex<double> *out,
	int sizeX
){
	transform(in, out, sizeX, 1);
}

inline void FourierTransform::inverse(
	std::complex<double> *in,
	std::complex<double> *out,
	int sizeX,
	int sizeY
){
	transform(in, out, sizeX, sizeY, 1);
}

inline void FourierTransform::inverse(
	std::complex<double> *in,
	std::complex<double> *out,
	int sizeX,
	int sizeY,
	int sizeZ
){
	transform(in, out, sizeX, sizeY, sizeZ, 1);
}

};	//End of namespace TBTK

#endif
