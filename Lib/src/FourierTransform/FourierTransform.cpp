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

/** @file FourierTransform.h
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/FourierTransform.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

void FourierTransform::transform(
	const complex<double> *in,
	complex<double> *out,
	int sizeX,
	int sign
){
	fftw_plan plan;

	//Const cast is OK since FFTW3 does not modify the input.
	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	plan = fftw_plan_dft_1d(
		sizeX,
		const_cast<fftw_complex*>(
			reinterpret_cast<const fftw_complex*>(in)
		),
		reinterpret_cast<fftw_complex*>(out),
		sign,
		FFTW_ESTIMATE
	);

	fftw_execute(plan);

	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	fftw_destroy_plan(plan);

	for(int n = 0; n < sizeX; n++)
		out[n] /= sqrt(sizeX);
}

void FourierTransform::transform(
	const complex<double> *in,
	complex<double> *out,
	int sizeX,
	int sizeY,
	int sign
){
	fftw_plan plan;

	//Const cast is OK since FFTW3 does not modify the input.
	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	plan = fftw_plan_dft_2d(
		sizeX,
		sizeY,
		const_cast<fftw_complex*>(
			reinterpret_cast<const fftw_complex*>(in)
		),
		reinterpret_cast<fftw_complex*>(out),
		sign,
		FFTW_ESTIMATE
	);

	fftw_execute(plan);

	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	fftw_destroy_plan(plan);

	for(int n = 0; n < sizeX*sizeY; n++)
		out[n] /= sqrt(sizeX*sizeY);
}

void FourierTransform::transform(
	const complex<double> *in,
	complex<double> *out,
	int sizeX,
	int sizeY,
	int sizeZ,
	int sign
){
	fftw_plan plan;

	//Const cast is OK since FFTW3 does not modify the input.
	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	plan = fftw_plan_dft_3d(
		sizeX,
		sizeY,
		sizeZ,
		const_cast<fftw_complex*>(
			reinterpret_cast<const fftw_complex*>(in)
		),
		reinterpret_cast<fftw_complex*>(out),
		sign,
		FFTW_ESTIMATE
	);

	fftw_execute(plan);

	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	fftw_destroy_plan(plan);

	for(int n = 0; n < sizeX*sizeY*sizeZ; n++)
		out[n] /= sqrt(sizeX*sizeY*sizeZ);
}

void FourierTransform::transform(
	const complex<double> *in,
	complex<double> *out,
	const vector<unsigned int> &ranges,
	int sign
){
	fftw_plan plan;

	//Const cast is OK since FFTW3 does not modify the input.
	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	plan = fftw_plan_dft(
		ranges.size(),
		(int*)ranges.data(),
		const_cast<fftw_complex*>(
			reinterpret_cast<const fftw_complex*>(in)
		),
		reinterpret_cast<fftw_complex*>(out),
		sign,
		FFTW_ESTIMATE
	);

	fftw_execute(plan);

	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	fftw_destroy_plan(plan);

	double numElements = 1;
	for(unsigned int n = 0; n < ranges.size(); n++)
		numElements *= ranges[n];

	for(int n = 0; n < numElements; n++)
		out[n] /= sqrt(numElements);
}

template<>
FourierTransform::Plan<complex<double>>::Plan(
	const complex<double> *in,
	complex<double> *out,
	int sizeX,
	int sign
){
	plan = new fftw_plan();

	//Const cast is OK since FFTW3 does not modify the input.
	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	*plan = fftw_plan_dft_1d(
		sizeX,
		const_cast<fftw_complex*>(
			reinterpret_cast<const fftw_complex*>(in)
		),
		reinterpret_cast<fftw_complex*>(out),
		sign,
		FFTW_ESTIMATE
	);

	input = in;
	output = out;
	size = sizeX;
	normalizationFactor = sqrt(sizeX);
}

template<>
FourierTransform::Plan<complex<double>>::Plan(
	const complex<double> *in,
	complex<double> *out,
	int sizeX,
	int sizeY,
	int sign
){
	plan = new fftw_plan();

	//Const cast is OK since FFTW3 does not modify the input.
	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	*plan = fftw_plan_dft_2d(
		sizeX,
		sizeY,
		const_cast<fftw_complex*>(
			reinterpret_cast<const fftw_complex*>(in)
		),
		reinterpret_cast<fftw_complex*>(out),
		sign,
		FFTW_ESTIMATE
	);

	input = in;
	output = out;
	size = sizeX*sizeY;
	normalizationFactor = sqrt(sizeX*sizeY);
}

template<>
FourierTransform::Plan<complex<double>>::Plan(
	const complex<double> *in,
	complex<double> *out,
	int sizeX,
	int sizeY,
	int sizeZ,
	int sign
){
	plan = new fftw_plan();

	//Const cast is OK since FFTW3 does not modify the input.
	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	*plan = fftw_plan_dft_3d(
		sizeX,
		sizeY,
		sizeZ,
		const_cast<fftw_complex*>(
			reinterpret_cast<const fftw_complex*>(in)
		),
		reinterpret_cast<fftw_complex*>(out),
		sign,
		FFTW_ESTIMATE
	);

	input = in;
	output = out;
	size = sizeX*sizeY*sizeZ;
	normalizationFactor = sqrt(sizeX*sizeY*sizeZ);
}

template<>
FourierTransform::Plan<complex<double>>::Plan(
	const complex<double> *in,
	complex<double> *out,
	const std::vector<unsigned int> &ranges,
	int sign
){
	plan = new fftw_plan();

	//Const cast is OK since FFTW3 does not modify the input.
	#pragma omp critical (TBTK_FOURIER_TRANSFORM)
	*plan = fftw_plan_dft(
		ranges.size(),
		(int*)ranges.data(),
		const_cast<fftw_complex*>(
			reinterpret_cast<const fftw_complex*>(in)
		),
		reinterpret_cast<fftw_complex*>(out),
		sign,
		FFTW_ESTIMATE
	);

	input = in;
	output = out;
	size = 1;
	for(unsigned int n = 0; n < ranges.size(); n++)
		size *= ranges[n];
	normalizationFactor = sqrt(size);
}

};
