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

#include "TBTK/CArray.h"
#include "TBTK/Index.h"

#include <fftw3.h>

#include <complex>
#include <vector>

namespace TBTK{

class FourierTransform{
public:
	/** Plan for executing the Fourier-transform. */
	template<typename DataType>
	class Plan{
	public:
		/** Constructor. */
		Plan(
			const CArray<DataType> &in,
			CArray<DataType> &out,
			const std::vector<unsigned int> &ranges,
			int sign
		);

		/** Copy constructor. */
		Plan(const Plan &plan) = delete;

		/** Move constructor. */
		Plan(Plan &&plan);

		/** Destructor. */
		~Plan();

		/** Assignment operator. */
		Plan& operator=(const Plan &plan) = delete;

		/** Move assignment operator. */
		Plan& operator=(Plan &&plan);

		/** Set normalization factor. */
		void setNormalizationFactor(double normalizationFactor);

		/** Get normalizationFactor. */
		double getNormalizationFactor() const;
	private:
		/** FFTW3 plan. */
		fftw_plan *plan;

		/** Normalization factor. */
		double normalizationFactor;

		/** Data size. */
		unsigned int size;

		/** Input data. */
		const CArray<DataType> &input;

		/** Output data. */
		CArray<DataType> &output;

		/** Get FFTW3 plan. */
		fftw_plan& getFFTWPlan();

		/** Get data size. */
		unsigned int getSize() const;

		/** Get input data. */
		CArray<DataType>& getInput();

		/** Get output data. */
		CArray<DataType>& getOutput();

		/** Make FourierTransform a friend class. */
		friend class FourierTransform;
	};

	/** Plan for executing forward Fourier-transform. */
	template<typename DataType>
	class ForwardPlan : public Plan<DataType>{
	public:
		/** Constructor. */
		ForwardPlan(
			const CArray<DataType> &in,
			CArray<DataType> &out,
			const std::vector<unsigned int> &ranges
		) : Plan<DataType>(in, out, ranges, -1){}
	};

	/** Plan for executing inverse Fourier-transform. */
	template<typename DataType>
	class InversePlan : public Plan<DataType>{
	public:
		/** Constructor. */
		InversePlan(
			const CArray<DataType> &in,
			CArray<DataType> &out,
			const std::vector<unsigned int> &ranges
		) : Plan<DataType>(in, out, ranges, 1
		){}
	};

	/** N-dimensional complex Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param ranges The dimensions of the data.
	 *  @param sign The sign to use in the exponent of the Fourier
	 *  transform. */
	static void transform(
		const CArray<std::complex<double>> &in,
		CArray<std::complex<double>> &out,
		const std::vector<unsigned int> &ranges,
		int sign
	);

	/** Execute a planned transform.
	 *
	 *  @param plan The plan to execute. */
	template<typename DataType>
	static void transform(Plan<DataType> &plan);

	/** N-dimensional complex forward Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param ranges The dimensions of the data. */
	static void forward(
		const CArray<std::complex<double>> &in,
		CArray<std::complex<double>> &out,
		const std::vector<unsigned int> &ranges
	);

	/** N-dimensional complex inverse Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param ranges The dimensions of the data. */
	static void inverse(
		const CArray<std::complex<double>> &in,
		CArray<std::complex<double>> &out,
		const std::vector<unsigned int> &ranges
	);

private:
};

template<typename DataType>
inline void FourierTransform::transform(Plan<DataType> &plan){
	fftw_execute(plan.getFFTWPlan());

	double normalizationFactor = plan.getNormalizationFactor();
	if(normalizationFactor != 1.){
		CArray<DataType> &output = plan.getOutput();
		for(unsigned int n = 0; n < plan.getSize(); n++)
			output[n] /= normalizationFactor;
	}
}

inline void FourierTransform::forward(
	const CArray<std::complex<double>> &in,
	CArray<std::complex<double>> &out,
	const std::vector<unsigned int> &ranges
){
	transform(in, out, ranges, -1);
}

inline void FourierTransform::inverse(
	const CArray<std::complex<double>> &in,
	CArray<std::complex<double>> &out,
	const std::vector<unsigned int> &ranges
){
	transform(in, out, ranges, 1);
}

template<typename DataType>
inline FourierTransform::Plan<DataType>::Plan(Plan &&plan){
	this->plan = plan.plan;
	plan.plan = nullptr;

	normalizationFactor = plan.normalizationFactor;
	size = plan.size;
	input = plan.input;
	output = plan.output;
}

template<typename DataType>
inline FourierTransform::Plan<DataType>::~Plan(){
	if(plan != nullptr){
		#pragma omp critical (TBTK_FOURIER_TRANSFORM)
		fftw_destroy_plan(*plan);

		delete plan;
	}

}

template<typename DataType>
inline FourierTransform::Plan<DataType>& FourierTransform::Plan<
	DataType
>::operator=(Plan &&rhs){
	if(this != &rhs){
		if(this->plan != nullptr){
			#pragma omp critical (TBTK_FOURIER_TRANSFORM)
			fftw_destroy_plan(*this->plan);

			delete this->plan;

			this->plan = rhs.plan;

			normalizationFactor = rhs.normalizationFactor;
			size = rhs.size;
			input = rhs.input;
			output = rhs.output;
		}
	}

	return *this;
}

template<typename DataType>
inline void FourierTransform::Plan<DataType>::setNormalizationFactor(
	double normalizationFactor
){
	this->normalizationFactor = normalizationFactor;
}

template<typename DataType>
inline double FourierTransform::Plan<DataType>::getNormalizationFactor() const{
	return normalizationFactor;
}

template<typename DataType>
inline fftw_plan& FourierTransform::Plan<DataType>::getFFTWPlan(){
	return *plan;
}

template<typename DataType>
inline unsigned int FourierTransform::Plan<DataType>::getSize() const{
	return size;
}

template<typename DataType>
inline CArray<DataType>& FourierTransform::Plan<DataType>::getInput(){
	return input;
}

template<typename DataType>
inline CArray<DataType>& FourierTransform::Plan<DataType>::getOutput(){
	return output;
}

};	//End of namespace TBTK

#endif
