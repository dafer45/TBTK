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
			DataType *in,
			DataType *out,
			int sizeX,
			int sign
		);

		/** Constructor. */
		Plan(
			DataType *in,
			DataType *out,
			int sizeX,
			int sizeY,
			int sign
		);

		/** Constructor. */
		Plan(
			DataType *in,
			DataType *out,
			int sizeX,
			int sizeY,
			int sizeZ,
			int sign
		);

		/** Constructor. */
		Plan(
			DataType *in,
			DataType *out,
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
		DataType *input;

		/** Output data. */
		DataType *output;

		/** Get FFTW3 plan. */
		fftw_plan& getFFTWPlan();

		/** Get data size. */
		unsigned int getSize() const;

		/** Get input data. */
		DataType* getInput();

		/** Get output data. */
		DataType* getOutput();

		/** Make FourierTransform a friend class. */
		friend class FourierTransform;
	};

	/** Plan for executing forward Fourier-transform. */
	template<typename DataType>
	class ForwardPlan : public Plan<DataType>{
	public:
		/** Constructor. */
		ForwardPlan(
			DataType *in,
			DataType *out,
			int sizeX
		) : Plan<DataType>(
			in,
			out,
			sizeX,
			-1
		){}

		/** Constructor. */
		ForwardPlan(
			DataType *in,
			DataType *out,
			int sizeX,
			int sizeY
		) : Plan<DataType>(
			in,
			out,
			sizeX,
			sizeY,
			-1
		){}

		/** Constructor. */
		ForwardPlan(
			DataType *in,
			DataType *out,
			int sizeX,
			int sizeY,
			int sizeZ
		) : Plan<DataType>(
			in,
			out,
			sizeX,
			sizeY,
			sizeZ,
			-1
		){}

		/** Constructor. */
		ForwardPlan(
			DataType *in,
			DataType *out,
			const std::vector<unsigned int> &ranges
		) : Plan<DataType>(
			in,
			out,
			ranges,
			-1
		){}
	};

	/** Plan for executing inverse Fourier-transform. */
	template<typename DataType>
	class InversePlan : public Plan<DataType>{
	public:
		/** Constructor. */
		InversePlan(
			DataType *in,
			DataType *out,
			int sizeX
		) : Plan<DataType>(
			in,
			out,
			sizeX,
			1
		){}

		/** Constructor. */
		InversePlan(
			DataType *in,
			DataType *out,
			int sizeX,
			int sizeY
		) : Plan<DataType>(
			in,
			out,
			sizeX,
			sizeY,
			1
		){}

		/** Constructor. */
		InversePlan(
			DataType *in,
			DataType *out,
			int sizeX,
			int sizeY,
			int sizeZ
		) : Plan<DataType>(
			in,
			out,
			sizeX,
			sizeY,
			sizeZ,
			1
		){}

		/** Constructor. */
		InversePlan(
			DataType *in,
			DataType *out,
			const std::vector<unsigned int> &ranges
		) : Plan<DataType>(
			in,
			out,
			ranges,
			1
		){}
	};

	/** One-dimensional complex Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The size of the data.
	 *  @param sign The sign to use in the exponent of the Fourier
	 *  transform. */
	static void transform(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sign
	);

	/** Two-dimensional complex Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The range of the first dimension.
	 *  @param sizeY The range of the second dimension.
	 *  @param sign The sign to use in the exponent of the Fourier
	 *  transform. */
	static void transform(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY,
		int sign
	);

	/** Three-dimensional complex Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The range of the first dimension.
	 *  @param sizeY The range of the second dimension.
	 *  @param sizeZ The range of the third dimension.
	 *  @param sign The sign to use in the exponent of the Fourier
	 *  transform. */
	static void transform(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY,
		int sizeZ,
		int sign
	);

	/** N-dimensional complex Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param ranges The dimensions of the data.
	 *  @param sign The sign to use in the exponent of the Fourier
	 *  transform. */
	static void transform(
		std::complex<double> *in,
		std::complex<double> *out,
		const std::vector<unsigned int> &ranges,
		int sign
	);

	/** Execute a planned transform.
	 *
	 *  @param plan The plan to execute. */
	template<typename DataType>
	static void transform(Plan<DataType> &plan);

	/** One-dimensional complex forward Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The size of the data. */
	static void forward(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX
	);

	/** Two-dimensional complex forward Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The range of the first dimension.
	 *  @param sizeY The range of the second dimension. */
	static void forward(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY
	);

	/** Three-dimensional complex forward Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The range of the first dimension.
	 *  @param sizeY The range of the second dimension.
	 *  @param sizeZ The range of the third dimension. */
	static void forward(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY,
		int sizeZ
	);

	/** N-dimensional complex forward Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param ranges The dimensions of the data. */
	static void forward(
		std::complex<double> *in,
		std::complex<double> *out,
		const std::vector<unsigned int> &ranges
	);

	/** One-dimensional complex inverse Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The size of the data. */
	static void inverse(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX
	);

	/** Two-dimensional complex inverse Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The range of the first dimension.
	 *  @param sizeY The range of the second dimension. */
	static void inverse(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY
	);

	/** Three-dimensional complex inverse Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param sizeX The range of the first dimension.
	 *  @param sizeY The range of the second dimension.
	 *  @param sizeZ The range of the third dimension. */
	static void inverse(
		std::complex<double> *in,
		std::complex<double> *out,
		int sizeX,
		int sizeY,
		int sizeZ
	);

	/** N-dimensional complex inverse Fourier transform.
	 *
	 *  @param in Pointer to array containing the input.
	 *  @param out Pointer to array that will contain the output.
	 *  @param ranges The dimensions of the data. */
	static void inverse(
		std::complex<double> *in,
		std::complex<double> *out,
		const std::vector<unsigned int> &ranges
	);

private:
};

template<typename DataType>
inline void FourierTransform::transform(Plan<DataType> &plan){
	fftw_execute(plan.getFFTWPlan());

	double normalizationFactor = plan.getNormalizationFactor();
	if(normalizationFactor != 1.){
		DataType *output = plan.getOutput();
		for(unsigned int n = 0; n < plan.getSize(); n++)
			output[n] /= normalizationFactor;
	}
}

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

inline void FourierTransform::forward(
	std::complex<double> *in,
	std::complex<double> *out,
	const std::vector<unsigned int> &ranges
){
	transform(in, out, ranges, -1);
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

inline void FourierTransform::inverse(
	std::complex<double> *in,
	std::complex<double> *out,
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
inline DataType* FourierTransform::Plan<DataType>::getInput(){
	return input;
}

template<typename DataType>
inline DataType* FourierTransform::Plan<DataType>::getOutput(){
	return output;
}

};	//End of namespace TBTK

#endif
