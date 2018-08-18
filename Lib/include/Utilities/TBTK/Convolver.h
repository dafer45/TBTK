/* Copyright 2018 Kristofer Björnson
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
 *  @file Convolver.h
 *  @brief Convolves multi-dimensional arrays.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_CONVOLVER
#define COM_DAFER45_TBTK_CONVOLVER

#include "TBTK/Array.h"
#include "TBTK/FourierTransform.h"

namespace TBTK{

class Convolver{
public:
	/** Calculates the convolution \f$\sum_{x}f(x)g(y-x)\f$ of two array
	 *  \f$f\f$ and \f$g\f$.
	 *
	 *  @param array0 The array \f$f\f$ in the convolution.
	 *  @param array1 The array \f$g\f$ in the convolution.
	 *
	 *  @return The resulting array from the convolution. */
	template<typename DataType>
	static Array<DataType> convolve(
		const Array<DataType> &array0,
		const Array<DataType> &array1
	);

	/** Calculates the cross correlation \f$\sum_{x}f^{*}(x)g(x+y) of two
	 *  arrays \f$f\f$ and \f$g\f$.
	 *
	 *  @param array0 The array \f$f\f$ in the cross correlation.
	 *  @param array1 The array \f$g\f$ in the cross correlation.
	 *
	 *  @return The resulting array from the cross correlation. */
	template<typename DataType>
	static Array<DataType> crossCorrelate(
		const Array<DataType> &array0,
		const Array<DataType> &array1
	);
private:
};

template<typename DataType>
Array<DataType> Convolver::convolve(
	const Array<DataType> &array0,
	const Array<DataType> &array1
){
	const std::vector<unsigned int> &ranges0 = array0.getRanges();
	const std::vector<unsigned int> &ranges1 = array1.getRanges();
	TBTKAssert(
		ranges0.size() == ranges1.size(),
		"Convolver::convolve()",
		"Incompatible ranges. The ranges must be equal, but array0 has"
		<< " '" << ranges0.size() << "' range parameters, while array1"
		<< " has '" << ranges1.size() << "' range parameters.",
		""
	);

	for(unsigned int n = 0; n < ranges0.size(); n++){
		TBTKAssert(
			ranges0[n] == ranges1[n],
			"Convolver::convolve()",
			"Incompatible ranges. The ranges must be equal, but"
			<< " range '" << n << "' is '" << ranges0[n] << "' for"
			<< " array0, while it is '" << ranges1[n] << "' for"
			<< " array1.",
			""
		);
	}

	Array<DataType> array0Out(array0.getRanges());
	Array<DataType> array1Out(array1.getRanges());

	FourierTransform::ForwardPlan<DataType> plan0(
		array0.getData(),
		array0Out.getData(),
		array0.getRanges()
	);
	plan0.setNormalizationFactor(1);
	FourierTransform::ForwardPlan<DataType> plan1(
		array1.getData(),
		array1Out.getData(),
		array1.getRanges()
	);
	plan1.setNormalizationFactor(1);

	FourierTransform::transform(plan0);
	FourierTransform::transform(plan1);

	Array<DataType> result(array0.getRanges());
	for(unsigned int n = 0; n < result.getSize(); n++)
		result[n] = array0Out[n]*array1Out[n];

	FourierTransform::InversePlan<DataType> planResult(
		result.getData(),
		result.getData(),
		result.getRanges()
	);
	planResult.setNormalizationFactor(1);

	FourierTransform::transform(planResult);

	for(unsigned int n = 0; n < result.getSize(); n++)
		result[n] /= result.getSize();

	return result;
}

template<typename DataType>
Array<DataType> Convolver::crossCorrelate(
	const Array<DataType> &array0,
	const Array<DataType> &array1
){
	const std::vector<unsigned int> &ranges0 = array0.getRanges();
	const std::vector<unsigned int> &ranges1 = array1.getRanges();
	TBTKAssert(
		ranges0.size() == ranges1.size(),
		"Convolver::crossCorrelate()",
		"Incompatible ranges. The ranges must be equal, but array0 has"
		<< " '" << ranges0.size() << "' range parameters, while array1"
		<< " has '" << ranges1.size() << "' range parameters.",
		""
	);

	for(unsigned int n = 0; n < ranges0.size(); n++){
		TBTKAssert(
			ranges0[n] == ranges1[n],
			"Convolver::crossCorrelate()",
			"Incompatible ranges. The ranges must be equal, but"
			<< " range '" << n << "' is '" << ranges0[n] << "' for"
			<< " array0, while it is '" << ranges1[n] << "' for"
			<< " array1.",
			""
		);
	}

	Array<DataType> array0Out(array0.getRanges());
	Array<DataType> array1Out(array1.getRanges());

	FourierTransform::ForwardPlan<DataType> plan0(
		array0.getData(),
		array0Out.getData(),
		array0.getRanges()
	);
	plan0.setNormalizationFactor(1);
	FourierTransform::ForwardPlan<DataType> plan1(
		array1.getData(),
		array1Out.getData(),
		array1.getRanges()
	);
	plan1.setNormalizationFactor(1);

	FourierTransform::transform(plan0);
	FourierTransform::transform(plan1);

	Array<DataType> result(array0.getRanges());
	for(unsigned int n = 0; n < result.getSize(); n++)
		result[n] = conj(array0Out[n])*array1Out[n];

	FourierTransform::InversePlan<DataType> planResult(
		result.getData(),
		result.getData(),
		result.getRanges()
	);
	planResult.setNormalizationFactor(1);

	FourierTransform::transform(planResult);

	for(unsigned int n = 0; n < result.getSize(); n++)
		result[n] /= result.getSize();

	return result;
}

}; //End of namespace TBTK

#endif
