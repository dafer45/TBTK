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
 *  @file Gauge.h
 *  @brief Generalized gauge transformation.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GAUGE
#ifndef COM_DAFER45_TBTK_GAUGE

#include <complex>

namespace TBTK{

class Gauge{
public:
	/** Constructor. */
	Gauge();

	/** Destructor. */
	~Gauge();
private:
	/** Number of HoppingAmplitudes before the Gauge transformation. */
	int numOriginalHoppingAmplitudes;

	/** Signs to apply to coefficients. */
	int *signs;

	/** Flags indicating whether or not to apply complex conjugation to coefficients. */
	bool *conjugations;

	/** Column indices for transformations. */
	int *unitaryColIndices;

	/** Row indices for transformation transformation. */
	int *unitaryRowIndices;

	/** Values of the unitary transformation. */
	complex<double> *unitaryValues;
};

};	//End of namespace TBTK

#endif
