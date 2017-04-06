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
 *  @file Matrix.h
 *  @brief Custom matrix.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SPIN_MATRIX
#define COM_DAFER45_TBTK_SPIN_MATRIX

#include "Matrix.h"

#include <complex>

namespace TBTK{

class SpinMatrix : public Matrix<std::complex<double>, 2, 2>{
public:
	/** Constructor. */
	SpinMatrix();

	/** Destructor. */
	~SpinMatrix();

	/** Assignment operator for assigning a single value to every element
	 *  of the matrix. */
	SpinMatrix& operator=(std::complex<double> value);
private:
};

};	//End namespace TBTK

#endif
