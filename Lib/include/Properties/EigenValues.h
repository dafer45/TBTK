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
 *  @file EigenValues.h
 *  @brief Property container for eigen values
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_EIGEN_VALUES
#define COM_DAFER45_TBTK_EIGEN_VALUES

#include "AbstractProperty.h"

namespace TBTK{
namespace Property{

/** Container for local density of states (LDOS). */
class EigenValues : public AbstractProperty<double>{
public:
	/** Constructor. */
	EigenValues(int size);

	/** Constructor. */
	EigenValues(int size, const double *data);

	/** Copy constructor. */
	EigenValues(const EigenValues &eigenValues);

	/** Move constructor. */
	EigenValues(EigenValues &&eigenValues);

	/** Destructor. */
	~EigenValues();

	/** Assignment operator. */
	EigenValues& operator=(const EigenValues &rhs);

	/** Move assignment operator. */
	EigenValues& operator=(EigenValues &&rhs);
private:
};

};	//End namespace Property
};	//End namespace TBTK

#endif
