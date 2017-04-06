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
 *  @file Magnetization.h
 *  @brief Property container for magnetization
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MAGNETIZATION
#define COM_DAFER45_TBTK_MAGNETIZATION

#include "AbstractProperty.h"
#include "SpinMatrix.h"

#include <complex>

namespace TBTK{
namespace Property{

/** Container for magnetization. */
class Magnetization : public AbstractProperty<SpinMatrix>{
public:
	/** Constructor. */
	Magnetization(int dimensions, const int* ranges);

	/** Constructor. */
	Magnetization(
		int dimensions,
		const int* ranges,
		const SpinMatrix *data
	);

	/** Constructor. */
	Magnetization(const IndexTree &indexTree);

	/** Constructor. */
	Magnetization(
		const IndexTree &indexTree,
		const SpinMatrix *data
	);

	/** Copy constructor. */
	Magnetization(const Magnetization &magnetization);

	/** Move constructor. */
	Magnetization(Magnetization &&magnetization);

	/** Destructor. */
	~Magnetization();

	/** Assignment operator. */
	Magnetization& operator=(const Magnetization &magnetization);

	/** Move assignment operator. */
	Magnetization& operator=(Magnetization &&magnetization);
private:
};

};	//End namespace Property
};	//End namespace TBTK

#endif
