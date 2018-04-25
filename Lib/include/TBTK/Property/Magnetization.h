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
 *  @brief Property container for magnetization.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_MAGNETIZATION
#define COM_DAFER45_TBTK_MAGNETIZATION

#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/SpinMatrix.h"

#include <complex>

namespace TBTK{
namespace Property{

/** @brief Property container for magnetization. */
class Magnetization : public AbstractProperty<SpinMatrix>{
public:
	/** Constructs Magnetization on the Ranges format. [See
	 *  AbstractProperty for detailed information about the Ranges
	 *  format.]
	 *
	 *  @param dimensions Number of dimensions for the grid.
	 *  @param ranges The upper limit (exclusive) for the corresponding
	 *  dimensions. */
	Magnetization(int dimensions, const int* ranges);

	/** Constructs Magnetization on the Ranges format and initializes it
	 *  with data. [See AbstractProperty for detailed information about the
	 *  Ranges format and the raw data format.]
	 *
	 *  @param dimensions Number of dimensions for the grid.
	 *  @param ranges The upper limit (exclusive) for the corresponding
	 *  dimensions.
	 *
	 *  @param data Raw data to initialize the Magnetization with. */
	Magnetization(
		int dimensions,
		const int* ranges,
		const SpinMatrix *data
	);

	/** Constructs Magnetization on the Custom format. [See
	 *  AbstractProperty for detailed information about the Custom
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indeices
	 *  @endlink for which the Magnetization should be contained. */
	Magnetization(const IndexTree &indexTree);

	/** Constructs Magnetization on the Custom format and initializes it
	 *  with data. [See AbstractProperty for detailed information about the
	 *  Custom format and the raw data format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indeices
	 *  @endlink for which the Magnetization should be contained.
	 *
	 *  @param data Raw data to initialize the Magnetization with. */
	Magnetization(
		const IndexTree &indexTree,
		const SpinMatrix *data
	);

	/** Copy constructor. */
//	Magnetization(const Magnetization &magnetization);

	/** Move constructor. */
//	Magnetization(Magnetization &&magnetization);

	/** Constructor. Constructs the Magnetization from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Magnetization.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Magnetization(const std::string &serialization, Mode mode);

	/** Destructor. */
//	~Magnetization();

	/** Assignment operator. */
//	Magnetization& operator=(const Magnetization &magnetization);

	/** Move assignment operator. */
//	Magnetization& operator=(Magnetization &&magnetization);

	/** Overrides AbstractProperty::serialize(). */
	std::string serialize(Mode mode) const;
private:
};

};	//End namespace Property
};	//End namespace TBTK

#endif
