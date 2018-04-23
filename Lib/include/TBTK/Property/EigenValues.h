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
 *  @brief Property container for eigen values.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_EIGEN_VALUES
#define COM_DAFER45_TBTK_EIGEN_VALUES

#include "TBTK/Property/AbstractProperty.h"

namespace TBTK{
namespace Property{

/** @brief Property container for eigen values.. */
class EigenValues : public AbstractProperty<double>{
public:
	/** Constructs EigenValues.
	 *
	 *  @param size The number of eigenvalues. */
	EigenValues(int size);

	/** Constructs EigenValues and initializes it with data.
	 *
	 *  @param size The number of eigenvalues.
	 *  @param data Raw data to initialize the EigenValues with. */
	EigenValues(int size, const double *data);

	/** Copy constructor. */
//	EigenValues(const EigenValues &eigenValues);

	/** Move constructor. */
//	EigenValues(EigenValues &&eigenValues);

	/** Constructor. Constructs the EigenValues from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the EigenValues.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	EigenValues(const std::string &serialization, Mode mode);

	/** Destructor. */
//	~EigenValues();

	/** Assignment operator. */
//	EigenValues& operator=(const EigenValues &rhs);

	/** Move assignment operator. */
//	EigenValues& operator=(EigenValues &&rhs);

	/** Overrides AbstractProperty::serialize(). */
	std::string serialize(Mode mode) const;
private:
};

};	//End namespace Property
};	//End namespace TBTK

#endif
