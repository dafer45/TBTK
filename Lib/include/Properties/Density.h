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
 *  @file Density.h
 *  @brief Property container for density.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DENSITY
#define COM_DAFER45_TBTK_DENSITY

#include "AbstractProperty.h"
#include "IndexTree.h"

namespace TBTK{
namespace Property{

/** @brief Property container for density. */
class Density : public AbstractProperty<double>{
public:
	/** Constructor. */
	Density(int dimensions, const int *ranges);

	/** Constructor. */
	Density(int dimensions, const int *ranges, const double *data);

	/** Constructor. */
	Density(const IndexTree &indexTree);

	/** Constructor. */
	Density(const IndexTree &indexTree, const double *data);

	/** Copy constructor. */
	Density(const Density &density);

	/** Move constructor. */
	Density(Density &&density);

	/** Constructor. Constructs the Density from a serializeation string.
	 */
	Density(const std::string &serialization, Mode mode);

	/** Destructor. */
	~Density();

	/** Assignment operator. */
	Density& operator=(const Density &rhs);

	/** Move assignment operator. */
	Density& operator=(Density &&rhs);

	/** Get min value. */
	double getMin() const;

	/** Get max value. */
	double getMax() const;

	/** Overrider AbstractProperty::serialize(). */
	virtual std::string serialize(Mode mode) const;
private:
};

};	//End namespace Property
};	//End namespace TBTK

#endif
