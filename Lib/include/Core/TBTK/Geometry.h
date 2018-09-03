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
 *  @file Geometry.h
 *  @brief Contains geometric inforamtion about a Model.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_GEOMETRY
#define COM_DAFER45_TBTK_GEOMETRY

#include "TBTK/IndexedDataTree.h"
#include "TBTK/Serializable.h"
#include "TBTK/SerializableVector.h"

namespace TBTK{
	class FileReader;

/** @brief Contains geometric information about a Model. */
class Geometry : public Serializable{
public:
	/** Constructs a Geomerty. */
	Geometry();

	/** Constructs a Geometry from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Geometry.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Geometry(
		const std::string &serialization,
		Mode mode
	);

	/** Destructor. */
	virtual ~Geometry();

	/** Set a coordinate for an index. The first added coordinate can have
	 *  arbitrary number of dimensions, while the rest has to agree with
	 *  the number of dimensions of the first.
	 *
	 *  @param index The Index for which to set the coordinate.
	 *  @param coordinate The coordinate. */
	void setCoordinate(
		const Index &index,
		const std::vector<double> &coordinate
	);

	/** Get the coordinate for a given Index.
	 *
	 *  @param index The Index to get the coordinate for.
	 *  @return The coordinate for the given Index. */
	const std::vector<double>& getCoordinate(const Index &index) const;

	/** Get the number of dimensions of the space. If no coordinate has
	 *  been set yet, the returned value is -1.
	 *
	 *  @return The number of dimensions. */
	int getDimensions() const;

	/** Translate all coordinates.
	 *
	 *  @param translation The translation to be applied. */
	void translate(const std::vector<double> &translation);

	/** Implements Serializable::serialize. */
	std::string serialize(Mode mode) const;
private:
	/** Number of spatial dimensions. */
	int dimensions;

	/** Spatial dimensions. */
	IndexedDataTree<SerializableVector<double>> coordinates;

	/** FileReader is a friend class to allow it to write Geometry data. */
	friend class FileReader;
};

inline void Geometry::setCoordinate(
	const Index &index,
	const std::vector<double> &coordinate
){
	if(dimensions == -1)
		dimensions = coordinate.size();

	TBTKAssert(
		coordinate.size() == (unsigned int)dimensions,
		"Geometry::setCoordinate()",
		"Incompatible dimensions. A coordinate with dimension '"
		<< dimensions << "' has already been added, which is"
		<< " incompatible with the current coordinate dimension of '"
		<< coordinate.size() << "'.",
		""
	);

	coordinates.add(coordinate, index);
}

inline const std::vector<double>& Geometry::getCoordinate(
	const Index &index
) const{
	return coordinates.get(index);
}

inline int Geometry::getDimensions() const{
	return dimensions;
}

};	//End of namespace TBTK

#endif
