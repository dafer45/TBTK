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

namespace TBTK{
	class FileReader;

/** @brief Contains geometric information about a Model. */
class Geometry : public Serializable{
public:
	/** Constructs a Geomerty. */
	Geometry();

	Geometry(
		const std::string &serialization,
		Mode mode
	);

	virtual ~Geometry();

	/** Add a coordinate for an index. */
	void setCoordinates(
		const Index &index,
		std::initializer_list<double> coordinates
	);

	/** Add a coordinate for an index. */
	void setCoordinates(
		const Index &index,
		const std::vector<double> &coordinates
	);

	/** Get dimensions. */
	int getDimensions() const;

	/** Get Coordinates using a physical index. */
	const std::vector<double>& getCoordinates(const Index &index) const;

	/** Translate all coordinates. */
	void translate(const std::vector<double> &translation);

	/** Implements Serializable::serialize. */
	std::string serialize(Mode mode) const;
private:
	/** Number of spatial dimensions. */
	int dimensions;

	/** Spatial dimensions. */
	IndexedDataTree<std::vector<double>> coordinates;

	/** FileReader is a friend class to allow it to write Geometry data. */
	friend class FileReader;
};

inline int Geometry::getDimensions() const{
	return dimensions;
}

inline const std::vector<double>& Geometry::getCoordinates(
	const Index &index
) const{
	return coordinates.get(index);
}

};	//End of namespace TBTK

#endif
