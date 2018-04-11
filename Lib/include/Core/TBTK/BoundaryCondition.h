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
 *  @file BoundaryCondition.h
 *  @brief A set of @link HoppingAmplitude HoppingAmplitudes @endlink, a
 *  SourceAmplitude, and an elimination Index, which together form a single
 *  linear equation that can be used to eliminate an Index from a larger set
 *  of linear equations.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BOUNDARY_CONDITION
#define COM_DAFER45_TBTK_BOUNDARY_CONDITION

#include "TBTK/HoppingAmplitude.h"
#include "TBTK/HoppingAmplitudeList.h"
#include "TBTK/Serializable.h"
#include "TBTK/SourceAmplitude.h"

#include <vector>

namespace TBTK{

/** @brief A set of @link HoppingAmplitude HoppingAmplitudes @endlink, a
 *  SourceAmplitude, and an elimination Index, which together form a single
 *  linear equation that can be used to eliminate an Index from a larger set
 *  of linear equations.
 *
 *  A BoundaryCondition consists of a set of HoppingAmplitudes and a
 *  SourceAmplitude, which together form a linear equation. I.e., if a matrix
 *  is written down using these HoppingAmplitudes and the SourceAmplitude,
 *  they form a single line in a matrix equation. The BoundaryCondition also
 *  contains an Index which indicates which Index the BoundaryCondition is
 *  supposed to eliminate in a linnear equation system. */
class BoundaryCondition : public Serializable{
public:
	/** Constructs a BoundaryCondition. */
	BoundaryCondition();

	/** Constructor. Constructs the BoundaryCondition from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the BoundaryCondition.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	BoundaryCondition(
		const std::string &serializeation,
		Serializable::Mode mode
	);

	/** Add HoppingAmplitude to the boundary condition.
	 *
	 *  @param hoppingAmplitude HoppingAmplitude to add. */
	void add(const HoppingAmplitude &hoppingAmplitude);

	/** Get a HoppingAmplitudeList containing all the @link
	 *  HoppingAmplitude HoppingAmplitudes @endlink contained in the
	 *  BoundaryCondition.
	 *
	 *  @return HoppingAmplitudeList with all contained @link
	 *  HoppingAmplitude HoppingAmplitudes @endlink. */
	const HoppingAmplitudeList& getHoppingAmplitudeList() const;

	/** Set the SourceAmplitude.
	 *
	 *  @param sourceAmplitude SourceAmplitude to use. */
	void set(const SourceAmplitude &sourceAmplitude);

	/** Get the SourceAmplitude.
	 *
	 *  @return The SourceAmplitude. */
	const SourceAmplitude& getSourceAmplitude() const;

	/** Set the Index that is to be eliminated by the boundary condition.
	 *
	 *  @param eliminationIndex Index that should be eliminated by the
	 *  boundary condition. */
	void setEliminationIndex(const Index &eliminationIndex);

	/** Get the Index that is to be eliminated by the boundary condition.
	 *
	 *  @return The Index that is to be eliminated by the boundary
	 *  condition. */
	const Index& getEliminationIndex() const;

	/** Serialize BoundaryCondition.
	 *
	 *  @param mode Serialization mode to use.
	 *
	 *  @return Serialized string representation of the BoundaryCondition.
	 */
	std::string serialize(Serializable::Mode mode) const;

	/** Get size in bytes.
	 *
	 *  @return Memory size required to store the BoundaryCondition. */
	unsigned int getSizeInBytes() const;
private:
	/** List of hopping amplitudes. */
	HoppingAmplitudeList hoppingAmplitudeList;

	/** Source amplitude. */
	SourceAmplitude sourceAmplitude;

	/** The Index that the boundary condition is to eliminate. */
	Index eliminationIndex;
};

inline BoundaryCondition::BoundaryCondition(){
}

inline void BoundaryCondition::add(const HoppingAmplitude &hoppingAmplitude){
	if(hoppingAmplitudeList.getSize() > 0){
		TBTKAssert(
			hoppingAmplitude.getToIndex().equals(
				hoppingAmplitudeList[0].getToIndex()
			),
			"BoundaryCondition::add()",
			"Incompatible to-Indices. All HoppingAmplitudes in a"
			<< " BoundaryCondition must have the same to-Index."
			<< " The added HoppingAmplitude has to-Index '"
			<< hoppingAmplitude.getToIndex().toString() << "'"
			<< " while a HoppingAmplitude with to-Index '"
			<< hoppingAmplitudeList[0].getToIndex().toString()
			<< "' already has been added.",
			""
		);
	}
	else{
		if(!sourceAmplitude.getIndex().equals({})){
			TBTKAssert(
				hoppingAmplitude.getToIndex().equals(
					sourceAmplitude.getIndex()
				),
				"BoundaryCondition::add()",
				"Incompatible Indices. The HoppingAmplitude"
				<< " must have the same to-Index as the Index"
				<< " of the SourceAmplitude. The"
				<< " HoppingAmplitude has to-Index '"
				<< hoppingAmplitude.getToIndex().toString()
				<< "' while the SourceAmplitude has Index '"
				<< sourceAmplitude.getIndex().toString() << "'.",
				""
		);
		}
	}

	hoppingAmplitudeList.add(hoppingAmplitude);
}

inline const HoppingAmplitudeList& BoundaryCondition::getHoppingAmplitudeList() const{
	return hoppingAmplitudeList;
}

inline void BoundaryCondition::set(const SourceAmplitude &sourceAmplitude){
	TBTKAssert(
		!sourceAmplitude.getIndex().equals({}),
		"BoundaryCondition::set()",
		"Invalid Index. The SourceAmplitudes Index cannot be the empty"
		<< "Index.",
		""
	);
	if(hoppingAmplitudeList.getSize() > 0){
		TBTKAssert(
			sourceAmplitude.getIndex().equals(
				hoppingAmplitudeList[0].getToIndex()
			),
			"BoundaryCondition::set()",
			"Incompatible Indices. The SourceAmplitude must have"
			<< " the same Index as the HoppingAmplitudes"
			<< " to-Indices. The SourceAmplitude has Index '"
			<< sourceAmplitude.getIndex().toString() << "' while a"
			<< " HoppingAmplitude with to-Index '"
			<< hoppingAmplitudeList[0].getToIndex().toString()
			<< "' already has been added.",
			""
		);
	}

	this->sourceAmplitude = sourceAmplitude;
}

inline const SourceAmplitude& BoundaryCondition::getSourceAmplitude() const{
	return sourceAmplitude;
}

inline void BoundaryCondition::setEliminationIndex(const Index &eliminationIndex){
	bool eliminationIndexFound = false;
	for(unsigned int n = 0; n < hoppingAmplitudeList.getSize(); n++){
		if(
			hoppingAmplitudeList[n].getFromIndex().equals(
				eliminationIndex
			)
		){
			eliminationIndexFound = true;
			break;
		}
	}
	TBTKAssert(
		eliminationIndexFound,
		"BoundaryCondition::setEliminationIndex()",
		"Invalid elimination Index. The elimination Index must be"
		<< " equal to the from-Index of at least one HoppingAmplitude"
		<< " added to the BoundaryCondition but no such from-Index was"
		<< " found.",
		"First add HoppingAmplitudes using BoundaryCondition::add()"
	);

	this->eliminationIndex = eliminationIndex;
}

inline const Index& BoundaryCondition::getEliminationIndex() const{
	return eliminationIndex;
}

inline unsigned int BoundaryCondition::getSizeInBytes() const{
	return sizeof(*this)
		+ hoppingAmplitudeList.getSizeInBytes() - sizeof(hoppingAmplitudeList)
		+ sourceAmplitude.getSizeInBytes() - sizeof(sourceAmplitude)
		+ eliminationIndex.getSizeInBytes() - sizeof(eliminationIndex);
}

};	//End of namespace TBTK

#endif
