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
 *  @file SourceAmplitudeSet.h
 *  @brief SourceAmplitude container.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOURCE_AMPLITUDE_SET
#define COM_DAFER45_TBTK_SOURCE_AMPLITUDE_SET

#include "TBTK/SourceAmplitude.h"
#include "TBTK/IndexedDataTree.h"
#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

#include <vector>

namespace TBTK{

/** @brief SourceAmplitude container.
 *
 *  A SourceAmplitudeSet is a container for @link SourceAmplitude
 *  SourceAmplitudes @endlink. */
class SourceAmplitudeSet : public Serializable{
public:
	/** Constructor. */
	SourceAmplitudeSet();

	/** Constructor. Constructs the HoppingAmplitudeSet from a
	 *  serialization string.
	 *
	 *  @param serialization Serialization from which to construct the
	 *  SourceAmplitudeSet.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	SourceAmplitudeSet(const std::string &serializeation, Mode mode);

	/** Destructor. */
	virtual ~SourceAmplitudeSet();

	/** Add a single SourceAmplitude.
	 *
	 *  @param sourceAmplitude SourceAmplitude to add. */
	void add(const SourceAmplitude &sourceAmplitude);

	/** Get all @link SourceAmplitude SourceAmplitudes @endlink with the
	 *  given Index.
	 *
	 *  @param index Index to get @link SourceAmplitude SourceAmplitudes
	 *  @endlink for.
	 *
	 *  @return All @link SourceAmplitude SourceAmplitudes @endlink for the
	 *  given Index. */
	std::vector<SourceAmplitude>& get(
		const Index &index
	);

	/** Iterator for iterating through the @link SourceAmplitude
	 *  SourceAmplitudes @endlink. */
	class Iterator{
	public:
		/** Increment operator. */
		void operator++();

		/** Dereference operator. */
		SourceAmplitude& operator*();

		/** Equality operator. */
		bool operator==(const Iterator &rhs);

		/** Inequality operator. */
		bool operator!=(const Iterator &rhs);
	private:
		/** Current source amplitude at the current Index. */
		unsigned int currentSourceAmplitude;

		/** Iterator iterating through the sourceAmplitudeTree. */
		IndexedDataTree<
			std::vector<SourceAmplitude>
		>::Iterator iterator;

		/** Pointer to the end ot the sourceAmplitudeTree. */
		IndexedDataTree<
			std::vector<SourceAmplitude>
		>::Iterator iteratorEnd;

		/** The iterator can only be constructed by the
		 *  SourceAmplitudeSet. */
		friend class SourceAmplitudeSet;

		/** Private constructor. Limits the ability to construct the
		 *  iterator to the SourceAmplitudeSet. */
		Iterator(
			IndexedDataTree<
				std::vector<SourceAmplitude>
			> &sourceAmplitudeTree,
			bool end = false
		);
	};

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  SourceAmplitudeSet. */
	SourceAmplitudeSet::Iterator begin();

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the SourceAmplitudeSet.
	 */
	SourceAmplitudeSet::Iterator end();

	/** Implements Serializable::serialize(). */
	virtual std::string serialize(Mode mode) const;

	/** Get size in bytes. */
	unsigned int getSizeInBytes() const;
private:
	/** Container for the SourceAmplitudes. */
	IndexedDataTree<std::vector<SourceAmplitude>> sourceAmplitudeTree;
};

inline void SourceAmplitudeSet::add(
	const SourceAmplitude &sourceAmplitude
){
	try{
		std::vector<SourceAmplitude> &sourceAmplitudes
			= sourceAmplitudeTree.get(sourceAmplitude.getIndex());
		sourceAmplitudes.push_back(sourceAmplitude);
	}
	catch(ElementNotFoundException &e){
		sourceAmplitudeTree.add(
			std::vector<SourceAmplitude>(),
			sourceAmplitude.getIndex()
		);
		std::vector<SourceAmplitude> &sourceAmplitudes
			= sourceAmplitudeTree.get(sourceAmplitude.getIndex());
		sourceAmplitudes.push_back(sourceAmplitude);
	}
}

inline std::vector<SourceAmplitude>& SourceAmplitudeSet::get(
	const Index &index
){
	return sourceAmplitudeTree.get(index);
}

inline unsigned int SourceAmplitudeSet::getSizeInBytes() const{
	return sizeof(this) - sizeof(sourceAmplitudeTree)
		+ sourceAmplitudeTree.getSizeInBytes();
}

};	//End of namespace TBTK

#endif
