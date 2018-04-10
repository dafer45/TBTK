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

	/** Get all @link SourceAmplitude SourceAmplitudes @endlink with the
	 *  given Index.
	 *
	 *  @param index Index to get @link SourceAmplitude SourceAmplitudes
	 *  @endlink for.
	 *
	 *  @return All @link SourceAmplitude SourceAmplitudes @endlink for the
	 *  given Index. */
	const std::vector<SourceAmplitude>& get(
		const Index &index
	) const;

	class Iterator;
	class ConstIterator;
private:
	/** Base class for Iterator and ConstIterator for iterating through the
	 *  @link SourceAmplitude SourceAmplitudes @endlink. */
	template<bool isConstIterator>
	class _Iterator{
	public:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type.*/
		typedef typename std::conditional<
			isConstIterator,
			const SourceAmplitude&,
			SourceAmplitude&
		>::type SourceAmplitudeReferenceType;

		/** Increment operator. */
		void operator++();

		/** Dereference operator. */
		SourceAmplitudeReferenceType operator*();

		/** Equality operator. */
		bool operator==(const _Iterator &rhs);

		/** Inequality operator. */
		bool operator!=(const _Iterator &rhs);
	private:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			IndexedDataTree<
				std::vector<SourceAmplitude>
			>::ConstIterator,
			IndexedDataTree<
				std::vector<SourceAmplitude>
			>::Iterator
		>::type IteratorType;

		/** Current source amplitude at the current Index. */
		unsigned int currentSourceAmplitude;

		/** Iterator iterating through the sourceAmplitudeTree. */
		IteratorType iterator;

		/** Pointer to the end ot the sourceAmplitudeTree. */
		IteratorType iteratorEnd;

		/** The iterator can only be constructed by the
		 *  SourceAmplitudeSet. */
		friend class Iterator;
		friend class ConstIterator;

		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			const IndexedDataTree<
				std::vector<SourceAmplitude>
			>,
			IndexedDataTree<
				std::vector<SourceAmplitude>
			>
		>::type SourceAmplitudeTreeType;

		/** Private constructor. Limits the ability to construct the
		 *  iterator to the SourceAmplitudeSet. */
		_Iterator(
			SourceAmplitudeTreeType &sourceAmplitudeTree,
			bool end = false
		);
	};
public:
	/** Iterator for iterating through the @link SourceAmplitude
	 *  SourceAmplitudes @endlink. */
	class Iterator : public _Iterator<false>{
	private:
		Iterator(
			IndexedDataTree<
				std::vector<SourceAmplitude>
			> &sourceAmplitudeTree,
			bool end = false
		) : _Iterator(sourceAmplitudeTree, end){};

		/** Make the SourceAmplitudeSet able to construct an Iterator.
		 */
		friend class SourceAmplitudeSet;
	};

	/** Iterator for iterating through the @link SourceAmplitude
	 *  SourceAmplitudes @endlink. */
	class ConstIterator : public _Iterator<true>{
	private:
		ConstIterator(
			const IndexedDataTree<
				std::vector<SourceAmplitude>
			> &sourceAmplitudeTree,
			bool end = false
		) : _Iterator(sourceAmplitudeTree, end){};

		/** Make the SourceAmplitudeSet able to construct an Iterator.
		 */
		friend class SourceAmplitudeSet;
	};

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  SourceAmplitudeSet. */
	SourceAmplitudeSet::Iterator begin();

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  SourceAmplitudeSet. */
	SourceAmplitudeSet::ConstIterator begin() const;

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  SourceAmplitudeSet. */
	SourceAmplitudeSet::ConstIterator cbegin() const;

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the SourceAmplitudeSet.
	 */
	SourceAmplitudeSet::Iterator end();

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the SourceAmplitudeSet.
	 */
	SourceAmplitudeSet::ConstIterator end() const;

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the SourceAmplitudeSet.
	 */
	SourceAmplitudeSet::ConstIterator cend() const;

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

inline const std::vector<SourceAmplitude>& SourceAmplitudeSet::get(
	const Index &index
) const{
	return sourceAmplitudeTree.get(index);
}

inline unsigned int SourceAmplitudeSet::getSizeInBytes() const{
	return sizeof(this) - sizeof(sourceAmplitudeTree)
		+ sourceAmplitudeTree.getSizeInBytes();
}

template<bool isConstIterator>
SourceAmplitudeSet::_Iterator<isConstIterator>::_Iterator(
	SourceAmplitudeTreeType &sourceAmplitudeTree,
	bool end
) :
        currentSourceAmplitude(0),
        iterator(
		end ? sourceAmplitudeTree.end() : sourceAmplitudeTree.begin()
	),
        iteratorEnd(sourceAmplitudeTree.end())
{
}

template<bool isConstIterator>
void SourceAmplitudeSet::_Iterator<isConstIterator>::operator++(){
	if(iterator != iteratorEnd){
		const std::vector<SourceAmplitude> &sourceAmplitudes = *iterator;
		if(currentSourceAmplitude+1 == sourceAmplitudes.size()){
			currentSourceAmplitude = 0;
			++iterator;
		}
		else{
			currentSourceAmplitude++;
		}
	}
}

template<bool isConstIterator>
typename SourceAmplitudeSet::_Iterator<isConstIterator>::SourceAmplitudeReferenceType
SourceAmplitudeSet::_Iterator<isConstIterator>::operator*(){
	return (*iterator)[currentSourceAmplitude];
}

template<bool isConstIterator>
bool SourceAmplitudeSet::_Iterator<isConstIterator>::operator==(
	const _Iterator<isConstIterator> &rhs
){
	if(
		iterator == rhs.iterator
		&& currentSourceAmplitude == rhs.currentSourceAmplitude
	){
		return true;
	}
	else{
		return false;
	}
}

template<bool isConstIterator>
bool SourceAmplitudeSet::_Iterator<isConstIterator>::operator!=(
	const _Iterator<isConstIterator> &rhs
){
	if(
		iterator != rhs.iterator
		|| currentSourceAmplitude != rhs.currentSourceAmplitude
	){
		return true;
	}
	else{
		return false;
	}
}

};	//End of namespace TBTK

#endif
