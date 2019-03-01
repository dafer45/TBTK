/* Copyright 2019 Kristofer Björnson
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
 *  @file OverlapAmplitudeSet.h
 *  @brief OverlapAmplitude container.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_OVERLAP_AMPLITUDE_SET
#define COM_DAFER45_TBTK_OVERLAP_AMPLITUDE_SET

#include "TBTK/OverlapAmplitude.h"
#include "TBTK/IndexedDataTree.h"
#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

namespace TBTK{

/** @brief OverlapAmplitude container.
 *
 *  An OverlapAmplitudeSet is a container for @link OverlapAmplitude
 *  OverlapAmplitudes @endlink. */
class OverlapAmplitudeSet : public Serializable{
public:
	/** Constructor. */
	OverlapAmplitudeSet();

	/** Constructor. Constructs the OverlapAmplitudeSet from a
	 *  serialization string.
	 *
	 *  @param serialization Serialization from which to construct the
	 *  OverlapAmplitudeSet.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	OverlapAmplitudeSet(const std::string &serializeation, Mode mode);

	/** Destructor. */
	virtual ~OverlapAmplitudeSet();

	/** Add a single OverlapAmplitude.
	 *
	 *  @param overlapAmplitude OverlapAmplitude to add. */
	void add(const OverlapAmplitude &overlapAmplitude);

	/** Get the OverlapAmplitude with the given Indices.
	 *
	 *  @param braIndex bra-Index to get the OverlapAmplitude for.
	 *  @param ketIndex ket-Index to get the OverlapAmplitude for.
	 *
	 *  @return The OverlapAmplitude for the given bra- and ket-Indices. */
	OverlapAmplitude& get(
		const Index &braIndex,
		const Index &ketIndex
	);

	/** Get the OverlapAmplitude for the given Indices.
	 *
	 *  @param braIndex bra-Index to get the OverlapAmplitude for.
	 *  @param ketIndex ket-Index to get the OverlapAmplitude for.
	 *
	 *  @return The OverlapAmplitude for the given bra- and ket-Indices. */
	const OverlapAmplitude& get(
		const Index &braIndex,
		const Index &ketIndex
	) const;

	/** Get whether an orthonormal basis should be assumed. This is true if
	 *  no OverlapAmplitude has been added to the OverlapAmplitudeSet.
	 *
	 *  @return True if no OverlapAmplitude has been added to the
	 *  OverlapAmplitudeSet. */
	bool getAssumeOrthonormalBasis() const;

	class Iterator;
	class ConstIterator;
private:
	/** Base class for Iterator and ConstIterator for iterating through the
	 *  @link OverlapAmplitude OverlapAmplitudes @endlink. */
	template<bool isConstIterator>
	class _Iterator{
	public:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type.*/
		typedef typename std::conditional<
			isConstIterator,
			const OverlapAmplitude&,
			OverlapAmplitude&
		>::type OverlapAmplitudeReferenceType;

		/** Increment operator. */
		void operator++();

		/** Dereference operator. */
		OverlapAmplitudeReferenceType operator*();

		/** Equality operator. */
		bool operator==(const _Iterator &rhs);

		/** Inequality operator. */
		bool operator!=(const _Iterator &rhs);
	private:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			IndexedDataTree<OverlapAmplitude>::ConstIterator,
			IndexedDataTree<OverlapAmplitude>::Iterator
		>::type IteratorType;

		/** Iterator iterating through the overlapAmplitudeTree. */
		IteratorType iterator;

		/** Pointer to the end of the overlapAmplitudeTree. */
		IteratorType iteratorEnd;

		/** The iterator can only be constructed by the
		 *  OverlapAmplitudeSet. */
		friend class Iterator;
		friend class ConstIterator;

		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			const IndexedDataTree<OverlapAmplitude>,
			IndexedDataTree<OverlapAmplitude>
		>::type OverlapAmplitudeTreeType;

		/** Private constructor. Limits the ability to construct the
		 *  iterator to the OverlapAmplitudeSet. */
		_Iterator(
			OverlapAmplitudeTreeType &overlapAmplitudeTree,
			bool end = false
		);
	};
public:
	/** Iterator for iterating through the @link OverlapAmplitude
	 *  OverlapAmplitudes @endlink. */
	class Iterator : public _Iterator<false>{
	private:
		Iterator(
			IndexedDataTree<
				OverlapAmplitude
			> &overlapAmplitudeTree,
			bool end = false
		) : _Iterator(overlapAmplitudeTree, end){};

		/** Make the OverlapAmplitudeSet able to construct an Iterator.
		 */
		friend class OverlapAmplitudeSet;
	};

	/** Iterator for iterating through the @link OverlapAmplitude
	 *  OverlapAmplitudes @endlink. */
	class ConstIterator : public _Iterator<true>{
	private:
		ConstIterator(
			const IndexedDataTree<
				OverlapAmplitude
			> &overlapAmplitudeTree,
			bool end = false
		) : _Iterator(overlapAmplitudeTree, end){};

		/** Make the OverlapAmplitudeSet able to construct an Iterator.
		 */
		friend class OverlapAmplitudeSet;
	};

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  OverlapAmplitudeSet. */
	OverlapAmplitudeSet::Iterator begin();

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  OverlapAmplitudeSet. */
	OverlapAmplitudeSet::ConstIterator begin() const;

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  OverlapAmplitudeSet. */
	OverlapAmplitudeSet::ConstIterator cbegin() const;

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the OverlapAmplitudeSet.
	 */
	OverlapAmplitudeSet::Iterator end();

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the OverlapAmplitudeSet.
	 */
	OverlapAmplitudeSet::ConstIterator end() const;

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the OverlapAmplitudeSet.
	 */
	OverlapAmplitudeSet::ConstIterator cend() const;

	/** Implements Serializable::serialize(). */
	virtual std::string serialize(Mode mode) const;

	/** Get size in bytes. */
	unsigned int getSizeInBytes() const;
private:
	/** Container for the OverlaAmplitudes. */
	IndexedDataTree<OverlapAmplitude> overlapAmplitudeTree;

	/** Flag indicating whether an orthogonal basis should be assumed. This
	 *  is true until a first OverlapAmplitude is added to the
	 *  OverlapAmplitudeSet. */
	bool assumeOrthonormalBasis;
};

inline void OverlapAmplitudeSet::add(
	const OverlapAmplitude &overlapAmplitude
){
	overlapAmplitudeTree.add(
		overlapAmplitude,
		{
			overlapAmplitude.getBraIndex(),
			overlapAmplitude.getKetIndex()
		}
	);

	assumeOrthonormalBasis = false;
}

inline OverlapAmplitude& OverlapAmplitudeSet::get(
	const Index &braIndex,
	const Index &ketIndex
){
	return overlapAmplitudeTree.get({braIndex, ketIndex});
}

inline const OverlapAmplitude& OverlapAmplitudeSet::get(
	const Index &braIndex,
	const Index &ketIndex
) const{
	return overlapAmplitudeTree.get({braIndex, ketIndex});
}

inline bool OverlapAmplitudeSet::getAssumeOrthonormalBasis() const{
	return assumeOrthonormalBasis;
}

inline unsigned int OverlapAmplitudeSet::getSizeInBytes() const{
	return sizeof(this) - sizeof(overlapAmplitudeTree)
		+ overlapAmplitudeTree.getSizeInBytes();
}

template<bool isConstIterator>
OverlapAmplitudeSet::_Iterator<isConstIterator>::_Iterator(
	OverlapAmplitudeTreeType &overlapAmplitudeTree,
	bool end
) :
	iterator(
		end ? overlapAmplitudeTree.end() : overlapAmplitudeTree.begin()
	),
	iteratorEnd(overlapAmplitudeTree.end())
{
}

template<bool isConstIterator>
void OverlapAmplitudeSet::_Iterator<isConstIterator>::operator++(){
	if(iterator != iteratorEnd)
		++iterator;
}

template<bool isConstIterator>
typename OverlapAmplitudeSet::_Iterator<
	isConstIterator
>::OverlapAmplitudeReferenceType
OverlapAmplitudeSet::_Iterator<isConstIterator>::operator*(){
	return *iterator;
}

template<bool isConstIterator>
bool OverlapAmplitudeSet::_Iterator<isConstIterator>::operator==(
	const _Iterator<isConstIterator> &rhs
){
	if(iterator == rhs.iterator)
		return true;
	else
		return false;
}

template<bool isConstIterator>
bool OverlapAmplitudeSet::_Iterator<isConstIterator>::operator!=(
	const _Iterator<isConstIterator> &rhs
){
	if(iterator != rhs.iterator)
		return true;
	else
		return false;
}

};	//End of namespace TBTK

#endif
