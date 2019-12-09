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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file BasisStateSet.h
 *  @brief Basis state container.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_BASIS_STATE_SET
#define COM_DAFER45_TBTK_BASIS_STATE_SET

#include "TBTK/AbstractState.h"
#include "TBTK/IndexedDataTree.h"
#include "TBTK/Serializable.h"
#include "TBTK/TBTKMacros.h"

#include <vector>

namespace TBTK{

/** @brief Basis state container.
 *
 *  A BasisStateSet is a container for instances of AbstractState. */
class BasisStateSet : public Serializable{
public:
	/** Constructor. */
	BasisStateSet();

	/** Constructor. Constructs the BasisStateSet from a serialization
	 *  string.
	 *
	 *  @param serialization Serialization from which to construct the
	 *  BasisStateSet.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	BasisStateSet(const std::string &serializeation, Mode mode);

	/** Destructor. */
	virtual ~BasisStateSet();

	/** Add a single basis state.
	 *
	 *  @param state Basis state to add. */
	void add(const AbstractState &state);

	/** Get the state with the given Index.
	 *
	 *  @param index Index to get the state for.
	 *
	 *  @return The state with the given Index. */
	AbstractState& get(const Index &index);

	/** Get the state with the given Index.
	 *
	 *  @param index Index to get the state for.
	 *
	 *  @return The state with the given Index. */
	const AbstractState& get(const Index &index) const;

	class Iterator;
	class ConstIterator;
private:
	/** Base class for Iterator and ConstIterator for iterating through the
	 *  basis states. */
	template<bool isConstIterator>
	class _Iterator{
	public:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type.*/
		typedef typename std::conditional<
			isConstIterator,
			const AbstractState&,
			AbstractState&
		>::type BasisStateReferenceType;

		/** Increment operator. */
		void operator++();

		/** Dereference operator. */
		BasisStateReferenceType operator*();

		/** Equality operator. */
		bool operator==(const _Iterator &rhs);

		/** Inequality operator. */
		bool operator!=(const _Iterator &rhs);
	private:
		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			IndexedDataTree<AbstractState*>::ConstIterator,
			IndexedDataTree<AbstractState*>::Iterator
		>::type IteratorType;

		/** Iterator iterating through the basisStateTree. */
		IteratorType iterator;

		/** Pointer to the end ot the basisStateTree. */
		IteratorType iteratorEnd;

		/** The iterator can only be constructed by the
		 *  BasisStateSet. */
		friend class Iterator;
		friend class ConstIterator;

		/** Typedef to allow for pointers to const and non-const
		 *  depending on Iterator type. */
		typedef typename std::conditional<
			isConstIterator,
			const IndexedDataTree<AbstractState*>,
			IndexedDataTree<AbstractState*>
		>::type BasisStateTreeType;

		/** Private constructor. Limits the ability to construct the
		 *  iterator to the SourceAmplitudeSet. */
		_Iterator(
			BasisStateTreeType &basisStateTree,
			bool end = false
		);
	};
public:
	/** Iterator for iterating through the basis states. */
	class Iterator : public _Iterator<false>{
	private:
		Iterator(
			IndexedDataTree<AbstractState*> &basisStateTree,
			bool end = false
		) : _Iterator(basisStateTree, end){};

		/** Make the BasisStateSet able to construct an Iterator. */
		friend class BasisStateSet;
	};

	/** Iterator for iterating through the basis states. */
	class ConstIterator : public _Iterator<true>{
	private:
		ConstIterator(
			const IndexedDataTree<
				AbstractState*
			> &basisStateTree,
			bool end = false
		) : _Iterator(basisStateTree, end){};

		/** Make the BasisStateSet able to construct an Iterator. */
		friend class BasisStateSet;
	};

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  BasisStateSet. */
	BasisStateSet::Iterator begin();

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  BasisStateSet. */
	BasisStateSet::ConstIterator begin() const;

	/** Create Iterator.
	 *
	 *  @return Iterator pointing at the first element in the
	 *  BasisStateSet. */
	BasisStateSet::ConstIterator cbegin() const;

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the BasisStateSet. */
	BasisStateSet::Iterator end();

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the BasisStateSet. */
	BasisStateSet::ConstIterator end() const;

	/** Get Iterator pointing to end.
	 *
	 *  @return An Iterator pointing at the end of the BasisStateSet. */
	BasisStateSet::ConstIterator cend() const;

	/** Implements Serializable::serialize(). */
	virtual std::string serialize(Mode mode) const;

	/** Get size in bytes. */
	unsigned int getSizeInBytes() const;
private:
	/** Container for the BasisStates. */
	IndexedDataTree<AbstractState*> basisStateTree;
};

inline void BasisStateSet::add(const AbstractState &state){
	basisStateTree.add(
		state.clone(),
		state.getIndex()
	);
}

inline AbstractState& BasisStateSet::get(const Index &index){
	return *basisStateTree.get(index);
}

inline const AbstractState& BasisStateSet::get(const Index &index) const{
	return *basisStateTree.get(index);
}

inline unsigned int BasisStateSet::getSizeInBytes() const{
	TBTKNotYetImplemented("BasisStateSet::getSizeInBytes()");
}

template<bool isConstIterator>
BasisStateSet::_Iterator<isConstIterator>::_Iterator(
	BasisStateTreeType &basisStateTree,
	bool end
) :
        iterator(
		end ? basisStateTree.end() : basisStateTree.begin()
	),
        iteratorEnd(basisStateTree.end())
{
}

template<bool isConstIterator>
void BasisStateSet::_Iterator<isConstIterator>::operator++(){
	if(iterator != iteratorEnd)
		++iterator;
}

template<bool isConstIterator>
typename BasisStateSet::_Iterator<isConstIterator>::BasisStateReferenceType
BasisStateSet::_Iterator<isConstIterator>::operator*(){
	return *(*iterator);
}

template<bool isConstIterator>
bool BasisStateSet::_Iterator<isConstIterator>::operator==(
	const _Iterator<isConstIterator> &rhs
){
	if(iterator == rhs.iterator)
		return true;
	else
		return false;
}

template<bool isConstIterator>
bool BasisStateSet::_Iterator<isConstIterator>::operator!=(
	const _Iterator<isConstIterator> &rhs
){
	if(iterator != rhs.iterator)
		return true;
	else
		return false;
}

};	//End of namespace TBTK

#endif
/// @endcond
