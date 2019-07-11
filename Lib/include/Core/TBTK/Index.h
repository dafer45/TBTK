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
 *  @file Index.h
 *  @brief Flexible physical index.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEX
#define COM_DAFER45_TBTK_INDEX

#include "TBTK/Subindex.h"
#include "TBTK/Serializable.h"
#include "TBTK/Streams.h"

#include <vector>

namespace TBTK{

/** \brief Enum for special subindex values.
 *
 *  While non-negative subindices in an Index corresponds to normal subindices,
 *  negative subindices are reserved for special purposes.<br><br>
 *  <b>IDX_ALL = _a_:</b><br>
 *    Wildcard Used to indicate that all indices are to be considered or that
 *    the particular subindex value is of no interest. To improve
 *    self-documentation for library code, only IDX_ALL should be used in
 *    actuall library code. '_a_' is syntactic suggar meant for use in
 *    application code.<br><br>
 *  <b>IDX_X, IDX_Y, IDX_Z:</b><br>
 *    Loop indices used to indicate that a particular index should be looped
 *    over.<br><br>
 *  <b>IDX_SPIN:</b><br>
 *    Used to indicate that a certain subindex should be interpreted as a
 *    spin-subindex.<br><br>
 *  <b>IDX_SEPARATOR:</b><br>
 *    Used as Index-separator in compound indices such as {{1, 2}, {3, 4}},
 *    which is stored as {1, 2, IDX_SEPARATOR, 3, 4}. */
enum : int{
	//_a_ and _aX_ are shorthand notation for IDX_ALL and IDX_ALL_X. Never
	//use shorthands in library code.
	IDX_ALL		= (int)(0xC0000000 | 0x20000000),
	_a_		= IDX_ALL,
	IDX_ALL_X	= (int)(0xC0000000 | 0x10000000),
	IDX_ALL_0	= (int)(IDX_ALL_X | 0x00000000),
	IDX_ALL_1	= (int)(IDX_ALL_X | 0x00000001),
	IDX_ALL_2	= (int)(IDX_ALL_X | 0x00000002),
	_a0_		= IDX_ALL_0,
	_a1_		= IDX_ALL_1,
	_a2_		= IDX_ALL_2,
	IDX_SUM_ALL	= (int)(0xC0000000 | 0x08000000),
	IDX_X		= (int)(0xC0000000 | 0x04000000),
	IDX_Y		= (int)(0xC0000000 | 0x04000001),
	IDX_Z		= (int)(0xC0000000 | 0x04000002),
	IDX_SPIN	= (int)(0xC0000000 | 0x02000000),
	IDX_SEPARATOR	= (int)(0xC0000000 | 0x01000000)
};

/** @brief Flexible physical index.
 *
 *  Flexible physical index for indexing arbitrary models. Each index can
 *  contain an arbitrary number of subindices. For example {x, y, spin},
 *  {x, y, z, orbital, spin}, and {subsystem, x, y, z, orbital, spin}.
 */
class Index{
public:
	/** Constructs an empty Index. */
	Index(){};

	/** Constructs an Index from an initializer list.
	 *
	 * @param i Initializer list from which the Index is constructed. */
	Index(std::initializer_list<Subindex> i) : indices(i){};

	/** Constructs an Index from an initializer list.
	 *
	 * @param i Initializer list from which the Index is constructed. */
/*	Index(
		std::initializer_list<unsigned int> i
	) : indices(i.begin(), i.end()){};*/

	/** Constructs an Index from an std::vector<int>.
	 *
	 *  @param i Vector from which the Index is constructed. */
	Index(std::vector<Subindex> i) : indices(i){};

	/** Constructs an Index from an std::vector<int>.
	 *
	 *  @param i Vector from which the Index is constructed. */
//	Index(std::vector<unsigned int> i) : indices(i.begin(), i.end()){};

	/** Copy constructor.
	 *
	 *  @param index Index to copy. */
	Index(const Index &index) : indices(index.indices){};

	/** Constructs a new Index by concatenating two indices into one total
	 *  index of the form {head, tail}.
	 *
	 *  @param head First part of the compund Index.
	 *  @param tail Second part of the compund Index.*/
	Index(const Index &head, const Index &tail);

	/** Constructs a compound Index by concatenating a list of indices,
	 *  adding IDX_SEPARATOR between every index.
	 *
	 *  @param indexList List of indices. */
//	Index(std::initializer_list<std::initializer_list<Subindex>> indexList);

	/** Constructs a compund Index by concatenating a list of indices,
	 *  adding IDX_SEPARATOR between every index.
	 *
	 *  @param indexList List of indices. */
	Index(const std::vector<std::vector<Subindex>> &indexList);

	/** Constructs a compound Index by Concatenating a list of indices,
	 *  adding IDX_SEPARATOR between every index.
	 *
	 *  @param indexList List of indices. */
	Index(std::initializer_list<Index> indexList);

	/** Constructs a compound Index by Concatenating a list of indices,
	 *  adding IDX_SEPARATOR between every index.
	 *
	 *  @param indexList List of indices. */
	Index(std::vector<Index> indexList);

	/** Constructs an Index from a string.
	 *
	 *  @param indexString String such as "{1, 2, 3} from which the Index is
	 *  constructed. */
	Index(const std::string &indexString);

	/** Constructs an Index from a serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the Index.
	 *
	 *  @param mode Mode with which the string has been serialized. */
	Index(const std::string &serialization, Serializable::Mode mode);

	/** Compare this index with another index. Returns true if the indices
	 *  have the same number of subindices and all subindices are equal.
	 *
	 *  @param index Index to compare with.
	 *  @param allowWildcard IDX_ALL is interpreted as wildcard.
	 *
	 *  @return True if the indices are equal, otherwise false. */
	bool equals(const Index &index, bool allowWildcard = false) const;

	/** Get subindex n.
	 *
	 *  @param n Subindex.
	 *
	 *  @return Subindex at position n. */
	Subindex& at(unsigned int n);

	/** Get subindex n. Constant version.
	 *
	 *  @param n Subindex.
	 *
	 *  @return Subindex at position n. */
	const Subindex& at(unsigned int n) const;

	/** Get size.
	 *
	 *  @return Number of subindices for individual indices such as
	 *  {1, 2, 3}. For compound indices such as {{1, 2}, {3,4}, {5, 6}},
	 *  the total number of subindices (here 6) plus the number of index
	 *  separators (here 2) are returned (here 6+2=8). */
	unsigned int getSize() const;

	/** Reserves memory for the Index.
	 *
	 *  @param Number of subindices to reserve space for. */
	void reserve(unsigned int size);

	/** Push subindex at the back of the index.
	 *
	 *  @param subindex Subindex to append to the Index. */
	void push_back(Subindex subindex);

	/** Removes and returns the first subindex.
	 *
	 *  @return The first subindex. */
	Subindex popFront();

	/** Removes and returns the last subindex.
	 *
	 *  @return The last subindex. */
	Subindex popBack();

	/** Returns an index with the same number or subindices, and each
	 *  subindex set to 1.
	 *
	 *  @return Index with all subindices set to 1. */
	Index getUnitRange();

	/** Returns an Index containing the subindices from position 'first' to
	 *  'last'.
	 *
	 *  @parameter first First index to include in range (inclusive).
	 *  @parameter last Last index to include in range (inclusive).
	 *
	 *  @return An index containing the subindices in the range first to
	 *  last (inclusive). */
	Index getSubIndex(int first, int last) const;

	/** Split a compound Index into its components.
	 *
	 *  @return An std::vector<Index> containing the individual @link Index
	 *  Indices @endlink.*/
	std::vector<Index> split() const;

	/** Returns true if the Index is a pattern index. That is, if it
	 *  contains a negative subindex.
	 *
	 *  @return True if the Index is a pattern index, otherwise false. */
	bool isPatternIndex() const;

	/** Print index. Mainly for debuging. */
	void print() const;

	/** Get string representation of the Index.
	 *
	 *  @return A string representation of the Index. */
	std::string toString() const;

	/** Comparison operator. Returns false if the TreeNode structure would
	 *  generate a smaller Hilbert space index for i1 than for i2.
	 *
	 *  @return True if i1 would generate a smaller Hilbert space index
	 *  than i2. */
	friend bool operator<(const Index &i1, const Index &i2);

	/** Comparison operator. Returns false if the TreeNode structure would
	 *  generate a larger Hilbert space index for i1 than for i2.
	 *
	 *  @return True if i1 would generate a larger Hilbert space index than
	 *  i2. */
	friend bool operator>(const Index &i1, const Index &i2);

	/** Subscript operator.
	 *
	 *  @param n Subindex.
	 *
	 *  @return Subindex at position n. */
	Subindex& operator[](unsigned int subindex);

	/** Subscript operator.
	 *
	 *  @param n Subindex.
	 *
	 *  @return Subindex at position n. */
	const Subindex& operator[](unsigned int subindex) const;

	/** Serialize Index. Note that Index is pseudo-Serializable in that it
	 *  implements the Serializable interface, but does so non-virtually.
	 *
	 *  @param mode Serialization mode to use.
	 *
	 *  @return Serialized string represenation of the Index. */
	std::string serialize(Serializable::Mode mode) const;

	/** Get size in bytes.
	 *
	 *  @return Memory size required to store the Index. */
	unsigned int getSizeInBytes() const;
private:
	/** Subindex container. */
	std::vector<Subindex> indices;
};

inline void Index::print() const{
	Streams::out << "{";
	for(unsigned int n = 0; n < indices.size(); n++){
		if(n != 0)
			Streams::out << ", ";
		Streams::out << indices.at(n);
	}
	Streams::out << "}\n";
}

inline std::string Index::toString() const{
	std::string str = "{";
	bool isFirstIndex = true;
	for(unsigned int n = 0; n < indices.size(); n++){
/*		if(n != 0)
			str += ", ";*/
		Subindex subindex = indices.at(n);
		if(!isFirstIndex && subindex != IDX_SEPARATOR)
			str += ", ";
		else
			isFirstIndex = false;
		switch(subindex){
		case IDX_ALL:
			str += "IDX_ALL";
			break;
		case IDX_SUM_ALL:
			str += "IDX_SUM_ALL";
			break;
		case IDX_X:
			str += "IDX_X";
			break;
		case IDX_Y:
			str += "IDX_Y";
			break;
		case IDX_Z:
			str += "IDX_Z";
			break;
		case IDX_SPIN:
			str += "IDX_SPIN";
			break;
		case IDX_SEPARATOR:
			str += "}, {";
			isFirstIndex = true;
			break;
		default:
			str += std::to_string(subindex);
			break;
		}
	}
	str += "}";

	return str;
}

inline bool Index::equals(const Index &index, bool allowWildcard) const{
	if(indices.size() == index.indices.size()){
		for(unsigned int n = 0; n < indices.size(); n++){
			if(indices.at(n) != index.indices.at(n)){
				if(!allowWildcard)
					return false;
				else{
					if(
						indices.at(n) == IDX_ALL ||
						index.indices.at(n) == IDX_ALL
					){
						continue;
					}
					else if(
						(indices.at(n) & IDX_ALL_X)
							== IDX_ALL_X
					){
						for(
							unsigned int c = 0;
							c < indices.size();
							c++
						){
							if(
								indices.at(c)
								== indices.at(n)
								&& index.indices.at(c)
								!= index.indices.at(n)
							){
								return false;
							}
						}
					}
					else if(
						(index.indices.at(n) & IDX_ALL_X)
							== IDX_ALL_X
					){
						for(
							unsigned int c = 0;
							c < indices.size();
							c++
						){
							if(
								index.indices.at(c)
								== index.indices.at(n)
								&& indices.at(c)
								!= indices.at(n)
							){
								return false;
							}
						}
					}
					else{
						return false;
					}
				}
			}
		}
	}
	else{
		return false;
	}

	return true;
}

inline Subindex& Index::at(unsigned int n){
	return indices.at(n);
}

inline const Subindex& Index::at(unsigned int n) const{
	return indices.at(n);
}

inline unsigned int Index::getSize() const{
	return indices.size();
}

inline void Index::reserve(unsigned int size){
	indices.reserve(size);
}

inline void Index::push_back(Subindex subindex){
	indices.push_back(subindex);
}

inline Subindex Index::popFront(){
	Subindex first = indices.at(0);
	indices.erase(indices.begin());

	return first;
}

inline Subindex Index::popBack(){
	Subindex last = indices.back();
	indices.pop_back();

	return last;
}

inline std::vector<Index> Index::split() const{
	std::vector<Index> components;
	components.push_back(Index());
	for(unsigned int n = 0; n < indices.size(); n++){
		if(indices[n] == IDX_SEPARATOR)
			components.push_back(Index());
		else
			components.back().push_back(indices[n]);
	}

	return components;
}

inline bool Index::isPatternIndex() const{
	for(unsigned int n = 0; n < indices.size(); n++)
		if(indices.at(n) < 0)
			return true;

	return false;
}

inline Subindex& Index::operator[](unsigned int subindex){
	return indices[subindex];
}

inline const Subindex& Index::operator[](unsigned int subindex) const{
	return indices[subindex];
}

inline unsigned int Index::getSizeInBytes() const{
	return sizeof(*this) + sizeof(int)*indices.capacity();
}

};	//End of namespace TBTK

#endif
