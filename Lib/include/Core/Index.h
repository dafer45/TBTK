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
 *  @brief Data structure for flexible physical indices
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEX
#define COM_DAFER45_TBTK_INDEX

#include "Streams.h"

#include <vector>

namespace TBTK{

enum {
	IDX_ALL = -1,
	IDX_SUM_ALL = -2,
	IDX_X = -3,
	IDX_Y = -4,
	IDX_Z = -5,
	IDX_SPIN = -6
};

/** Flexible physical index for indexing arbitrary models. Each index can
 *  contain an arbitrary number of subindices. For example {x, y, spin},
 *  {x, y, z, orbital, spin}, and {subsystem, x, y, z, orbital, spin}.
 */
class Index{
public:
	/** Constructor. */
	Index(){};

	/** Constructor. */
	Index(std::initializer_list<int> i) : indices(i){};

	/** Constructor. */
	Index(std::vector<int> i) : indices(i){};

	/** Copy constructor. */
	Index(const Index &index) : indices(index.indices){};

	/** Constructor. Concatenates two indices into one total index of the
	 *  form {head, tail}. */
	Index(const Index &head, const Index &tail);

	/** Compare this index with another index. Returns true if the indices
	 * have the same number of subindices and all subindices are equal.
	 * @param index Index to compare with.
	 * @param allowWildcard IDX_ALL is interpreted as wildcard. */
	bool equals(const Index &index, bool allowWildcard = false) const;

	/** Get subindex n. */
	int& at(unsigned int n);

	/** Get subindex n. Constant version. */
	const int& at(unsigned int n) const;

	/** Get size. */
	unsigned int size() const;

	/** Push subindex at the back of the index. */
	void push_back(int subindex);

	/** Removes and returns the first subindex. */
	int popFront();

	/** Removes and returns the last subindex. */
	int popBack();

	/** Returns an index with the same number or subindices, and each
	 *  subindex set to 1. */
	Index getUnitRange();

	/** Returns an Index containing the subindices from position 'first' to
	 *  'last'. */
	Index getSubIndex(int first, int last);

	/** Returns true if the Index is a pattern index. That is, if it
	 *  contains a negative subindex. */
	bool isPatternIndex() const;

	/** Print index. Mainly for debuging. */
	void print() const;

	/** Print index. Mainly for debuging. */
	std::string toString() const;

	/** Comparison operator. Returns false if the TreeNode structure would
	 *  generate a smaller Hilbert space index for i1 than for i2. */
	friend bool operator<(const Index &i1, const Index &i2);

	/** Comparison operator. Returns false if the TreeNode structure would
	 *  generate a larger Hilbert space index for i1 than for i2. */
	friend bool operator>(const Index &i1, const Index &i2);
private:
	/** Subindex container. */
	std::vector<int> indices;
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
	for(unsigned int n = 0; n < indices.size(); n++){
		if(n != 0)
			str += ", ";
		int subindex = indices.at(n);
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
					)
						continue;
					else
						return false;
				}
			}
		}
	}
	else{
		return false;
	}

	return true;
}

inline int& Index::at(unsigned int n){
	return indices.at(n);
}

inline const int& Index::at(unsigned int n) const{
	return indices.at(n);
}

inline unsigned int Index::size() const{
	return indices.size();
}

inline void Index::push_back(int subindex){
	indices.push_back(subindex);
}

inline int Index::popFront(){
	int first = indices.at(0);
	indices.erase(indices.begin());

	return first;
}

inline int Index::popBack(){
	int last = indices.back();
	indices.pop_back();

	return last;
}

inline bool Index::isPatternIndex() const{
	for(unsigned int n = 0; n < indices.size(); n++)
		if(indices.at(n) < 0)
			return true;

	return false;
}

};	//End of namespace TBTK

#endif
