/** @package TBTKcalc
 *  @file Index.h
 *  @brief Data structure for flexible physical indices
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEX
#define COM_DAFER45_TBTK_INDEX

#include <iostream>
#include <vector>

namespace TBTK{

/** Flexible physical index for indexing arbitrary models. Each index can
 *  contain an arbitrary number of subindices. For example {x, y, spin},
 *  {x, y, z, orbital, spin}, and {subsystem, x, y, z, orbital, spin}.
 */
class Index{
public:
	/** Constructor. */
	Index(std::initializer_list<int> i) : indices(i){};

	/** Constructor. */
	Index(std::vector<int> i) : indices(i){};

	/** Copy constructor. */
	Index(const Index &index) : indices(index.indices){};

	/** Compare this index with another index. Returns true if the indices
	 * have the same number of subindices and all subindices are equal.
	 * @param index Index to compare with. */
	bool equals(const Index &index) const;

	/** Get subindex n. */
	int& at(unsigned int n);

	/** Get subindex n. Constant version. */
	const int& at(unsigned int n) const;

	/** Get size. */
	unsigned int size() const;

	/** Removes and returns the first subindex. */
	int popFront();

	/** Removes and returns the last subindex. */
	int popBack();

	/** Print index. Mainly for debuging. */
	void print() const;

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
	std::cout << "{";
	for(unsigned int n = 0; n < indices.size(); n++){
		if(n != 0)
			std::cout << ", ";
		std::cout << indices.at(n);
	}
	std::cout << "}\n";
}

inline bool Index::equals(const Index &index) const{
	if(indices.size() == index.indices.size()){
		for(unsigned int n = 0; n < indices.size(); n++){
			if(indices.at(n) != index.indices.at(n))
				return false;
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

};	//End of namespace TBTK

#endif
