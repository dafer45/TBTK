/** @package TBTKcalc
 *  @file Index.h
 *  @brief Data structure for flexible physical indices
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_INDEX
#define COM_DAFER45_TBTK_INDEX

#include <iostream>

/** Flexible physical index for indexing arbitrary models. Each index can
 *  contain an arbitrary number of subindices. For example {x, y, spin},
 *  {x, y, z, orbital, spin}, and {subsystem, x, y, z, orbital, spin}.
 */
class Index{
public:
	/** Subindex container. */
	std::vector<int> indices;

	/** Constructor. */
	Index(std::initializer_list<int> i) : indices(i){};

	/** Constructor. */
	Index(std::vector<int> i) : indices(i){};

	/** Copy constructor. */
	Index(const Index &index) : indices(index.indices){};

	/** Compare this index with another index. Returns true if the indices
	 * have the same number of subindices and all subindices are equal.
	 * @param index Index to compare with. */
	bool equals(Index &index);

	/** Print index. Mainly for debuging. */
	void print() const;
};

inline void Index::print() const{
	for(unsigned int n = 0; n < indices.size(); n++)
		std::cout << indices.at(n) << " ";
	std::cout << "\n";
}

inline bool Index::equals(Index &index){
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

#endif

