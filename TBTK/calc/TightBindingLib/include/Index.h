#ifndef COM_DAFER45_TBTK_INDEX
#define COM_DAFER45_TBTK_INDEX

#include <iostream>

class Index{
public:
	std::vector<int> indices;

	Index(std::initializer_list<int> i) : indices(i){};
	Index(std::vector<int> i) : indices(i){};
	Index(const Index &index) : indices(index.indices){};

	bool equals(Index &index);

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

