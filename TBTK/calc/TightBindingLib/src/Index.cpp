#include "../include/Index.h"
#include "../include/Streams.h"

using namespace std;

namespace TBTK{

bool operator<(const Index &i1, const Index &i2){
	int minNumIndices;
	if(i1.size() < i2.size())
		minNumIndices = i1.size();
	else
		minNumIndices = i2.size();

	for(int n = 0; n < minNumIndices; n++){
		if(i1.at(n) == i2.at(n))
			continue;

		if(i1.at(n) < i2.at(n))
			return true;
		else
			return false;
	}

	Util::Streams::err << "Error in operator<(Index &i1, Index &i2): Comparison between indices of types mutually incompatible with the TreeNode structure.\n";
	exit(1);
}

bool operator>(const Index &i1, const Index &i2){
	int minNumIndices;
	if(i1.size() < i2.size())
		minNumIndices = i1.size();
	else
		minNumIndices = i2.size();

	for(int n = 0; n < minNumIndices; n++){
		if(i1.at(n) == i2.at(n))
			continue;

		if(i1.at(n) < i2.at(n))
			return false;
		else
			return true;
	}

	Util::Streams::err << "Error in operator>(Index &i1, Index &i2): Comparison between indices of types mutually incompatible with the TreeNode structure.\n";
	exit(1);
}

Index Index::getUnitRange(){
	Index unitRange = *this;

	for(unsigned int n = 0; n < size(); n++)
		unitRange.at(n) = 1;

	return unitRange;
}

};
