#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Subindex");

//! [Subindex]
#include "TBTK/Index.h"
#include "TBTK/Subindex.h"
#include "TBTK/Streams.h"

using namespace TBTK;

int main(){
	Index index({1, 2, IDX_ALL, IDX_SUM_ALL, IDX_SPIN, IDX_X, IDX_ALL_(1)});
	Streams::out << index << "\n";

	Streams::out << index[0] << "\n";
	Streams::out << index[1] << "\n";

	for(unsigned int n = 0; n < index.getSize(); n++){
		if(index[n].isWildcard())
			Streams::out << "Wildcard: " << n << "\n";
		if(index[n].isSummationIndex())
			Streams::out << "Summation index: " << n << "\n";
		if(index[n].isSpinIndex())
			Streams::out << "Spin index: " << n << "\n";
		if(index[n].isRangeIndex())
			Streams::out << "Range index: " << n << "\n";
		if(index[n].isLabeledWildcard())
			Streams::out << "Labeled wildcard: " << n << "\n";
	}

	Streams::out << "Wildcard label: " << index[6].getWildcardLabel()
		<< "\n";
}
//! [Subindex]
