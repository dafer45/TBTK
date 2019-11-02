#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("PatternValidator");

//! [PatternValidator]
#include "TBTK/PropertyExtractor/PatternValidator.h"

using namespace TBTK;
using namespace PropertyExtractor;

int main(){
	PatternValidator patternValidator;
	patternValidator.setNumRequiredComponentIndices(2);
	patternValidator.setAllowedSubindexFlags({IDX_ALL, IDX_SPIN});

	patternValidator.validate({
		{{1, IDX_ALL, IDX_SPIN}, {2, 3, 4}},
		{{2, IDX_ALL, IDX_SPIN}, {2, 3, 4}}
	});
	Streams::out << "Validation passed\n";
}
//! [PatternValidator]
