#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("IndexTreeGenerator");

//! [IndexTreeGenerator]
#include "TBTK/Model.h"
#include "TBTK/PropertyExtractor/IndexTreeGenerator.h"

using namespace TBTK;
using namespace PropertyExtractor;

int main(){
	Model model;
	for(unsigned int x = 0; x < 2; x++){
		for(unsigned int y = 0; y < 2; y++){
			for(unsigned int spin = 0; spin < 2; spin++){
				model << HoppingAmplitude(
					1,
					{x, y, spin},
					{x, y, spin}
				);
			}
		}
	}
	model.construct();

	IndexTreeGenerator indexTreeGenerator(model);

	IndexTree indexTree
		= indexTreeGenerator.generate({{1, IDX_ALL, IDX_SUM_ALL}});
	Streams::out << indexTree << "\n";

	indexTreeGenerator.setKeepSummationWildcards(true);
	indexTree = indexTreeGenerator.generate({{1, IDX_ALL, IDX_SUM_ALL}});
	Streams::out << indexTree << "\n";

	indexTree = indexTreeGenerator.generate({{1, IDX_ALL, IDX_SPIN}});
	Streams::out << indexTree << "\n";

	indexTreeGenerator.setKeepSpinWildcards(true);
	indexTree = indexTreeGenerator.generate({{1, IDX_ALL, IDX_SPIN}});
	Streams::out << indexTree << "\n";

	indexTree = indexTreeGenerator.generate({
		{{1, IDX_ALL, IDX_SPIN}, {1, IDX_ALL, IDX_SPIN}}
	});
	Streams::out << indexTree << "\n";
}
//! [IndexTreeGenerator]
