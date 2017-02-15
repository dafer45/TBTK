#include "EDPropertyExtractor.h"

using namespace std;

namespace TBTK{

EDPropertyExtractor::EDPropertyExtractor(ExactDiagonalizationSolver *edSolver){
	this->edSolver = edSolver;
}

EDPropertyExtractor::~EDPropertyExtractor(){
}

};	//End of namespace TBTK
