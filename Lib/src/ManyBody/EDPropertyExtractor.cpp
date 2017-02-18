#include "EDPropertyExtractor.h"

using namespace std;

namespace TBTK{

EDPropertyExtractor::EDPropertyExtractor(ExactDiagonalizationSolver *edSolver){
	this->edSolver = edSolver;
}

EDPropertyExtractor::~EDPropertyExtractor(){
}

complex<double>* EDPropertyExtractor::calculateGreensFunction(
	Index to,
	Index from,
	ChebyshevSolver::GreensFunctionType type
){
	TBTKNotYetImplemented("EDPropertyExtractor::calculateGreensFunction");
	unsigned int subspaceID = edSolver->addSubspace(edSolver->getModel()->getManyBodyContext()->getFockStateRuleSet());

	switch(type){
	case ChebyshevSolver::GreensFunctionType::Retarded:
		break;
	default:
		TBTKExit(
			"EDPropertyExtractor::calculateGreensFunction()",
			"Only support for ChebyshevSolver::GreensFunctionType::Retarded implemented so far.",
			""
		);
	}
}

};	//End of namespace TBTK
