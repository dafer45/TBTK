#include "EDPropertyExtractor.h"

using namespace std;

static complex<double> i(0, 1);

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
	ManyBodyContext *manyBodyContext = edSolver->getModel()->getManyBodyContext();

	const FockStateRuleSet ruleSet0 = manyBodyContext->getFockStateRuleSet();
	unsigned int subspaceID0 = edSolver->addSubspace(ruleSet0);

	unsigned int subspaceID1;
	if(manyBodyContext->wrapsBitRegister()){
		const FockSpace<BitRegister> *fockSpace = manyBodyContext->getFockSpaceBitRegister();
		const HoppingAmplitudeSet *hoppingAmplitudeSet = fockSpace->getHoppingAmplitudeSet();
		LadderOperator<BitRegister> **operators = fockSpace->getOperators();
		LadderOperator<BitRegister> *fromOperator;
		LadderOperator<BitRegister> *toOperator;
		double energySign = 0;
		switch(type){
		case ChebyshevSolver::GreensFunctionType::Retarded:
			fromOperator = &operators[hoppingAmplitudeSet->getBasisIndex(from)][0];
			toOperator = &operators[hoppingAmplitudeSet->getBasisIndex(to)][1];
			energySign = 1.;
			break;
		case ChebyshevSolver::GreensFunctionType::Advanced:
			fromOperator = &operators[hoppingAmplitudeSet->getBasisIndex(from)][1];
			toOperator = &operators[hoppingAmplitudeSet->getBasisIndex(to)][0];
			energySign = -1.;
			break;
		default:
			TBTKExit(
				"EDPropertyExtractor::calculateGreensFunction()",
				"Only support for ChebyshevSolver::GreensFunctionType::Retarded implemented so far.",
				""
			);
		}

		FockStateRuleSet ruleSet1 = (*fromOperator)*ruleSet0;
		subspaceID1 = edSolver->addSubspace(ruleSet1);

		edSolver->run(subspaceID0);
		edSolver->run(subspaceID1);

		FockStateMap::FockStateMap<BitRegister> *fockStateMap0 = fockSpace->createFockStateMap(
			ruleSet0
		);
		FockStateMap::FockStateMap<BitRegister> *fockStateMap1 = fockSpace->createFockStateMap(
			ruleSet1
		);

		complex<double> *greensFunction = new complex<double>[energyResolution];
		for(int n = 0; n < energyResolution; n++)
			greensFunction[n] = 0;

		double groundStateEnergy = edSolver->getEigenValue(subspaceID0, 0);
		for(unsigned int n = 0; n < fockStateMap1->getBasisSize(); n++){
			double E = edSolver->getEigenValue(subspaceID1, n);

			complex<double> amplitude0 = 0.;
			for(unsigned int c = 0; c < fockStateMap0->getBasisSize(); c++){
				FockState<BitRegister> psi = fockStateMap0->getFockState(c);
				(*fromOperator)*psi;
				if(psi.isNull())
					continue;

				unsigned int subspace1Index = fockStateMap1->getBasisIndex(psi);

				complex<double> a0 = edSolver->getAmplitude(subspaceID0, 0, {(int)c});
				complex<double> a1 = edSolver->getAmplitude(subspaceID1, n, {(int)subspace1Index});

				amplitude0 += conj(a1)*a0;
			}
			complex<double> amplitude1 = 0.;
			for(unsigned int c = 0; c < fockStateMap1->getBasisSize(); c++){
				FockState<BitRegister> psi = fockStateMap1->getFockState(c);
				(*toOperator)*psi;
				if(psi.isNull())
					continue;

				unsigned int subspace0Index = fockStateMap0->getBasisIndex(psi);

				complex<double> a0 = edSolver->getAmplitude(subspaceID1, n, {(int)c});
				complex<double> a1 = edSolver->getAmplitude(subspaceID0, 0, {(int)subspace0Index});

				amplitude1 += conj(a1)*a0;
			}

			int e = energyResolution*((-lowerBound + energySign*(E - groundStateEnergy))/(upperBound - lowerBound));
			if(e >= 0 && e < energyResolution)
				greensFunction[e] += conj(amplitude1)*amplitude0;
		}

		for(int n = 0; n < energyResolution; n++)
			greensFunction[n] *= -i;

		return greensFunction;
	}
	else if(manyBodyContext->wrapsExtensiveBitRegister()){
		const FockSpace<ExtensiveBitRegister> *fockSpace = manyBodyContext->getFockSpaceExtensiveBitRegister();
		const HoppingAmplitudeSet *hoppingAmplitudeSet = fockSpace->getHoppingAmplitudeSet();
		LadderOperator<ExtensiveBitRegister> **operators = fockSpace->getOperators();
		LadderOperator<ExtensiveBitRegister> *fromOperator;
		LadderOperator<ExtensiveBitRegister> *toOperator;
		double energySign = 0;
		switch(type){
		case ChebyshevSolver::GreensFunctionType::Retarded:
			fromOperator = &operators[hoppingAmplitudeSet->getBasisIndex(from)][0];
			toOperator = &operators[hoppingAmplitudeSet->getBasisIndex(to)][1];
			energySign = 1.;
			break;
		case ChebyshevSolver::GreensFunctionType::Advanced:
			fromOperator = &operators[hoppingAmplitudeSet->getBasisIndex(from)][1];
			toOperator = &operators[hoppingAmplitudeSet->getBasisIndex(to)][0];
			energySign = -1.;
			break;
		default:
			TBTKExit(
				"EDPropertyExtractor::calculateGreensFunction()",
				"Only support for ChebyshevSolver::GreensFunctionType::Retarded implemented so far.",
				""
			);
		}

		FockStateRuleSet ruleSet1 = (*fromOperator)*ruleSet0;
		subspaceID1 = edSolver->addSubspace(ruleSet1);

		edSolver->run(subspaceID0);
		edSolver->run(subspaceID1);

		FockStateMap::FockStateMap<ExtensiveBitRegister> *fockStateMap0 = fockSpace->createFockStateMap(
			ruleSet0
		);
		FockStateMap::FockStateMap<ExtensiveBitRegister> *fockStateMap1 = fockSpace->createFockStateMap(
			ruleSet1
		);

		complex<double> *greensFunction = new complex<double>[energyResolution];
		for(int n = 0; n < energyResolution; n++)
			greensFunction[n] = 0;

		double groundStateEnergy = edSolver->getEigenValue(subspaceID0, 0);
		for(unsigned int n = 0; n < fockStateMap1->getBasisSize(); n++){
			double E = edSolver->getEigenValue(subspaceID1, n);

			complex<double> amplitude0 = 0.;
			for(unsigned int c = 0; c < fockStateMap0->getBasisSize(); c++){
				FockState<ExtensiveBitRegister> psi = fockStateMap0->getFockState(c);
				(*fromOperator)*psi;
				if(psi.isNull())
					continue;

				unsigned int subspace1Index = fockStateMap1->getBasisIndex(psi);

				complex<double> a0 = edSolver->getAmplitude(subspaceID0, 0, {(int)c});
				complex<double> a1 = edSolver->getAmplitude(subspaceID1, n, {(int)subspace1Index});

				amplitude0 += conj(a1)*a0;
			}
			complex<double> amplitude1 = 0.;
			for(unsigned int c = 0; c < fockStateMap1->getBasisSize(); c++){
				FockState<ExtensiveBitRegister> psi = fockStateMap1->getFockState(c);
				(*toOperator)*psi;
				if(psi.isNull())
					continue;

				unsigned int subspace0Index = fockStateMap0->getBasisIndex(psi);

				complex<double> a0 = edSolver->getAmplitude(subspaceID1, n, {(int)c});
				complex<double> a1 = edSolver->getAmplitude(subspaceID0, 0, {(int)subspace0Index});

				amplitude1 += conj(a1)*a0;
			}

			int e = energyResolution*((-lowerBound + energySign*(E - groundStateEnergy))/(upperBound - lowerBound));
			if(e >= 0 && e < energyResolution)
				greensFunction[e] += conj(amplitude1)*amplitude0;
		}

		for(int n = 0; n < energyResolution; n++)
			greensFunction[n] *= -i;

		return greensFunction;
	}
	else{
		TBTKExit(
			"EDPropertyExtractor::calculateGreensFunction()",
			"Unknown BitRegister type.",
			""
		);
	}
}

};	//End of namespace TBTK
