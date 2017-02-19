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
	switch(type){
	case ChebyshevSolver::GreensFunctionType::Principal:
	{
		complex<double> *greensFunctionA = calculateGreensFunction(
			to,
			from,
			ChebyshevSolver::GreensFunctionType::Advanced
		);

		complex<double> *greensFunctionR = calculateGreensFunction(
			to,
			from,
			ChebyshevSolver::GreensFunctionType::Retarded
		);

		complex<double> *greensFunction = new complex<double>[energyResolution];
		for(int n = 0; n < energyResolution; n++)
			greensFunction[n] = (greensFunctionA[n] + greensFunctionR[n])/2.;

		delete [] greensFunctionA;
		delete [] greensFunctionR;

		return greensFunction;
	}
	case ChebyshevSolver::GreensFunctionType::NonPrincipal:
	{
		complex<double> *greensFunctionA = calculateGreensFunction(
			to,
			from,
			ChebyshevSolver::GreensFunctionType::Advanced
		);

		complex<double> *greensFunctionR = calculateGreensFunction(
			to,
			from,
			ChebyshevSolver::GreensFunctionType::Retarded
		);

		complex<double> *greensFunction = new complex<double>[energyResolution];
		for(int n = 0; n < energyResolution; n++)
			greensFunction[n] = (greensFunctionA[n] - greensFunctionR[n])/2.;

		delete [] greensFunctionA;
		delete [] greensFunctionR;

		return greensFunction;
	}
	default:
		break;
	}

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

				amplitude0 += conj(a1)*a0*(double)psi.getPrefactor();
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

				amplitude1 += conj(a1)*a0*(double)psi.getPrefactor();
			}

			int e = energyResolution*((-lowerBound + energySign*(E - groundStateEnergy))/(upperBound - lowerBound));
			if(e >= 0 && e < energyResolution)
				greensFunction[e] += amplitude1*amplitude0;
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

				amplitude0 += conj(a1)*a0*(double)psi.getPrefactor();
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

				amplitude1 += conj(a1)*a0*(double)psi.getPrefactor();
			}

			int e = energyResolution*((-lowerBound + energySign*(E - groundStateEnergy))/(upperBound - lowerBound));
			if(e >= 0 && e < energyResolution)
				greensFunction[e] += amplitude1*amplitude0;
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

complex<double> EDPropertyExtractor::calculateExpectationValue(
	Index to,
	Index from
){
	TBTKNotYetImplemented("EDPropertyExtractor::calculateExpectationValue()");
}

Property::Density* EDPropertyExtractor::calculateDensity(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions = 0;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Density *density = new Property::Density(lDimensions, lRanges);

	calculate(calculateDensityCallback, (void*)density->data, pattern, ranges, 0, 1);

	return density;
}

Property::Magnetization* EDPropertyExtractor::calculateMagnetization(
	Index pattern,
	Index ranges
){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		delete [] (int*)hint;
		TBTKExit(
			"EDPropertyExtractor::calculateMagnetization()",
			"No spin index found.",
			"Use IDX_SPIN to indicate position of spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::Magnetization *magnetization = new Property::Magnetization(
		lDimensions,
		lRanges
	);

	calculate(
		calculateMagnetizationCallback,
		(void*)magnetization->data,
		pattern,
		ranges,
		0,
		1
	);

	delete [] (int*)hint;

	return magnetization;
}

Property::LDOS* EDPropertyExtractor::calculateLDOS(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::LDOS *ldos = new Property::LDOS(
		lDimensions,
		lRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateLDOSCallback,
		(void*)ldos->data,
		pattern,
		ranges,
		0,
		1
	);

	return ldos;
}

Property::SpinPolarizedLDOS* EDPropertyExtractor::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	hint = new int[1];
	((int*)hint)[0] = -1;
	for(unsigned int n = 0; n < pattern.size(); n++){
		if(pattern.at(n) == IDX_SPIN){
			((int*)hint)[0] = n;
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(((int*)hint)[0] == -1){
		delete [] (int*)hint;
		TBTKExit(
			"EDPropertyExtractor::calculateSpinPolarizedLDOS()",
			"No spin index found.",
			"Use IDX_SPIN to indicate position of spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	int lDimensions;
	int *lRanges;
	getLoopRanges(pattern, ranges, &lDimensions, &lRanges);
	Property::SpinPolarizedLDOS *spinPolarizedLDOS = new Property::SpinPolarizedLDOS(
		lDimensions,
		lRanges,
		lowerBound,
		upperBound,
		energyResolution
	);

	calculate(
		calculateSpinPolarizedLDOSCallback,
		(void*)spinPolarizedLDOS->data,
		pattern,
		ranges,
		0,
		1
	);

	delete [] (int*) hint;

	return spinPolarizedLDOS;
}

void EDPropertyExtractor::calculateDensityCallback(
	PropertyExtractor *cb_this,
	void *density,
	const Index &index,
	int offset
){
	TBTKNotYetImplemented("EDPropertyExtractor::calculateDensityCallback()");
}

void EDPropertyExtractor::calculateMagnetizationCallback(
	PropertyExtractor *cb_this,
	void *magnetization,
	const Index &index,
	int offset
){
	TBTKNotYetImplemented("EDPropertyExtractor::calculateMagnetizationCallback()");
}

void EDPropertyExtractor::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	void *ldos,
	const Index &index,
	int offset
){
	EDPropertyExtractor *pe = (EDPropertyExtractor*)cb_this;

	complex<double> *greensFunction = pe->calculateGreensFunction(
		index,
		index,
		ChebyshevSolver::GreensFunctionType::NonPrincipal
	);

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(int n = 0; n < pe->energyResolution; n++)
		((double*)ldos)[pe->energyResolution*offset + n] += imag(greensFunction[n])/M_PI*dE;

	delete [] greensFunction;
}

void EDPropertyExtractor::calculateSpinPolarizedLDOSCallback(
	PropertyExtractor *cb_this,
	void *spinPolarizedLDOS,
	const Index &index,
	int offset
){
	EDPropertyExtractor *pe = (EDPropertyExtractor*)cb_this;

	int spinIndex = ((int*)(pe->hint))[0];
	Index to(index);
	Index from(index);
	complex<double> *greensFunction;

	const double dE = (pe->upperBound - pe->lowerBound)/pe->energyResolution;
	for(unsigned int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;
		from.at(spinIndex) = n%2;

		greensFunction = pe->calculateGreensFunction(
			to,
			from,
			ChebyshevSolver::GreensFunctionType::NonPrincipal
		);

		for(int e = 0; e < pe->energyResolution; e++)
			((complex<double>*)spinPolarizedLDOS)[4*pe->energyResolution*offset + 4*e + n] += imag(greensFunction[e])/M_PI*dE;

		delete [] greensFunction;
	}
}

};	//End of namespace TBTK
