#include "TBTK/PropertyExtractor/ExactDiagonalizer.h"

using namespace std;

static complex<double> i(0, 1);

namespace TBTK{
namespace PropertyExtractor{

ExactDiagonalizer::ExactDiagonalizer(Solver::ExactDiagonalizer &edSolver){
	this->edSolver = &edSolver;
}

ExactDiagonalizer::~ExactDiagonalizer(){
}

Property::GreensFunction* ExactDiagonalizer::calculateGreensFunction(
	Index to,
	Index from,
	Property::GreensFunction::Type type
){
	double lowerBound = getLowerBound();
	double upperBound = getUpperBound();
	int energyResolution = getEnergyResolution();

	IndexTree memoryLayout;
	memoryLayout.add({to, from});
	memoryLayout.generateLinearMap();

	switch(type){
	case Property::GreensFunction::Type::Principal:
	{
		Property::GreensFunction *greensFunctionA = calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::Advanced
		);

		Property::GreensFunction *greensFunctionR = calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::Retarded
		);

		const std::vector<complex<double>> &greensFunctionAData
			= greensFunctionA->getData();
		const std::vector<complex<double>> &greensFunctionRData
			= greensFunctionR->getData();

		complex<double> *greensFunctionData = new complex<double>[energyResolution];
		for(int n = 0; n < energyResolution; n++)
			greensFunctionData[n] = (greensFunctionAData[n] + greensFunctionRData[n])/2.;

		delete greensFunctionA;
		delete greensFunctionR;

		Property::GreensFunction *greensFunction = new Property::GreensFunction(
			memoryLayout,
			type,
			lowerBound,
			upperBound,
			energyResolution,
			greensFunctionData
		);

		delete [] greensFunctionData;

		return greensFunction;
	}
	case Property::GreensFunction::Type::NonPrincipal:
	{
		Property::GreensFunction *greensFunctionA = calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::Advanced
		);

		Property::GreensFunction *greensFunctionR = calculateGreensFunction(
			to,
			from,
			Property::GreensFunction::Type::Retarded
		);

		const std::vector<complex<double>> &greensFunctionAData
			= greensFunctionA->getData();
		const std::vector<complex<double>> &greensFunctionRData
			= greensFunctionR->getData();

		complex<double> *greensFunctionData
			= new complex<double>[energyResolution];
		for(int n = 0; n < energyResolution; n++)
			greensFunctionData[n] = (greensFunctionAData[n] - greensFunctionRData[n])/2.;

		delete greensFunctionA;
		delete greensFunctionR;

		Property::GreensFunction *greensFunction = new Property::GreensFunction(
			memoryLayout,
			type,
			lowerBound,
			upperBound,
			energyResolution,
			greensFunctionData
		);

		delete [] greensFunctionData;

		return greensFunction;
	}
	default:
		break;
	}

	ManyParticleContext *manyParticleContext = edSolver->getModel().getManyParticleContext();

	const FockStateRuleSet ruleSet0 = manyParticleContext->getFockStateRuleSet();
	unsigned int subspaceID0 = edSolver->addSubspace(ruleSet0);

	unsigned int subspaceID1;
	if(manyParticleContext->wrapsBitRegister()){
		const FockSpace<BitRegister> *fockSpace = manyParticleContext->getFockSpaceBitRegister();
		const HoppingAmplitudeSet *hoppingAmplitudeSet = fockSpace->getHoppingAmplitudeSet();
		LadderOperator<BitRegister> const* const* operators = fockSpace->getOperators();
		const LadderOperator<BitRegister> *fromOperator;
		const LadderOperator<BitRegister> *toOperator;
		double energySign = 0;
		switch(type){
		case Property::GreensFunction::Type::Retarded:
			fromOperator = &operators[hoppingAmplitudeSet->getBasisIndex(from)][0];
			toOperator = &operators[hoppingAmplitudeSet->getBasisIndex(to)][1];
			energySign = 1.;
			break;
		case Property::GreensFunction::Type::Advanced:
			fromOperator = &operators[hoppingAmplitudeSet->getBasisIndex(from)][1];
			toOperator = &operators[hoppingAmplitudeSet->getBasisIndex(to)][0];
			energySign = -1.;
			break;
		default:
			TBTKExit(
				"PropertyExtractor::ExactDiagonalizer::calculateGreensFunction()",
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

		complex<double> *greensFunctionData = new complex<double>[energyResolution];
		for(int n = 0; n < energyResolution; n++)
			greensFunctionData[n] = 0;

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
				greensFunctionData[e] += amplitude1*amplitude0;
		}

		for(int n = 0; n < energyResolution; n++)
			greensFunctionData[n] *= -i;

		Property::GreensFunction *greensFunction = new Property::GreensFunction(
			memoryLayout,
			type,
			lowerBound,
			upperBound,
			energyResolution,
			greensFunctionData
		);

		delete [] greensFunctionData;

		return greensFunction;
	}
	else if(manyParticleContext->wrapsExtensiveBitRegister()){
		const FockSpace<ExtensiveBitRegister> *fockSpace = manyParticleContext->getFockSpaceExtensiveBitRegister();
		const HoppingAmplitudeSet *hoppingAmplitudeSet = fockSpace->getHoppingAmplitudeSet();
		LadderOperator<ExtensiveBitRegister> const* const* operators = fockSpace->getOperators();
		const LadderOperator<ExtensiveBitRegister> *fromOperator;
		const LadderOperator<ExtensiveBitRegister> *toOperator;
		double energySign = 0;
		switch(type){
		case Property::GreensFunction::Type::Retarded:
			fromOperator = &operators[hoppingAmplitudeSet->getBasisIndex(from)][0];
			toOperator = &operators[hoppingAmplitudeSet->getBasisIndex(to)][1];
			energySign = 1.;
			break;
		case Property::GreensFunction::Type::Advanced:
			fromOperator = &operators[hoppingAmplitudeSet->getBasisIndex(from)][1];
			toOperator = &operators[hoppingAmplitudeSet->getBasisIndex(to)][0];
			energySign = -1.;
			break;
		default:
			TBTKExit(
				"PropertyExtractor::ExactDiagonalizer::calculateGreensFunction()",
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

		complex<double> *greensFunctionData = new complex<double>[energyResolution];
		for(int n = 0; n < energyResolution; n++)
			greensFunctionData[n] = 0;

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
				greensFunctionData[e] += amplitude1*amplitude0;
		}

		for(int n = 0; n < energyResolution; n++)
			greensFunctionData[n] *= -i;

		Property::GreensFunction *greensFunction = new Property::GreensFunction(
			memoryLayout,
			type,
			lowerBound,
			upperBound,
			energyResolution,
			greensFunctionData
		);

		delete [] greensFunctionData;

		return greensFunction;
	}
	else{
		TBTKExit(
			"PropertyExtractor::ExactDiagonalizer::calculateGreensFunction()",
			"Unknown BitRegister type.",
			""
		);
	}
}

complex<double> ExactDiagonalizer::calculateExpectationValue(
	Index to,
	Index from
){
	TBTKNotYetImplemented("PropertyExtractor::ExactDiagonalizer::calculateExpectationValue()");
}

Property::Density ExactDiagonalizer::calculateDensity(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::Density density(loopRanges);

	Information information;
	calculate(
		calculateDensityCallback,
		density,
		pattern,
		ranges,
		0,
		1,
		information
	);

	return density;
}

Property::Magnetization ExactDiagonalizer::calculateMagnetization(
	Index pattern,
	Index ranges
){
	Information information;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n).isSpinIndex()){
			information.setSpinIndex(n);
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(information.getSpinIndex() == -1){
		TBTKExit(
			"PropertyExtractor::ExactDiagonalizer::calculateMagnetization()",
			"No spin index found.",
			"Use IDX_SPIN to indicate position of spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::Magnetization magnetization(loopRanges);

	calculate(
		calculateMagnetizationCallback,
		magnetization,
		pattern,
		ranges,
		0,
		1,
		information
	);

	return magnetization;
}

Property::LDOS ExactDiagonalizer::calculateLDOS(
	Index pattern,
	Index ranges
){
	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::LDOS ldos(
		loopRanges,
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);

	Information information;
	calculate(
		calculateLDOSCallback,
		ldos,
		pattern,
		ranges,
		0,
		1,
		information
	);

	return ldos;
}

Property::SpinPolarizedLDOS ExactDiagonalizer::calculateSpinPolarizedLDOS(
	Index pattern,
	Index ranges
){
	Information information;
	for(unsigned int n = 0; n < pattern.getSize(); n++){
		if(pattern.at(n).isSpinIndex()){
			information.setSpinIndex(n);
			pattern.at(n) = 0;
			ranges.at(n) = 1;
			break;
		}
	}
	if(information.getSpinIndex() == -1){
		TBTKExit(
			"PropertyExtractor::ExactDiagonalizer::calculateSpinPolarizedLDOS()",
			"No spin index found.",
			"Use IDX_SPIN to indicate position of spin index."
		);
	}

	ensureCompliantRanges(pattern, ranges);

	vector<int> loopRanges = getLoopRanges(pattern, ranges);
	Property::SpinPolarizedLDOS spinPolarizedLDOS(
		loopRanges,
		getLowerBound(),
		getUpperBound(),
		getEnergyResolution()
	);

	calculate(
		calculateSpinPolarizedLDOSCallback,
		spinPolarizedLDOS,
		pattern,
		ranges,
		0,
		1,
		information
	);

	return spinPolarizedLDOS;
}

void ExactDiagonalizer::calculateDensityCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	TBTKNotYetImplemented("PropertyExtractor::ExactDiagonalizer::calculateDensityCallback()");
}

void ExactDiagonalizer::calculateMagnetizationCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	TBTKNotYetImplemented("PropertyExtractor::ExactDiagonalizer::calculateMagnetizationCallback()");
}

void ExactDiagonalizer::calculateLDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	ExactDiagonalizer *pe = (ExactDiagonalizer*)cb_this;
	Property::LDOS &ldos = (Property::LDOS&)property;
	vector<double> &data = ldos.getDataRW();

	Property::GreensFunction *greensFunction = pe->calculateGreensFunction(
		index,
		index,
		Property::GreensFunction::Type::NonPrincipal
	);
	const std::vector<complex<double>> &greensFunctionData
		= greensFunction->getData();

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	const double dE = (upperBound - lowerBound)/energyResolution;
	for(int n = 0; n < energyResolution; n++){
		data[energyResolution*offset + n]
			+= imag(greensFunctionData[n])/M_PI*dE;
	}

	delete greensFunction;
}

void ExactDiagonalizer::calculateSpinPolarizedLDOSCallback(
	PropertyExtractor *cb_this,
	Property::Property &property,
	const Index &index,
	int offset,
	Information &information
){
	TBTKNotYetImplemented("PropertyExtractor::ExactDiagonalizer::calculateMagnetizationCallback()");

/*	ExactDiagonalizer *pe = (ExactDiagonalizer*)cb_this;
	Property::SpinPolarizedLDOS &spinPolarizedLDOS
		= (Property::SpinPolarizedLDOS&)property;
	vector<SpinMatrix> &data = spinPolarizedLDOS.getDataRW();

	int spinIndex = ((int*)(pe->hint))[0];
	Index to(index);
	Index from(index);

	double lowerBound = pe->getLowerBound();
	double upperBound = pe->getUpperBound();
	int energyResolution = pe->getEnergyResolution();

	const double dE = (upperBound - lowerBound)/energyResolution;
	for(unsigned int n = 0; n < 4; n++){
		to.at(spinIndex) = n/2;
		from.at(spinIndex) = n%2;

		Property::GreensFunction *greensFunction
			= pe->calculateGreensFunction(
				to,
				from,
				Property::GreensFunction::Type::NonPrincipal
			);
		const std::vector<complex<double>> &greensFunctionData
			= greensFunction->getData();

		for(int e = 0; e < energyResolution; e++)
			data[4*energyResolution*offset + 4*e + n] += imag(greensFunctionData[e])/M_PI*dE;
//			((complex<double>*)spinPolarizedLDOS)[4*energyResolution*offset + 4*e + n] += imag(greensFunctionData[e])/M_PI*dE;

		delete greensFunction;
	}*/
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
