#include "Solver/ExactDiagonalizer.h"
#include "FockSpace.h"
#include "FileWriter.h"
#include "DOS.h"
#include "Solver/Diagonalizer.h"
#include "DPropertyExtractor.h"
#include "SumRule.h"
#include "DifferenceRule.h"
#include "WrapperRule.h"
#include "Timer.h"

using namespace std;

namespace TBTK{
namespace Solver{

ExactDiagonalizer::ExactDiagonalizer(/*Model *model*/){
//	this->model = model;
}

ExactDiagonalizer::~ExactDiagonalizer(){
}

unsigned int ExactDiagonalizer::addSubspace(initializer_list<const FockStateRule::WrapperRule> rules){
	FockStateRuleSet fockStateRuleSet;
	for(unsigned int n = 0; n < rules.size(); n++)
		fockStateRuleSet.addFockStateRule(*(rules.begin()+n));

	return addSubspace(fockStateRuleSet);
}

unsigned int ExactDiagonalizer::addSubspace(vector<FockStateRule::WrapperRule> rules){
	FockStateRuleSet fockStateRuleSet;
	for(unsigned int n = 0; n < rules.size(); n++)
		fockStateRuleSet.addFockStateRule(rules.at(n));

	return addSubspace(fockStateRuleSet);
}

unsigned int ExactDiagonalizer::addSubspace(const FockStateRuleSet &rules){
	for(unsigned int n = 0; n < subspaceContexts.size(); n++)
		if(rules == subspaceContexts.at(n).fockStateRuleSet)
			return n;

	subspaceContexts.push_back(SubspaceContext(rules));

	return subspaceContexts.size()-1;
}

void ExactDiagonalizer::run(unsigned int subspace){
	SubspaceContext &subspaceContext = subspaceContexts.at(subspace);
	if(subspaceContext.manyBodyModel == NULL){
		setupManyBodyModel(subspace);
		subspaceContext.dSolver.reset(new Diagonalizer());
		subspaceContext.dSolver->setModel(*subspaceContext.manyBodyModel.get());
		subspaceContext.dSolver->run();
	}
}

template<>
void ExactDiagonalizer::setupManyBodyModel<BitRegister>(unsigned int subspace){
	FockSpace<BitRegister> *fockSpace = getModel().getManyBodyContext()->getFockSpaceBitRegister();
	LadderOperator<BitRegister> **operators = fockSpace->getOperators();
	SubspaceContext &subspaceContext = subspaceContexts.at(subspace);
	FockStateMap::FockStateMap<BitRegister> *fockStateMap = fockSpace->createFockStateMap(
		subspaceContext.fockStateRuleSet
	);

	subspaceContext.manyBodyModel.reset(new Model());
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		HoppingAmplitudeSet::Iterator it = getModel().getHoppingAmplitudeSet()->getIterator();
		const HoppingAmplitude *ha;
		while((ha = it.getHA())){
			it.searchNextHA();

			FockState<BitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

//			operators[getModel()->getBasisIndex(ha->fromIndex)][1]*fockState;
			operators[getModel().getBasisIndex(ha->getFromIndex())][1]*fockState;
			if(fockState.isNull())
				continue;
//			operators[getModel()->getBasisIndex(ha->toIndex)][0]*fockState;
			operators[getModel().getBasisIndex(ha->getToIndex())][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			*subspaceContext.manyBodyModel << HoppingAmplitude(
				ha->getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			);
		}

		for(unsigned int c = 0; c < getModel().getManyBodyContext()->getInteractionAmplitudeSet()->getNumInteractionAmplitudes(); c++){
			FockState<BitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			InteractionAmplitude ia = getModel().getManyBodyContext()->getInteractionAmplitudeSet()->getInteractionAmplitude(c);
			for(int k =  ia.getNumAnnihilationOperators() - 1; k >= 0; k--){
				operators[getModel().getBasisIndex(ia.getAnnihilationOperatorIndex(k))][1]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			for(int k =  ia.getNumCreationOperators() - 1; k >= 0; k--){
				operators[getModel().getBasisIndex(ia.getCreationOperatorIndex(k))][0]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			*subspaceContext.manyBodyModel <<HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			);
		}
	}
	subspaceContext.manyBodyModel->construct();

	delete fockStateMap;
}

template<>
void ExactDiagonalizer::setupManyBodyModel<ExtensiveBitRegister>(unsigned int subspace){
	FockSpace<ExtensiveBitRegister> *fockSpace = getModel().getManyBodyContext()->getFockSpaceExtensiveBitRegister();
	LadderOperator<ExtensiveBitRegister> **operators = fockSpace->getOperators();
	SubspaceContext &subspaceContext = subspaceContexts.at(subspace);
	FockStateMap::FockStateMap<ExtensiveBitRegister> *fockStateMap = fockSpace->createFockStateMap(
		subspaceContext.fockStateRuleSet
	);

	subspaceContext.manyBodyModel.reset(new Model());
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		HoppingAmplitudeSet::Iterator it = getModel().getHoppingAmplitudeSet()->getIterator();
		const HoppingAmplitude *ha;
		while((ha = it.getHA())){
			it.searchNextHA();

			FockState<ExtensiveBitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

//			operators[getModel()->getBasisIndex(ha->fromIndex)][1]*fockState;
			operators[getModel().getBasisIndex(ha->getFromIndex())][1]*fockState;
			if(fockState.isNull())
				continue;
//			operators[getModel()->getBasisIndex(ha->toIndex)][0]*fockState;
			operators[getModel().getBasisIndex(ha->getToIndex())][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			*subspaceContext.manyBodyModel << HoppingAmplitude(
				ha->getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			);
		}

		for(unsigned int c = 0; c < getModel().getManyBodyContext()->getInteractionAmplitudeSet()->getNumInteractionAmplitudes(); c++){
			FockState<ExtensiveBitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			InteractionAmplitude ia = getModel().getManyBodyContext()->getInteractionAmplitudeSet()->getInteractionAmplitude(c);
			for(int k =  ia.getNumAnnihilationOperators() - 1; k >= 0; k--){
				operators[getModel().getBasisIndex(ia.getAnnihilationOperatorIndex(k))][1]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			for(int k =  ia.getNumCreationOperators() - 1; k >= 0; k--){
				operators[getModel().getBasisIndex(ia.getCreationOperatorIndex(k))][0]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			*subspaceContext.manyBodyModel << HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			);
		}
	}
	subspaceContext.manyBodyModel->construct();

	delete fockStateMap;
}

void ExactDiagonalizer::setupManyBodyModel(unsigned int subspace){
	if(getModel().getManyBodyContext()->wrapsBitRegister())
		setupManyBodyModel<BitRegister>(subspace);
	else
		setupManyBodyModel<ExtensiveBitRegister>(subspace);
}

ExactDiagonalizer::SubspaceContext::SubspaceContext(
	initializer_list<const FockStateRule::WrapperRule> rules
){
	for(unsigned int n = 0; n < rules.size(); n++)
		fockStateRuleSet.addFockStateRule(*(rules.begin()+n));

	manyBodyModel = NULL;
	dSolver = NULL;
}

ExactDiagonalizer::SubspaceContext::SubspaceContext(
	vector<FockStateRule::WrapperRule> rules
) :
	manyBodyModel(nullptr),
	dSolver(nullptr)
{
	for(unsigned int n = 0; n < rules.size(); n++)
		fockStateRuleSet.addFockStateRule(rules.at(n));
}

ExactDiagonalizer::SubspaceContext::SubspaceContext(
	const FockStateRuleSet &rules
) :
	manyBodyModel(nullptr),
	dSolver(nullptr)
{
	fockStateRuleSet = rules;
}

ExactDiagonalizer::SubspaceContext::~SubspaceContext(){
}

};	//End of namespace Solver
};	//End of namespace TBTK
