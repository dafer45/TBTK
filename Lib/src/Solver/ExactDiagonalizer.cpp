#include "TBTK/Solver/ExactDiagonalizer.h"
#include "TBTK/FockSpace.h"
#include "TBTK/FileWriter.h"
#include "TBTK/FockStateRule/DifferenceRule.h"
#include "TBTK/FockStateRule/SumRule.h"
#include "TBTK/FockStateRule/WrapperRule.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Timer.h"

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
	if(subspaceContext.manyParticleModel == NULL){
		setupManyParticleModel(subspace);
		subspaceContext.dSolver.reset(new Diagonalizer());
		subspaceContext.dSolver->setModel(*subspaceContext.manyParticleModel.get());
		subspaceContext.dSolver->run();
	}
}

template<>
void ExactDiagonalizer::setupManyParticleModel<BitRegister>(unsigned int subspace){
	FockSpace<BitRegister> *fockSpace = getModel().getManyParticleContext()->getFockSpaceBitRegister();
	LadderOperator<BitRegister> const* const* operators = fockSpace->getOperators();
	SubspaceContext &subspaceContext = subspaceContexts.at(subspace);
	FockStateMap::FockStateMap<BitRegister> *fockStateMap = fockSpace->createFockStateMap(
		subspaceContext.fockStateRuleSet
	);

	subspaceContext.manyParticleModel.reset(new Model());
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		for(
			HoppingAmplitudeSet::ConstIterator iterator
				= getModel().getHoppingAmplitudeSet().cbegin();
			iterator != getModel().getHoppingAmplitudeSet().cend();
			++iterator
		){
			FockState<BitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			operators[getModel().getBasisIndex((*iterator).getFromIndex())][1]*fockState;
			if(fockState.isNull())
				continue;
			operators[getModel().getBasisIndex((*iterator).getToIndex())][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			*subspaceContext.manyParticleModel << HoppingAmplitude(
				(*iterator).getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			);
		}

		for(unsigned int c = 0; c < getModel().getManyParticleContext()->getInteractionAmplitudeSet()->getNumInteractionAmplitudes(); c++){
			FockState<BitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			InteractionAmplitude ia = getModel().getManyParticleContext()->getInteractionAmplitudeSet()->getInteractionAmplitude(c);
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

			*subspaceContext.manyParticleModel <<HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			);
		}
	}
	subspaceContext.manyParticleModel->construct();

	delete fockStateMap;
}

template<>
void ExactDiagonalizer::setupManyParticleModel<ExtensiveBitRegister>(unsigned int subspace){
	FockSpace<ExtensiveBitRegister> *fockSpace = getModel().getManyParticleContext()->getFockSpaceExtensiveBitRegister();
	LadderOperator<ExtensiveBitRegister> const* const* operators = fockSpace->getOperators();
	SubspaceContext &subspaceContext = subspaceContexts.at(subspace);
	FockStateMap::FockStateMap<ExtensiveBitRegister> *fockStateMap = fockSpace->createFockStateMap(
		subspaceContext.fockStateRuleSet
	);

	subspaceContext.manyParticleModel.reset(new Model());
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		for(
			HoppingAmplitudeSet::ConstIterator iterator
				= getModel().getHoppingAmplitudeSet().cbegin();
			iterator != getModel().getHoppingAmplitudeSet().cend();
			++iterator
		){
			FockState<ExtensiveBitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			operators[getModel().getBasisIndex((*iterator).getFromIndex())][1]*fockState;
			if(fockState.isNull())
				continue;
			operators[getModel().getBasisIndex((*iterator).getToIndex())][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			*subspaceContext.manyParticleModel << HoppingAmplitude(
				(*iterator).getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			);
		}

		for(unsigned int c = 0; c < getModel().getManyParticleContext()->getInteractionAmplitudeSet()->getNumInteractionAmplitudes(); c++){
			FockState<ExtensiveBitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			InteractionAmplitude ia = getModel().getManyParticleContext()->getInteractionAmplitudeSet()->getInteractionAmplitude(c);
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

			*subspaceContext.manyParticleModel << HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			);
		}
	}
	subspaceContext.manyParticleModel->construct();

	delete fockStateMap;
}

void ExactDiagonalizer::setupManyParticleModel(unsigned int subspace){
	if(getModel().getManyParticleContext()->wrapsBitRegister())
		setupManyParticleModel<BitRegister>(subspace);
	else
		setupManyParticleModel<ExtensiveBitRegister>(subspace);
}

ExactDiagonalizer::SubspaceContext::SubspaceContext(
	initializer_list<const FockStateRule::WrapperRule> rules
){
	for(unsigned int n = 0; n < rules.size(); n++)
		fockStateRuleSet.addFockStateRule(*(rules.begin()+n));

	manyParticleModel = NULL;
	dSolver = NULL;
}

ExactDiagonalizer::SubspaceContext::SubspaceContext(
	vector<FockStateRule::WrapperRule> rules
) :
	manyParticleModel(nullptr),
	dSolver(nullptr)
{
	for(unsigned int n = 0; n < rules.size(); n++)
		fockStateRuleSet.addFockStateRule(rules.at(n));
}

ExactDiagonalizer::SubspaceContext::SubspaceContext(
	const FockStateRuleSet &rules
) :
	manyParticleModel(nullptr),
	dSolver(nullptr)
{
	fockStateRuleSet = rules;
}

ExactDiagonalizer::SubspaceContext::~SubspaceContext(){
}

};	//End of namespace Solver
};	//End of namespace TBTK
