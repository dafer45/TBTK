#include "ExactDiagonalizationSolver.h"
#include "FockSpace.h"
#include "FileWriter.h"
#include "DOS.h"
#include "DiagonalizationSolver.h"
#include "DPropertyExtractor.h"
#include "SumRule.h"
#include "DifferenceRule.h"
#include "WrapperRule.h"
#include "Timer.h"

using namespace std;

namespace TBTK{

ExactDiagonalizationSolver::ExactDiagonalizationSolver(
	Model *singleParticleModel,
	InteractionAmplitudeSet *interactionAmplitudeSet,
	ManyBodyContext manyBodyContext
) :
	manyBodyContext(manyBodyContext)
{
	this->singleParticleModel = singleParticleModel;
	this->interactionAmplitudeSet = interactionAmplitudeSet;
}

ExactDiagonalizationSolver::~ExactDiagonalizationSolver(){
}

unsigned int ExactDiagonalizationSolver::addSubspace(initializer_list<const FockStateRule::WrapperRule> rules){
	unsigned int numSubspaces = subspaceContexts.size();

	subspaceContexts.push_back(SubspaceContext(rules));

	return numSubspaces;
}

void ExactDiagonalizationSolver::run(unsigned int subspace){
	SubspaceContext &subspaceContext = subspaceContexts.at(subspace);
	if(subspaceContext.manyBodyModel == NULL){
		setupManyBodyModel(subspace);
		subspaceContext.dSolver = new DiagonalizationSolver();
		subspaceContext.dSolver->setModel(subspaceContext.manyBodyModel);
		subspaceContext.dSolver->run();

/*		DPropertyExtractor pe(subspaceContext.dSolver);
		pe.setEnergyWindow(-10., 10., 1000);
		Property::DOS *dos = pe.calculateDOS();
		FileWriter::writeDOS(dos);
		delete dos;*/
	}
}

template<>
void ExactDiagonalizationSolver::setupManyBodyModel<BitRegister>(unsigned int subspace){
	FockSpace<BitRegister> *fockSpace = manyBodyContext.getFockSpaceBitRegister();
	LadderOperator<BitRegister> **operators = fockSpace->getOperators();
	SubspaceContext &subspaceContext = subspaceContexts.at(subspace);
	FockStateMap::FockStateMap<BitRegister> *fockStateMap = fockSpace->createFockStateMap(
		subspaceContext.rules
	);

	subspaceContext.manyBodyModel = new Model();
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		AmplitudeSet::Iterator it = singleParticleModel->getAmplitudeSet()->getIterator();
		const HoppingAmplitude *ha;
		while((ha = it.getHA())){
			it.searchNextHA();

			FockState<BitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			operators[singleParticleModel->getBasisIndex(ha->fromIndex)][1]*fockState;
			if(fockState.isNull())
				continue;
			operators[singleParticleModel->getBasisIndex(ha->toIndex)][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			subspaceContext.manyBodyModel->addHA(HoppingAmplitude(
				ha->getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
		}

		for(unsigned int c = 0; c < interactionAmplitudeSet->getNumInteractionAmplitudes(); c++){
			FockState<BitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			InteractionAmplitude ia = interactionAmplitudeSet->getInteractionAmplitude(c);
			for(int k =  ia.getNumAnnihilationOperators() - 1; k >= 0; k--){
				operators[singleParticleModel->getBasisIndex(ia.getAnnihilationOperatorIndex(k))][1]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			for(int k =  ia.getNumCreationOperators() - 1; k >= 0; k--){
				operators[singleParticleModel->getBasisIndex(ia.getCreationOperatorIndex(k))][0]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			subspaceContext.manyBodyModel->addHA(HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
		}
	}
	subspaceContext.manyBodyModel->construct();

	delete fockStateMap;
}

template<>
void ExactDiagonalizationSolver::setupManyBodyModel<ExtensiveBitRegister>(unsigned int subspace){
	FockSpace<ExtensiveBitRegister> *fockSpace = manyBodyContext.getFockSpaceExtensiveBitRegister();
	LadderOperator<ExtensiveBitRegister> **operators = fockSpace->getOperators();
	SubspaceContext &subspaceContext = subspaceContexts.at(subspace);
	FockStateMap::FockStateMap<ExtensiveBitRegister> *fockStateMap = fockSpace->createFockStateMap(
		subspaceContext.rules
	);

	subspaceContext.manyBodyModel = new Model();
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		AmplitudeSet::Iterator it = singleParticleModel->getAmplitudeSet()->getIterator();
		const HoppingAmplitude *ha;
		while((ha = it.getHA())){
			it.searchNextHA();

			FockState<ExtensiveBitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			operators[singleParticleModel->getBasisIndex(ha->fromIndex)][1]*fockState;
			if(fockState.isNull())
				continue;
			operators[singleParticleModel->getBasisIndex(ha->toIndex)][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			subspaceContext.manyBodyModel->addHA(HoppingAmplitude(
				ha->getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
		}

		for(unsigned int c = 0; c < interactionAmplitudeSet->getNumInteractionAmplitudes(); c++){
			FockState<ExtensiveBitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			InteractionAmplitude ia = interactionAmplitudeSet->getInteractionAmplitude(c);
			for(int k =  ia.getNumAnnihilationOperators() - 1; k >= 0; k--){
				operators[singleParticleModel->getBasisIndex(ia.getAnnihilationOperatorIndex(k))][1]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			for(int k =  ia.getNumCreationOperators() - 1; k >= 0; k--){
				operators[singleParticleModel->getBasisIndex(ia.getCreationOperatorIndex(k))][0]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			subspaceContext.manyBodyModel->addHA(HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
		}
	}
	subspaceContext.manyBodyModel->construct();

	delete fockStateMap;
}

void ExactDiagonalizationSolver::setupManyBodyModel(unsigned int subspace){
	if(manyBodyContext.wrapsBitRegister())
		setupManyBodyModel<BitRegister>(subspace);
	else
		setupManyBodyModel<ExtensiveBitRegister>(subspace);
}

ExactDiagonalizationSolver::SubspaceContext::SubspaceContext(
	initializer_list<const FockStateRule::WrapperRule> rules
){
	for(unsigned int n = 0; n < rules.size(); n++)
		this->rules.push_back(*(rules.begin()+n));

	manyBodyModel = NULL;
	dSolver = NULL;
}

ExactDiagonalizationSolver::SubspaceContext::~SubspaceContext(){
	if(manyBodyModel != NULL)
		delete manyBodyModel;
	if(dSolver != NULL)
		delete dSolver;
}

};	//End of namespace TBTK
