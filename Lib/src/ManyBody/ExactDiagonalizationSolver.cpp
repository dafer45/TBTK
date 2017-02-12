#include "ExactDiagonalizationSolver.h"
#include "FockSpace.h"
#include "FileWriter.h"
#include "DOS.h"
#include "DiagonalizationSolver.h"
#include "DPropertyExtractor.h"
#include "SumRule.h"
#include "DifferenceRule.h"
#include "WrapperRule.h"

using namespace std;

namespace TBTK{

ExactDiagonalizationSolver::ExactDiagonalizationSolver(
	Model *singleParticleModel,
	InteractionAmplitudeSet *interactionAmplitudeSet
){
	this->singleParticleModel = singleParticleModel;
	this->interactionAmplitudeSet = interactionAmplitudeSet;
}

ExactDiagonalizationSolver::~ExactDiagonalizationSolver(){
	for(unsigned int n = 0; n < manyBodyModels.size(); n++)
		delete manyBodyModels.at(n);
}

unsigned int ExactDiagonalizationSolver::addSubspace(initializer_list<const FockStateRule::WrapperRule> rules){
	unsigned int numSubspaces = subspaceRules.size();

	subspaceRules.push_back(vector<FockStateRule::WrapperRule>());
	for(unsigned int n = 0; n < rules.size(); n++)
		subspaceRules.at(numSubspaces).push_back(*(rules.begin()+n));

	manyBodyModels.push_back(NULL);

	return numSubspaces;
}

void ExactDiagonalizationSolver::run(unsigned int subspace){
	setupManyBodyModel(subspace);
}

template<>
void ExactDiagonalizationSolver::setupManyBodyModel<BitRegister>(unsigned int subspace){
	const int NUM_PARTICLES = singleParticleModel->getBasisSize()/2;

	FockSpace<BitRegister> fockSpace(
		singleParticleModel->getAmplitudeSet(),
		singleParticleModel->getStatistics(),
		NUM_PARTICLES
	);

	LadderOperator<BitRegister> **operators = fockSpace.getOperators();
	FockStateMap::FockStateMap<BitRegister> *fockStateMap = fockSpace.createFockSpaceMap(
		subspaceRules.at(subspace)
	);

	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		Streams::out << n << ":\t";
		fockStateMap->getFockState(n).print();
	}

	manyBodyModels.at(subspace) = new Model();
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

			manyBodyModels.at(subspace)->addHA(HoppingAmplitude(
				ha->getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
//			Streams::out << ha->getAmplitude()*(double)fockState.getPrefactor() << "\t" << to << "\t" << from << "\n";
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

			manyBodyModels.at(subspace)->addHA(HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
//			Streams::out << ia.getAmplitude()*(double)fockState.getPrefactor() << "\t" << to << "\t" << from << "\n";
		}
	}
	manyBodyModels.at(subspace)->construct();

	DiagonalizationSolver dSolver;
	dSolver.setModel(manyBodyModels.at(subspace));
	dSolver.run();

/*	double dosData[1000];
	for(int n = 0; n < 1000; n++)
		dosData[n] = 0;

	const double *eigenValues = dSolver.getEigenValues();
	for(int n = 0; n < manyBodyModel->getBasisSize(); n++){
		FockState<BitRegister> fockState = fockStateMap->getFockState(n);

		operators[0][1]*fockState;
		if(fockState.isNull())
			continue;
		operators[0][0]*fockState;
		if(fockState.isNull())
			continue;

		for(int c = 0; c < manyBodyModel->getBasisSize(); c++){
			double energy = eigenValues[c];
			int i = ((energy+10)/20.)*1000;
			if(i >= 0 && i < 1000){
				dosData[i] += pow(abs(dSolver.getAmplitude(c, {n})), 2);
			}
		}
	}

	Property::DOS *dos = new Property::DOS(-10, 10, 1000, dosData);
	FileWriter::writeDOS(dos);
	delete dos;*/

	DPropertyExtractor pe(&dSolver);
	pe.setEnergyWindow(-10, 10, 1000);

	Property::DOS *dos = pe.calculateDOS();
	FileWriter::writeDOS(dos);
	delete dos;

	Property::EigenValues *ev = pe.getEigenValues();
	FileWriter::writeEigenValues(ev);
	delete ev;

/*	const double *eigenValues = dSolver.getEigenValues();
	for(int n = 0; n < manyBodyModel->getBasisSize(); n++){
		if(eigenValues[n] > -1.5 && eigenValues[n] < -0.5){
			for(int c = 0; c < manyBodyModel->getBasisSize(); c++){
				Streams::out << dSolver.getAmplitude(n, {c}) << "\n";
			}

			Streams::out << "\n";
		}
	}*/
}

template<>
void ExactDiagonalizationSolver::setupManyBodyModel<ExtensiveBitRegister>(unsigned int subspace){
	const int NUM_PARTICLES = 1;

	FockSpace<ExtensiveBitRegister> fockSpace(
		singleParticleModel->getAmplitudeSet(),
		singleParticleModel->getStatistics(),
		NUM_PARTICLES
	);

	LadderOperator<ExtensiveBitRegister> **operators = fockSpace.getOperators();
	FockStateMap::FockStateMap<ExtensiveBitRegister> *fockStateMap = fockSpace.createFockSpaceMap(
		subspaceRules.at(subspace)
	);

	manyBodyModels.at(subspace) = new Model();
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

			manyBodyModels.at(subspace)->addHA(HoppingAmplitude(
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

			manyBodyModels.at(subspace)->addHA(HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
		}
	}
	manyBodyModels.at(subspace)->construct();
}

void ExactDiagonalizationSolver::setupManyBodyModel(unsigned int subspace){
	if(singleParticleModel->getBasisSize() < 32)
		setupManyBodyModel<BitRegister>(subspace);
	else
		setupManyBodyModel<ExtensiveBitRegister>(subspace);
}

};	//End of namespace TBTK
