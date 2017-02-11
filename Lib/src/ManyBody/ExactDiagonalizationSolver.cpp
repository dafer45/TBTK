#include "ExactDiagonalizationSolver.h"
#include "FockSpace.h"
#include "FileWriter.h"
#include "DOS.h"
#include "DiagonalizationSolver.h"
#include "DPropertyExtractor.h"

namespace TBTK{

ExactDiagonalizationSolver::ExactDiagonalizationSolver(
	Model *model,
	InteractionAmplitudeSet *interactionAmplitudeSet
){
	this->model = model;
	this->interactionAmplitudeSet = interactionAmplitudeSet;
	manyBodyModel = NULL;
}

ExactDiagonalizationSolver::~ExactDiagonalizationSolver(){
	if(manyBodyModel != NULL)
		delete manyBodyModel;
}

void ExactDiagonalizationSolver::run(){
	setupManyBodyModel();

}

template<>
void ExactDiagonalizationSolver::setupManyBodyModel<BitRegister>(){
	const int NUM_PARTICLES = model->getBasisSize()/2;

	FockSpace<BitRegister> fockSpace(
		model->getAmplitudeSet(),
		model->getStatistics(),
		NUM_PARTICLES
	);

	LadderOperator<BitRegister> **operators = fockSpace.getOperators();
	FockStateMap::FockStateMap<BitRegister> *fockStateMap = fockSpace.createFockSpaceMap(NUM_PARTICLES);

	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		Streams::out << n << ":\t";
		fockStateMap->getFockState(n).print();
	}

	manyBodyModel = new Model();
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		AmplitudeSet::Iterator it = model->getAmplitudeSet()->getIterator();
		const HoppingAmplitude *ha;
		while((ha = it.getHA())){
			it.searchNextHA();

			FockState<BitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			operators[model->getBasisIndex(ha->fromIndex)][1]*fockState;
			if(fockState.isNull())
				continue;
			operators[model->getBasisIndex(ha->toIndex)][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			manyBodyModel->addHA(HoppingAmplitude(
				ha->getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
			Streams::out << ha->getAmplitude()*(double)fockState.getPrefactor() << "\t" << to << "\t" << from << "\n";
		}

		for(unsigned int c = 0; c < interactionAmplitudeSet->getNumInteractionAmplitudes(); c++){
			FockState<BitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			InteractionAmplitude ia = interactionAmplitudeSet->getInteractionAmplitude(c);
			for(int k =  ia.getNumAnnihilationOperators() - 1; k >= 0; k--){
				operators[model->getBasisIndex(ia.getAnnihilationOperatorIndex(k))][1]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			for(int k =  ia.getNumCreationOperators() - 1; k >= 0; k--){
				operators[model->getBasisIndex(ia.getCreationOperatorIndex(k))][0]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			manyBodyModel->addHA(HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
			Streams::out << ia.getAmplitude()*(double)fockState.getPrefactor() << "\t" << to << "\t" << from << "\n";
		}
	}
	manyBodyModel->construct();

	DiagonalizationSolver dSolver;
	dSolver.setModel(manyBodyModel);
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

	const double *eigenValues = dSolver.getEigenValues();
	for(int n = 0; n < manyBodyModel->getBasisSize(); n++){
		if(eigenValues[n] > -1.5 && eigenValues[n] < -0.5){
			for(int c = 0; c < manyBodyModel->getBasisSize(); c++){
				Streams::out << dSolver.getAmplitude(n, {c}) << "\n";
			}

			Streams::out << "\n";
		}
	}
}

template<>
void ExactDiagonalizationSolver::setupManyBodyModel<ExtensiveBitRegister>(){
	const int NUM_PARTICLES = 1;

	FockSpace<ExtensiveBitRegister> fockSpace(
		model->getAmplitudeSet(),
		model->getStatistics(),
		NUM_PARTICLES
	);

	LadderOperator<ExtensiveBitRegister> **operators = fockSpace.getOperators();
	FockStateMap::FockStateMap<ExtensiveBitRegister> *fockStateMap = fockSpace.createFockSpaceMap(NUM_PARTICLES);

	manyBodyModel = new Model();
	for(unsigned int n = 0; n < fockStateMap->getBasisSize(); n++){
		AmplitudeSet::Iterator it = model->getAmplitudeSet()->getIterator();
		const HoppingAmplitude *ha;
		while((ha = it.getHA())){
			it.searchNextHA();

			FockState<ExtensiveBitRegister> fockState = fockStateMap->getFockState(n);

			int from = fockStateMap->getBasisIndex(fockState);

			operators[model->getBasisIndex(ha->fromIndex)][1]*fockState;
			if(fockState.isNull())
				continue;
			operators[model->getBasisIndex(ha->toIndex)][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			manyBodyModel->addHA(HoppingAmplitude(
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
				operators[model->getBasisIndex(ia.getAnnihilationOperatorIndex(k))][1]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			for(int k =  ia.getNumCreationOperators() - 1; k >= 0; k--){
				operators[model->getBasisIndex(ia.getCreationOperatorIndex(k))][0]*fockState;
				if(fockState.isNull())
					break;
			}
			if(fockState.isNull())
				continue;

			int to = fockStateMap->getBasisIndex(fockState);

			manyBodyModel->addHA(HoppingAmplitude(
				ia.getAmplitude()*(double)fockState.getPrefactor(),
				{to},
				{from}
			));
		}
	}
	manyBodyModel->construct();
}

void ExactDiagonalizationSolver::setupManyBodyModel(){
	if(model->getBasisSize() < 32)
		setupManyBodyModel<BitRegister>();
	else
		setupManyBodyModel<ExtensiveBitRegister>();
}

};	//End of namespace TBTK
