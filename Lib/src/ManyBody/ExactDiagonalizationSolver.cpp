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

	DiagonalizationSolver dSolver;
	dSolver.setModel(manyBodyModel);
	dSolver.run();

	DPropertyExtractor pe(&dSolver);
	pe.setEnergyWindow(-10, 10, 1000);

	Property::DOS *dos = pe.calculateDOS();
	FileWriter::writeDOS(dos);
	delete dos;

	Property::EigenValues *ev = pe.getEigenValues();
	FileWriter::writeEigenValues(ev);
	delete ev;
}

void ExactDiagonalizationSolver::setupManyBodyModel(){
	const int NUM_PARTICLES = 1;

	Streams::out << "Model size: " << model->getBasisSize() << "\n";

	FockSpace<BitRegister> fockSpace(
		model->getAmplitudeSet(),
		model->getStatistics(),
		NUM_PARTICLES
	);

	LadderOperator<BitRegister> **operators = fockSpace.getOperators();

	manyBodyModel = new Model();
	for(unsigned int n = 0; n < fockSpace.getBasisSize(); n++){
		AmplitudeSet::Iterator it = model->getAmplitudeSet()->getIterator();
		const HoppingAmplitude *ha;
		while((ha = it.getHA())){
			it.searchNextHA();

			FockState<BitRegister> fockState = fockSpace.getFockState(n);

			int from = fockSpace.getBasisIndex(fockState);

			operators[model->getBasisIndex(ha->fromIndex)][1]*fockState;
			if(fockState.isNull())
				continue;
			operators[model->getBasisIndex(ha->toIndex)][0]*fockState;
			if(fockState.isNull())
				continue;

			int to = fockSpace.getBasisIndex(fockState);

			manyBodyModel->addHA(HoppingAmplitude(
				ha->getAmplitude(),
				{to},
				{from}
			));
		}

		for(unsigned int c = 0; c < interactionAmplitudeSet->getNumInteractionAmplitudes(); c++){
			FockState<BitRegister> fockState = fockSpace.getFockState(n);

			int from = fockSpace.getBasisIndex(fockState);

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

			int to = fockSpace.getBasisIndex(fockState);

			manyBodyModel->addHA(HoppingAmplitude(
				ia.getAmplitude(),
				{to},
				{from}
			));
		}
	}
	manyBodyModel->construct();
}

};	//End of namespace TBTK
