#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("Solver");

//! [Solver]
#include "TBTK/Model.h"
#include "TBTK/Solver/Solver.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

using namespace TBTK;

int main(){
	Initialize();

	Model model;
	model.setChemicalPotential(10);
	model.setTemperature(300);

	Solver::Solver solver;
	solver.setModel(model);

	Streams::out << solver.getModel() << "\n";
}
//! [Solver]
