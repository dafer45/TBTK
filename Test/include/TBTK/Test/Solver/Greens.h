#include "TBTK/Matrix.h"
#include "TBTK/Model.h"
#include "TBTK/MultiCounter.h"
#include "TBTK/Property/TransmissionRate.h"
#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/Range.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Solver/Greens.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Solver{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

TEST(Greens, Destructor){
	//Not testable on its own.
}

TEST(Greens, setGreensFunction){
	//Tested through Greens::getGreensFunction().
}

TEST(Greens, getGreensFunction){
	Property::GreensFunction greensFunction;
	Greens solver;
	solver.setGreensFunction(greensFunction);
	EXPECT_TRUE(&solver.getGreensFunction() == &greensFunction);
}

TEST(Greens, addSelfEnergy){
	double LOWER_BOUND = -5;
	double UPPER_BOUND = 5;
	const int RESOLUTION = 10;

	////////////////////////////////////
	// Test for single block problem. //
	////////////////////////////////////

	//Setup the model.
	Model modelA;
	modelA.setVerbose(false);
	modelA << HoppingAmplitude(-1, {1, 0}, {0, 0}) + HC;
	modelA << HoppingAmplitude(-1, {0, 1}, {0, 0}) + HC;
	modelA << HoppingAmplitude(-1, {1, 1}, {0, 1}) + HC;
	modelA << HoppingAmplitude(-1, {1, 1}, {1, 0}) + HC;
	modelA.construct();

	//Setup and run the solver.
	Diagonalizer diagonalizer;
	diagonalizer.setVerbose(false);
	diagonalizer.setModel(modelA);
	diagonalizer.run();

	//Setup the property extractor and calculate the non-interacting
	//Green's function.
	PropertyExtractor::Diagonalizer propertyExtractorA;
	propertyExtractorA.setSolver(diagonalizer);
	propertyExtractorA.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	double infinitesimalA = 1;
	propertyExtractorA.setEnergyInfinitesimal(infinitesimalA);
	Property::GreensFunction greensFunctionA0
		= propertyExtractorA.calculateGreensFunction(
			{{{IDX_ALL, IDX_ALL}, {IDX_ALL, IDX_ALL}}},
			Property::GreensFunction::Type::Retarded
		);

	//Setup the self-energy.
	IndexTree memoryLayoutA;
	for(int x = 0; x < 2; x++)
		for(int y = 0; y < 2; y++)
			for(int xp = 0; xp < 2; xp++)
				for(int yp = 0; yp < 2; yp++)
					memoryLayoutA.add({{x, y}, {xp, yp}});
	memoryLayoutA.generateLinearMap();
	Property::SelfEnergy selfEnergyA(
		memoryLayoutA,
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	for(unsigned int n = 0; n < RESOLUTION; n++){
		for(int x = 0; x < 2; x++){
			for(int y = 0; y < 2; y++){
				for(int xp = 0; xp < 2; xp++){
					for(int yp = 0; yp < 2; yp++){
						selfEnergyA(
							{{x, y}, {xp, yp}},
							n
						) = x + y + xp + yp + n;
					}
				}
			}
		}
	}

	//Setup the Green's solver and calculate the interacting Green's
	//function.
	Greens solverA;
	solverA.setVerbose(false);
	solverA.setModel(modelA);
	solverA.setGreensFunction(greensFunctionA0);
	Property::GreensFunction greensFunctionA
		= solverA.calculateInteractingGreensFunction(
			selfEnergyA
		);

	//Check the interacting Green's function against a reference
	//calculation.
	double dE = (UPPER_BOUND - LOWER_BOUND)/(RESOLUTION - 1);
	for(
		unsigned int e = 0;
		e < RESOLUTION;
		e++
	){
		//Multi counter used to loop over x, y, xp, and yp indices.
		MultiCounter<int> counter(
			{0, 0, 0, 0},
			{2, 2, 2, 2},
			{1, 1, 1, 1}
		);

		//Create reference Green's function (will be inverted
		//eventually, so should currently be seen as the denominator of
		//the Green's funtion).
		Matrix<std::complex<double>> referenceGreensFunction(4, 4);

		//Add the self energy to the denominator of the reference
		//Green's function.
		for(counter.reset(); !counter.done(); ++counter){
			int x = counter[0];
			int y = counter[1];
			int xp = counter[2];
			int yp = counter[3];

			int row = 2*x + y;
			int column = 2*xp + yp;

			referenceGreensFunction.at(row, column)
				= -(double)(x + y + xp + yp + e);
		}

		//Subtract the Hamiltonian from the denominator of the
		//reference Green's function.
		//
		//Index linearization:
		//(0, 0) -> 0,
		//(0, 1) -> 1,
		//(1, 0) -> 2,
		//(1, 1) -> 3.
		referenceGreensFunction.at(2, 0) += 1;
		referenceGreensFunction.at(0, 2) += 1;
		referenceGreensFunction.at(1, 0) += 1;
		referenceGreensFunction.at(0, 1) += 1;
		referenceGreensFunction.at(3, 1) += 1;
		referenceGreensFunction.at(1, 3) += 1;
		referenceGreensFunction.at(3, 2) += 1;
		referenceGreensFunction.at(2, 3) += 1;

		//Add the energy and infinitesimal i\delta to the denominator
		//of the reference Green's function.
		double E = LOWER_BOUND + e*dE;
		for(unsigned int n = 0; n < 4; n++){
			referenceGreensFunction.at(n, n)
				+= E + std::complex<double>(0, 1)*infinitesimalA;
		}

		//Invert to get the reference Green's function.
		referenceGreensFunction.invert();

		//Compare the interacting Green's function calculated using the
		//Green's solver to the reference Green's funtion.
		for(counter.reset(); !counter.done(); ++counter){
			int x = counter[0];
			int y = counter[1];
			int xp = counter[2];
			int yp = counter[3];

			//Real part.
			EXPECT_NEAR(
				real(greensFunctionA({{x, y}, {xp, yp}}, e)),
				real(
					referenceGreensFunction.at(
						2*x + y,
						2*xp + yp
					)
				),
				EPSILON_100
			);

			//Imaginary part.
			EXPECT_NEAR(
				imag(greensFunctionA({{x, y}, {xp, yp}}, e)),
				imag(
					referenceGreensFunction.at(
						2*x + y,
						2*xp + yp
					)
				),
				EPSILON_100
			);
		}
	}

	///////////////////////////////////////
	//Test for block structured problem. //
	///////////////////////////////////////
	LOWER_BOUND = -100;
	UPPER_BOUND = 100;

	//Setup the model.
	Model modelB;
	modelB.setVerbose(false);
	for(int k = 0; k < 10; k++){
		modelB << HoppingAmplitude(k*k-10,	{k, 0},	{k, 0});
		modelB << HoppingAmplitude(-k*k-10,	{k, 1},	{k, 1});
		modelB << HoppingAmplitude(1, {k, 0}, {k, 1}) + HC;
	}
	modelB.construct();

	//Setup and run the solver.
	BlockDiagonalizer blockDiagonalizer;
	blockDiagonalizer.setVerbose(false);
	blockDiagonalizer.setModel(modelB);
	blockDiagonalizer.run();


	//Setup the property extractor and calculate the non-interacting
	//Green's function.
	PropertyExtractor::BlockDiagonalizer propertyExtractorB;
	propertyExtractorB.setSolver(blockDiagonalizer);
	propertyExtractorB.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	propertyExtractorB.setEnergyInfinitesimal(10);
	std::vector<Index> patterns;
	for(int k = 0; k < 10; k++)
		patterns.push_back({{k, IDX_ALL}, {k, IDX_ALL}});
	Property::GreensFunction greensFunctionB0
		= propertyExtractorB.calculateGreensFunction(
			patterns,
			Property::GreensFunction::Type::Retarded
		);

	//Setup the self-energy.
	IndexTree memoryLayoutB;
	for(int k = 0; k < 10; k++)
		for(int m = 0; m < 2; m++)
			for(int n = 0; n < 2; n++)
				memoryLayoutB.add({{k, m}, {k, n}});
	memoryLayoutB.generateLinearMap();
	Property::SelfEnergy selfEnergyB(
		memoryLayoutB,
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	for(int k = 0; k < 10; k++){
		for(int m = 0; m < 2; m++){
			for(int n = 0; n < 2; n++){
				for(unsigned int e = 0; e < RESOLUTION; e++){
					selfEnergyB({{k, m}, {k, n}}, e)
						= k + m + n + e;
				}
			}
		}
	}

	//Setup the Green's solver and calculate the interacting Green's
	//function.
	Greens solverB;
	solverB.setVerbose(false);
	solverB.setModel(modelB);
	solverB.setGreensFunction(greensFunctionB0);
	Property::GreensFunction greensFunction
		= solverB.calculateInteractingGreensFunction(
			selfEnergyB
		);

	//Check the interacting Green's function calculated for every block at
	//once against the Green's function calculated for a single block at a
	//time. (The single block calculation was tested above and is here used
	//to verify the block calculation).
	for(int k = 0; k < 10; k++){
		//Setup reference model.
		Model referenceModel;
		referenceModel.setVerbose(false);
		referenceModel << HoppingAmplitude(k*k-10,	{0},	{0});
		referenceModel << HoppingAmplitude(-k*k-10,	{1},	{1});
		referenceModel << HoppingAmplitude(1, {0}, {1}) + HC;
		referenceModel.construct();

		//Setup reference solver.
		Diagonalizer referenceDiagonalizer;
		referenceDiagonalizer.setVerbose(false);
		referenceDiagonalizer.setModel(referenceModel);
		referenceDiagonalizer.run();

		//Setup reference solver and extract the reference
		//non-interacting Green's function.
		PropertyExtractor::Diagonalizer referencePropertyExtractor;
		referencePropertyExtractor.setSolver(referenceDiagonalizer);
		referencePropertyExtractor.setEnergyWindow(
			LOWER_BOUND,
			UPPER_BOUND,
			RESOLUTION
		);
		referencePropertyExtractor.setEnergyInfinitesimal(10);
		Property::GreensFunction referenceGreensFunction0
			= referencePropertyExtractor.calculateGreensFunction(
				{{Index({IDX_ALL}), Index({IDX_ALL})}},
				Property::GreensFunction::Type::Retarded
			);

		//Setup the self-energy.
		IndexTree referenceMemoryLayout;
		for(int m = 0; m < 2; m++){
			for(int n = 0; n < 2; n++){
				referenceMemoryLayout.add(
					{Index({m}), Index({n})}
				);
			}
		}
		referenceMemoryLayout.generateLinearMap();
		Property::SelfEnergy referenceSelfEnergy(
			referenceMemoryLayout,
			LOWER_BOUND,
			UPPER_BOUND,
			RESOLUTION
		);
		for(int m = 0; m < 2; m++){
			for(int n = 0; n < 2; n++){
				for(unsigned int e = 0; e < RESOLUTION; e++){
					referenceSelfEnergy(
						{Index({m}), Index({n})},
						e
					) = k + m + n + e;
				}
			}
		}

		//Setup the Green's solver and calculate the interacting Green's
		//function.
		Greens referenceSolver;
		referenceSolver.setVerbose(false);
		referenceSolver.setModel(referenceModel);
		referenceSolver.setGreensFunction(referenceGreensFunction0);
		Property::GreensFunction referenceGreensFunction
			= referenceSolver.calculateInteractingGreensFunction(
				referenceSelfEnergy
			);

		for(int m = 0; m < 2; m++){
			for(int n = 0; n < 2; n++){
				for(unsigned int e = 0; e < RESOLUTION; e++){
					EXPECT_NEAR(
						real(
							greensFunction(
								{
									{k, m},
									{k, n}
								},
								e
							)
						),
						real(
							referenceGreensFunction(
								{
									Index({m}),
									Index({n})
								},
								e
							)
						),
						EPSILON_100
					);
					EXPECT_NEAR(
						imag(
							greensFunction(
								{
									{k, m},
									{k, n}
								},
								e
							)
						),
						imag(
							referenceGreensFunction(
								{
									Index({m}),
									Index({n})
								},
								e
							)
						),
						EPSILON_100
					);
				}
			}
		}
	}
}

TEST(Greens, calculateSpectralFunction){
	double LOWER_BOUND = -5;
	double UPPER_BOUND = 5;
	const int RESOLUTION = 10;

	////////////////////////////////////
	// Test for single block problem. //
	////////////////////////////////////

	//Setup the model.
	Model model;
	model.setVerbose(false);
	model << HoppingAmplitude(-1, {1, 0}, {0, 0}) + HC;
	model << HoppingAmplitude(-1, {0, 1}, {0, 0}) + HC;
	model << HoppingAmplitude(-1, {1, 1}, {0, 1}) + HC;
	model << HoppingAmplitude(-1, {1, 1}, {1, 0}) + HC;
	model.construct();

	//Setup and run the solver.
	Diagonalizer diagonalizer;
	diagonalizer.setVerbose(false);
	diagonalizer.setModel(model);
	diagonalizer.run();

	//Setup the property extractor and calculate the non-interacting
	//Green's function.
	PropertyExtractor::Diagonalizer propertyExtractor;
	propertyExtractor.setSolver(diagonalizer);
	propertyExtractor.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	double infinitesimal = 1;
	propertyExtractor.setEnergyInfinitesimal(infinitesimal);
	Property::GreensFunction greensFunction
		= propertyExtractor.calculateGreensFunction(
			{{{IDX_ALL, IDX_ALL}, {IDX_ALL, IDX_ALL}}},
			Property::GreensFunction::Type::Retarded
		);

	Greens solver;
	solver.setGreensFunction(greensFunction);
	Property::SpectralFunction spectralFunction
		= solver.calculateSpectralFunction();

	EXPECT_EQ(
		spectralFunction.getLowerBound(),
		greensFunction.getLowerBound()
	);
	EXPECT_EQ(
		spectralFunction.getUpperBound(),
		greensFunction.getUpperBound()
	);
	EXPECT_EQ(
		spectralFunction.getResolution(),
		greensFunction.getResolution()
	);

	std::complex<double> i(0, 1);
	IndexTree indexTree = model.getHoppingAmplitudeSet().getIndexTree();
	for(auto iterator0 : indexTree){
		for(auto iterator1 : indexTree){
			for(
				unsigned int n = 0;
				n < spectralFunction.getResolution();
				n++
			){
				EXPECT_EQ(
					spectralFunction(
						{iterator0, iterator1},
						n
					),
					i*(
						greensFunction(
							{iterator0, iterator1},
							n
						) - conj(
							 greensFunction(
								{
									iterator1,
									iterator0
								},
								n
							)
						)
					)
				);
			}
		}
	}
}

TEST(Greens, calculateTransmission){
	const double LOWER_BOUND = -1;
	const double UPPER_BOUND = 1;
	const int RESOLUTION = 10;
	std::complex<double> i(0, 1);

	//Setup Model to test against.
	Model model;
	model.setVerbose(false);
	for(int x = 0; x < 3; x++)
		model << HoppingAmplitude(-1, {x+1}, {x}) + HC;
	model.construct();

	//Calculate the bare Green's function.
	Diagonalizer solver0;
	solver0.setVerbose(false);
	solver0.setModel(model);
	solver0.run();
	PropertyExtractor::Diagonalizer propertyExtractor0;
	propertyExtractor0.setSolver(solver0);
	propertyExtractor0.setEnergyInfinitesimal(0);
	propertyExtractor0.setEnergyWindow(
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	Property::GreensFunction greensFunction0
		= propertyExtractor0.calculateGreensFunction(
			{{Index({IDX_ALL}), Index({IDX_ALL})}}
		);

	//Setup self-energies.
	IndexTree indexTree;
	for(int x = 0; x < 4; x++)
		for(int xp = 0; xp < 4; xp++)
			indexTree.add({Index({x}), Index({xp})});
	indexTree.generateLinearMap();
	Property::SelfEnergy selfEnergy0(
		indexTree,
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	Property::SelfEnergy selfEnergy1(
		indexTree,
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);
	Range energies(LOWER_BOUND, UPPER_BOUND, RESOLUTION, true, true);
	for(unsigned int n = 0; n < RESOLUTION; n++){
		selfEnergy0({Index({0}), Index({0})}, n) = i*1.*energies[n];
		selfEnergy0({Index({0}), Index({1})}, n) = i*2.*energies[n];
		selfEnergy0({Index({1}), Index({0})}, n) = i*3.*energies[n];
		selfEnergy0({Index({1}), Index({1})}, n) = i*4.*energies[n];
		selfEnergy1({Index({2}), Index({2})}, n) = i*5.*energies[n];
		selfEnergy1({Index({2}), Index({3})}, n) = i*6.*energies[n];
		selfEnergy1({Index({3}), Index({2})}, n) = i*7.*energies[n];
		selfEnergy1({Index({3}), Index({3})}, n) = i*8.*energies[n];
	}

	//Calculate the full Green's function.
	Greens solver1;
	solver1.setVerbose(false);
	solver1.setModel(model);
	solver1.setGreensFunction(greensFunction0);
	Property::GreensFunction greensFunction
		= solver1.calculateInteractingGreensFunction(
			selfEnergy0 + selfEnergy1
		);

	//Calculate the transmission rate.
	solver1.setGreensFunction(greensFunction);
	Property::TransmissionRate transmissionRate
		= solver1.calculateTransmissionRate(selfEnergy0, selfEnergy1);

	//Check that the energy window is correct.
	EXPECT_DOUBLE_EQ(transmissionRate.getLowerBound(), LOWER_BOUND);
	EXPECT_DOUBLE_EQ(transmissionRate.getUpperBound(), UPPER_BOUND);
	EXPECT_DOUBLE_EQ(transmissionRate.getResolution(), RESOLUTION);

	//Check that the transmission rate has been calculated corectly.
	for(int n = 0; n < RESOLUTION; n++){
		//Create Hamiltonian and self-energy matrices to perform
		//reference calculation with.
		SparseMatrix<std::complex<double>> H(
			SparseMatrix<std::complex<double>>::StorageFormat::CSC,
			4,
			4
		);
		SparseMatrix<std::complex<double>> s0(
			SparseMatrix<std::complex<double>>::StorageFormat::CSC,
			4,
			4
		);
		SparseMatrix<std::complex<double>> s1(
			SparseMatrix<std::complex<double>>::StorageFormat::CSC,
			4,
			4
		);

		//Initialize the Hamiltonian and self-energies. These values
		//must correspond to the values fed to the Model and SelfEnergy
		//objects above.
		for(unsigned int x = 0; x < 3; x++){
			H.add(x, x+1, -1);
			H.add(x+1, x, -1);
		}
		s0.add(0, 0, i*1.*energies[n]);
		s0.add(0, 1, i*2.*energies[n]);
		s0.add(1, 0, i*3.*energies[n]);
		s0.add(1, 1, i*4.*energies[n]);
		s1.add(2, 2, i*5.*energies[n]);
		s1.add(2, 3, i*6.*energies[n]);
		s1.add(3, 2, i*7.*energies[n]);
		s1.add(3, 3, i*8.*energies[n]);

		H.construct();
		s0.construct();
		s1.construct();

		//Add self energies to the Hamiltonian.
		H = H + s0 + s1;

		//Copy the sparse matrix to a dense matrix g to prepare for
		//inversion.
		Matrix<std::complex<double>> g(4, 4);
		for(unsigned int x = 0; x < 4; x++){
			for(unsigned int xp = 0; xp < 4; xp++){
				g.at(x, xp) = 0;
			}
		}
		const unsigned int *columnPointers = H.getCSCColumnPointers();
		const unsigned int *rows = H.getCSCRows();
		const std::complex<double> *values = H.getCSCValues();
		for(unsigned int col = 0; col < H.getNumColumns(); col++){
			for(
				unsigned int n = columnPointers[col];
				n < columnPointers[col+1];
				n++
			){
				g.at(rows[n], col) -= values[n];
			}
		}
		//Add energies so that g takes the form (E - H - s0 - s1).
		for(unsigned int x = 0; x < 4; x++)
			g.at(x, x) += energies[n];
		//Invert to get the full Green's function.
		g.invert();

		//Copy the full Green's function back to a SparseMatrix.
		SparseMatrix<std::complex<double>> G(
			SparseMatrix<std::complex<double>>::StorageFormat::CSC,
			4,
			4
		);
		for(unsigned int x = 0; x < 4; x++)
			for(unsigned int xp = 0; xp < 4; xp++)
				G.add(x, xp, g.at(x, xp));
		G.construct();

		//Calculate broadenings from self-energies.
		SparseMatrix<std::complex<double>> broadening0 = i*(s0 - s0.hermitianConjugate());
		SparseMatrix<std::complex<double>> broadening1 = i*(s1 - s1.hermitianConjugate());

		//Calculate the broduct Gamma_0*G*Gamma_1*G^{\dagger}.
		SparseMatrix<std::complex<double>> product = broadening0*G*broadening1*G.hermitianConjugate();

		//Calculate Tr[Gamma_0*G*Gamma1*G^{\dagger}].
		double trace = real(product.trace());

		//Compare the two results.
		EXPECT_NEAR(trace, transmissionRate(n), EPSILON_100);
	}
}

};	//End of namespace Solver
};	//End of namespace TBTK
