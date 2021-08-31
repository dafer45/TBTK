#include "TBTK/IndexException.h"
#include "TBTK/Models/SquareLattice.h"
#include "TBTK/PropertyExtractor/ArnoldiIterator.h"
#include "TBTK/UnitHandler.h"
#include <cmath>
#include <complex>

#include "gtest/gtest.h"

namespace TBTK{
namespace PropertyExtractor{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();
const double EPSILON_10000 = 10000*std::numeric_limits<double>::epsilon();
const double CHEMICAL_POTENTIAL = 1;

#define CREATE_MODEL() \
	const unsigned int SIZE = 10; \
	Model model = Models::SquareLattice({SIZE, SIZE}, {0, -1}); \
	model.construct();

#define SETUP_SOLVER() \
	const int NUM_EIGEN_VALUES = 10; \
	const int NUM_LANCZOS_VECTORS = 20; \
	int MAX_ITERATIONS = 200; \
	Solver::ArnoldiIterator solver; \
	solver.setModel(model); \
	solver.setNumEigenValues(NUM_EIGEN_VALUES); \
	solver.setNumLanczosVectors(NUM_LANCZOS_VECTORS); \
	solver.setMaxIterations(MAX_ITERATIONS);

TEST(ArnoldiIterator, getWaveFunctions0){
	CREATE_MODEL();
	SETUP_SOLVER();
	solver.setCalculateEigenVectors(true);
	solver.run();

	ArnoldiIterator propertyExtractor;
	propertyExtractor.setSolver(solver);
	Property::WaveFunctions waveFunctions
		= propertyExtractor.calculateWaveFunctions(
			{{_a_, _a_}},
			{_a_}
		);
	EXPECT_EQ(waveFunctions.getStates().size(), NUM_EIGEN_VALUES);
}

TEST(ArnoldiIterator, getWaveFunctions1){
	CREATE_MODEL();
	SETUP_SOLVER();
	solver.setCalculateEigenVectors(false);
	solver.run();

	ArnoldiIterator propertyExtractor;
	propertyExtractor.setSolver(solver);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			Property::WaveFunctions waveFunctions
				= propertyExtractor.calculateWaveFunctions(
					{{_a_, _a_}},
					{_a_}
				);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

};	//End of namespace PropertyExtractor
};	//End of namespace TBTK
