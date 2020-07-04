#include "TBTK/Property/InteractionVertex.h"
#include "TBTK/MultiCounter.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

class InteractionVertexTest : public ::testing::Test{
protected:
	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const unsigned int RESOLUTION = 3;

	const int LOWER_MATSUBARA_ENERGY_INDEX = -2;
	const int UPPER_MATSUBARA_ENERGY_INDEX = 2;
	const double FUNDAMENTAL_MATSUBARA_ENERGY = 1.5;

	InteractionVertex interactionVertex[2];
	IndexTree indexTree;

	void SetUp() override{
		CArray<std::complex<double>> data(2*2*2*2*3);

		MultiCounter<unsigned int> counter(
			{0, 0, 0, 0},
			{2, 2, 2, 2},
			{1, 1, 1, 1}
		);
		for(counter.reset(); !counter.done(); ++counter){
			unsigned int i = counter[0];
			unsigned int j = counter[1];
			unsigned int k = counter[2];
			unsigned int l = counter[3];

			indexTree.add(
				{
					Index({i}),
					Index({j}),
					Index({k}),
					Index({l})
				}
			);
			for(unsigned int e = 0; e < 3; e++){
				data[3*(2*(2*(2*i + j) + k) + l) + e]
					= i*j*k*l*e;
			}
		}
		indexTree.generateLinearMap();

		interactionVertex[0] = InteractionVertex(
			indexTree,
			Range(LOWER_BOUND, UPPER_BOUND, RESOLUTION),
			data
		);
		interactionVertex[1] = InteractionVertex(
			indexTree,
			LOWER_MATSUBARA_ENERGY_INDEX,
			UPPER_MATSUBARA_ENERGY_INDEX,
			FUNDAMENTAL_MATSUBARA_ENERGY,
			data
		);
	}
};

//TBTKFeature Property.InteractionVertex.construction.1 2019-11-14
TEST_F(InteractionVertexTest, construction1){
	InteractionVertex interactionVertex0(
		indexTree,
		Range(LOWER_BOUND, UPPER_BOUND, RESOLUTION)
	);

	EXPECT_EQ(interactionVertex0.getLowerBound(), LOWER_BOUND);
	EXPECT_EQ(interactionVertex0.getUpperBound(), UPPER_BOUND);
	EXPECT_EQ(interactionVertex0.getResolution(), RESOLUTION);
	EXPECT_EQ(
		interactionVertex0.getEnergyType(),
		EnergyResolvedProperty<std::complex<double>>::EnergyType::Real
	);
}

//TBTKFeature Property.InteractionVertex.construction.2 2019-11-14
TEST_F(InteractionVertexTest, construction2){
	InteractionVertex interactionVertex0(
		indexTree,
		LOWER_MATSUBARA_ENERGY_INDEX,
		UPPER_MATSUBARA_ENERGY_INDEX,
		FUNDAMENTAL_MATSUBARA_ENERGY
	);

	EXPECT_EQ(
		interactionVertex0.getLowerMatsubaraEnergyIndex(),
		LOWER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		interactionVertex0.getUpperMatsubaraEnergyIndex(),
		UPPER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		interactionVertex0.getFundamentalMatsubaraEnergy(),
		FUNDAMENTAL_MATSUBARA_ENERGY
	);
	EXPECT_EQ(
		interactionVertex0.getEnergyType(),
		EnergyResolvedProperty<std::complex<double>>::EnergyType::BosonicMatsubara
	);
}

//TBTKFeature Property.InteractionVertex.operatorAdditionAssignment.1 2019-11-14
TEST_F(InteractionVertexTest, operatorAdditionAssignment1){
	interactionVertex[0] += interactionVertex[0];

	MultiCounter<unsigned int> counter(
		{0, 0, 0, 0},
		{2, 2, 2, 2},
		{1, 1, 1, 1}
	);
	for(counter.reset(); !counter.done(); ++counter){
		unsigned int i = counter[0];
		unsigned int j = counter[1];
		unsigned int k = counter[2];
		unsigned int l = counter[3];

		for(unsigned int e = 0; e < 3; e++){
			EXPECT_EQ(
				interactionVertex[0](
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				),
				(double)2*i*j*k*l*e
			);
		}
	}
}

//TBTKFeature Property.InteractionVertex.operatorSubtractionAssignment.1 2019-11-14
TEST_F(InteractionVertexTest, operatorSubtractionAssignment1){
	interactionVertex[0] -= interactionVertex[0];

	MultiCounter<unsigned int> counter(
		{0, 0, 0, 0},
		{2, 2, 2, 2},
		{1, 1, 1, 1}
	);
	for(counter.reset(); !counter.done(); ++counter){
		unsigned int i = counter[0];
		unsigned int j = counter[1];
		unsigned int k = counter[2];
		unsigned int l = counter[3];

		for(unsigned int e = 0; e < 3; e++){
			EXPECT_EQ(
				interactionVertex[0](
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				),
				0.
			);
		}
	}
}

//TBTKFeature Property.InteractionVertex.operatorMultiplicationAssignment.1 2019-11-14
TEST_F(InteractionVertexTest, operatorMultiplicationAssignment1){
	interactionVertex[0] *= 2;

	MultiCounter<unsigned int> counter(
		{0, 0, 0, 0},
		{2, 2, 2, 2},
		{1, 1, 1, 1}
	);
	for(counter.reset(); !counter.done(); ++counter){
		unsigned int i = counter[0];
		unsigned int j = counter[1];
		unsigned int k = counter[2];
		unsigned int l = counter[3];

		for(unsigned int e = 0; e < 3; e++){
			EXPECT_EQ(
				interactionVertex[0](
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				),
				(double)2*i*j*k*l*e
			);
		}
	}
}

//TBTKFeature Property.InteractionVertex.operatorMultiplication.1 2019-11-14
TEST_F(InteractionVertexTest, operatorMultiplication1){
	InteractionVertex result = 2*interactionVertex[0];

	MultiCounter<unsigned int> counter(
		{0, 0, 0, 0},
		{2, 2, 2, 2},
		{1, 1, 1, 1}
	);
	for(counter.reset(); !counter.done(); ++counter){
		unsigned int i = counter[0];
		unsigned int j = counter[1];
		unsigned int k = counter[2];
		unsigned int l = counter[3];

		for(unsigned int e = 0; e < 3; e++){
			EXPECT_EQ(
				result(
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				),
				(double)2*i*j*k*l*e
			);
		}
	}
}

//TBTKFeature Property.InteractionVertex.operatorMultiplication.2 2019-11-14
TEST_F(InteractionVertexTest, operatorMultiplication2){
	InteractionVertex result = interactionVertex[0]*2;

	MultiCounter<unsigned int> counter(
		{0, 0, 0, 0},
		{2, 2, 2, 2},
		{1, 1, 1, 1}
	);
	for(counter.reset(); !counter.done(); ++counter){
		unsigned int i = counter[0];
		unsigned int j = counter[1];
		unsigned int k = counter[2];
		unsigned int l = counter[3];

		for(unsigned int e = 0; e < 3; e++){
			EXPECT_EQ(
				result(
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				),
				(double)2*i*j*k*l*e
			);
		}
	}
}

//TBTKFeature Property.InteractionVertex.operatorDivision.1 2019-11-14
TEST_F(InteractionVertexTest, operatorDivision1){
	interactionVertex[0] /= 2.;

	MultiCounter<unsigned int> counter(
		{0, 0, 0, 0},
		{2, 2, 2, 2},
		{1, 1, 1, 1}
	);
	for(counter.reset(); !counter.done(); ++counter){
		unsigned int i = counter[0];
		unsigned int j = counter[1];
		unsigned int k = counter[2];
		unsigned int l = counter[3];

		for(unsigned int e = 0; e < 3; e++){
			EXPECT_EQ(
				interactionVertex[0](
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				),
				((double)i*j*k*l*e)/2.
			);
		}
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
