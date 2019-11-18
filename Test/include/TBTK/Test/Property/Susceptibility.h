#include "TBTK/MultiCounter.h"
#include "TBTK/Property/Susceptibility.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

class SusceptibilityTest : public ::testing::Test{
protected:
	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const unsigned int RESOLUTION = 3;

	const int LOWER_MATSUBARA_ENERGY_INDEX = -2;
	const int UPPER_MATSUBARA_ENERGY_INDEX = 2;
	const double FUNDAMENTAL_MATSUBARA_ENERGY = 1.5;

	Susceptibility susceptibility[2];
	IndexTree indexTree;

	void SetUp() override{
		std::complex<double> data[2*2*2*2*3];

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
			for(unsigned int e = 0; e < 3; e++)
				data[3*(2*(2*(2*i + j) + k) + l) + e] = i*j*k*l*e;
		}
		indexTree.generateLinearMap();

		susceptibility[0] = Susceptibility(
			indexTree,
			LOWER_BOUND,
			UPPER_BOUND,
			RESOLUTION,
			data
		);
		susceptibility[1] = Susceptibility(
			indexTree,
			LOWER_MATSUBARA_ENERGY_INDEX,
			UPPER_MATSUBARA_ENERGY_INDEX,
			FUNDAMENTAL_MATSUBARA_ENERGY,
			data
		);
	}
};

//TBTKFeature Property.Susceptibility.construction.1 2019-11-17
TEST_F(SusceptibilityTest, construction1){
	Susceptibility susceptibility0(
		indexTree,
		LOWER_BOUND,
		UPPER_BOUND,
		RESOLUTION
	);

	EXPECT_EQ(susceptibility0.getLowerBound(), LOWER_BOUND);
	EXPECT_EQ(susceptibility0.getUpperBound(), UPPER_BOUND);
	EXPECT_EQ(susceptibility0.getResolution(), RESOLUTION);
	EXPECT_EQ(
		susceptibility0.getEnergyType(),
		EnergyResolvedProperty<std::complex<double>>::EnergyType::Real
	);
}

//TBTKFeature Property.Susceptibility.construction.2 2019-11-17
TEST_F(SusceptibilityTest, construction2){
	Susceptibility susceptibility0(
		indexTree,
		LOWER_MATSUBARA_ENERGY_INDEX,
		UPPER_MATSUBARA_ENERGY_INDEX,
		FUNDAMENTAL_MATSUBARA_ENERGY
	);

	EXPECT_EQ(
		susceptibility0.getLowerMatsubaraEnergyIndex(),
		LOWER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		susceptibility0.getUpperMatsubaraEnergyIndex(),
		UPPER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		susceptibility0.getFundamentalMatsubaraEnergy(),
		FUNDAMENTAL_MATSUBARA_ENERGY
	);
	EXPECT_EQ(
		susceptibility0.getEnergyType(),
		EnergyResolvedProperty<std::complex<double>>::EnergyType::BosonicMatsubara
	);
}

//TBTKFeature Property.Susceptibility.serialization.1 2019-11-17
TEST_F(SusceptibilityTest, serialization1){
	Susceptibility copy(
		susceptibility[0].serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(copy.getLowerBound(), susceptibility[0].getLowerBound());
	EXPECT_EQ(copy.getUpperBound(), susceptibility[0].getUpperBound());
	EXPECT_EQ(copy.getResolution(), susceptibility[0].getResolution());
	EXPECT_EQ(copy.getEnergyType(), susceptibility[0].getEnergyType());

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
				copy(
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				),
				susceptibility[0](
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				)
			);
		}
	}
}

//TBTKFeature Property.Susceptibility.serialization.2 2019-11-17
TEST_F(SusceptibilityTest, serialization2){
	Susceptibility copy(
		susceptibility[1].serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(
		copy.getLowerMatsubaraEnergyIndex(),
		susceptibility[1].getLowerMatsubaraEnergyIndex()
	);
	EXPECT_EQ(
		copy.getUpperMatsubaraEnergyIndex(),
		susceptibility[1].getUpperMatsubaraEnergyIndex()
	);
	EXPECT_EQ(
		copy.getFundamentalMatsubaraEnergy(),
		susceptibility[1].getFundamentalMatsubaraEnergy()
	);
	EXPECT_EQ(copy.getEnergyType(), susceptibility[1].getEnergyType());

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
				copy(
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				),
				susceptibility[1](
					{
						Index({i}),
						Index({j}),
						Index({k}),
						Index({l})
					},
					e
				)
			);
		}
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
