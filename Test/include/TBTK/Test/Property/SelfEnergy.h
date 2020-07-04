#include "TBTK/Property/SelfEnergy.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

class SelfEnergyTest : public ::testing::Test{
protected:
	const double LOWER_BOUND = -10;
	const double UPPER_BOUND = 10;
	const unsigned int RESOLUTION = 3;

	const int LOWER_MATSUBARA_ENERGY_INDEX = -1;
	const int UPPER_MATSUBARA_ENERGY_INDEX = 3;
	const double FUNDAMENTAL_MATSUBARA_ENERGY = 1.5;

	SelfEnergy selfEnergy[2];
	IndexTree indexTree;

	void SetUp() override{
		CArray<std::complex<double>> data(2*2*3);

		for(int x = 0; x < 2; x++){
			for(int y = 0; y < 2; y++){
				indexTree.add({Index({x}), Index({y})});
				for(unsigned int e = 0; e < 3; e++)
					data[3*(2*x + y) + e] = x*y*e;
			}
		}
		indexTree.generateLinearMap();

		selfEnergy[0] = SelfEnergy(
			indexTree,
			Range(LOWER_BOUND, UPPER_BOUND, RESOLUTION),
			data
		);
		selfEnergy[1] = SelfEnergy(
			indexTree,
			LOWER_MATSUBARA_ENERGY_INDEX,
			UPPER_MATSUBARA_ENERGY_INDEX,
			FUNDAMENTAL_MATSUBARA_ENERGY,
			data
		);
	}
};

//TBTKFeature Property.SelfEnergy.construction.1 2019-11-15
TEST_F(SelfEnergyTest, construction1){
	SelfEnergy selfEnergy0(
		indexTree,
		Range(LOWER_BOUND, UPPER_BOUND, RESOLUTION)
	);

	EXPECT_NEAR(selfEnergy0.getLowerBound(), LOWER_BOUND, EPSILON_100);
	EXPECT_NEAR(selfEnergy0.getUpperBound(), UPPER_BOUND, EPSILON_100);
	EXPECT_EQ(selfEnergy0.getResolution(), RESOLUTION);
	EXPECT_EQ(
		selfEnergy0.getEnergyType(),
		EnergyResolvedProperty<std::complex<double>>::EnergyType::Real
	);
}

//TBTKFeature Property.SelfEnergy.construction.2 2019-11-15
TEST_F(SelfEnergyTest, construction2){
	SelfEnergy selfEnergy0(
		indexTree,
		LOWER_MATSUBARA_ENERGY_INDEX,
		UPPER_MATSUBARA_ENERGY_INDEX,
		FUNDAMENTAL_MATSUBARA_ENERGY
	);

	EXPECT_EQ(
		selfEnergy0.getLowerMatsubaraEnergyIndex(),
		LOWER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		selfEnergy0.getUpperMatsubaraEnergyIndex(),
		UPPER_MATSUBARA_ENERGY_INDEX
	);
	EXPECT_EQ(
		selfEnergy0.getFundamentalMatsubaraEnergy(),
		FUNDAMENTAL_MATSUBARA_ENERGY
	);
	EXPECT_EQ(
		selfEnergy0.getEnergyType(),
		EnergyResolvedProperty<std::complex<double>>::EnergyType::FermionicMatsubara
	);
}

//TBTKFeature Property.SelfEnergy.serialization.1 2019-11-17
TEST_F(SelfEnergyTest, serialization1){
	SelfEnergy copy(
		selfEnergy[0].serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(copy.getLowerBound(), selfEnergy[0].getLowerBound());
	EXPECT_EQ(copy.getUpperBound(), selfEnergy[0].getUpperBound());
	EXPECT_EQ(copy.getResolution(), selfEnergy[0].getResolution());
	EXPECT_EQ(copy.getEnergyType(), selfEnergy[0].getEnergyType());

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					copy({Index({x}), Index({y})}, e),
					selfEnergy[0](
						{Index({x}), Index({y})},
						e
					)
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.serialization.2 2019-11-17
TEST_F(SelfEnergyTest, serialization2){
	SelfEnergy copy(
		selfEnergy[1].serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	EXPECT_EQ(
		copy.getLowerMatsubaraEnergyIndex(),
		selfEnergy[1].getLowerMatsubaraEnergyIndex()
	);
	EXPECT_EQ(
		copy.getUpperMatsubaraEnergyIndex(),
		selfEnergy[1].getUpperMatsubaraEnergyIndex()
	);
	EXPECT_EQ(
		copy.getFundamentalMatsubaraEnergy(),
		selfEnergy[1].getFundamentalMatsubaraEnergy()
	);
	EXPECT_EQ(copy.getEnergyType(), selfEnergy[1].getEnergyType());

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					copy({Index({x}), Index({y})}, e),
					selfEnergy[1](
						{Index({x}), Index({y})},
						e
					)
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorAdditionAssignment.1 2019-11-17
TEST_F(SelfEnergyTest, operatorAdditionAssignment1){
	selfEnergy[0] += selfEnergy[0];

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					selfEnergy[0](
						{Index({x}), Index({y})},
						e
					),
					(double)2*x*y*e
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorAddition.1 2019-11-17
TEST_F(SelfEnergyTest, operatorAddition1){
	SelfEnergy result = selfEnergy[0] + selfEnergy[0];

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					result(
						{Index({x}), Index({y})},
						e
					),
					(double)2*x*y*e
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorSubtractionAssignment.1 2019-11-17
TEST_F(SelfEnergyTest, operatorSubtractionAssignment1){
	selfEnergy[0] -= selfEnergy[0];

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					selfEnergy[0](
						{Index({x}), Index({y})},
						e
					),
					0.
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorSubtraction.1 2019-11-17
TEST_F(SelfEnergyTest, operatorSubtraction1){
	SelfEnergy result = selfEnergy[0] - selfEnergy[0];

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					result(
						{Index({x}), Index({y})},
						e
					),
					0.
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorMultiplicationAssignment.1 2019-11-17
TEST_F(SelfEnergyTest, operatorMultiplicationAssignment1){
	selfEnergy[0] *= 2;

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					selfEnergy[0](
						{Index({x}), Index({y})},
						e
					),
					(double)2*x*y*e
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorMultiplication.1 2019-11-17
TEST_F(SelfEnergyTest, operatorMultiplication1){
	SelfEnergy result = 2*selfEnergy[0];

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					result(
						{Index({x}), Index({y})},
						e
					),
					(double)2*x*y*e
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorMultiplication.2 2019-11-17
TEST_F(SelfEnergyTest, operatorMultiplication2){
	SelfEnergy result = selfEnergy[0]*2;

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					result(
						{Index({x}), Index({y})},
						e
					),
					(double)2*x*y*e
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorDivisionAssignment.1 2019-11-17
TEST_F(SelfEnergyTest, operatorDivisionAssignment1){
	selfEnergy[0] /= 2.;

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					selfEnergy[0](
						{Index({x}), Index({y})},
						e
					),
					((double)x*y*e)/2.
				);
			}
		}
	}
}

//TBTKFeature Property.SelfEnergy.operatorDivision.1 2019-11-17
TEST_F(SelfEnergyTest, operatorDivision1){
	SelfEnergy result = selfEnergy[0]/2.;

	for(int x = 0; x < 2; x++){
		for(int y = 0; y < 2; y++){
			for(unsigned int e = 0; e < 3; e++){
				EXPECT_EQ(
					result(
						{Index({x}), Index({y})},
						e
					),
					((double)x*y*e)/2.
				);
			}
		}
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
