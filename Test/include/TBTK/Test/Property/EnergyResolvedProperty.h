#include "TBTK/Property/EnergyResolvedProperty.h"
#include "TBTK/IndexException.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(EnergyResolvedProperty, Constructor0){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EQ(
		energyResolvedProperty.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::Real
	);
	EXPECT_EQ(
		energyResolvedProperty.getLowerBound(),
		-10
	);
	EXPECT_EQ(
		energyResolvedProperty.getUpperBound(),
		10
	);
	EXPECT_EQ(
		energyResolvedProperty.getResolution(),
		1000
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 1000;
			c++
		){
			EXPECT_EQ(energyResolvedProperty({n}, c), 0);
		}
	}
}

TEST(EnergyResolvedProperty, Constructor1){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	int data[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data[n] = n;
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000,
		data
	);
	EXPECT_EQ(
		energyResolvedProperty.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::Real
	);
	EXPECT_EQ(
		energyResolvedProperty.getLowerBound(),
		-10
	);
	EXPECT_EQ(
		energyResolvedProperty.getUpperBound(),
		10
	);
	EXPECT_EQ(
		energyResolvedProperty.getResolution(),
		1000
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 1000;
			c++
		){
			EXPECT_EQ(energyResolvedProperty({n}, c), 1000*n + c);
		}
	}
}

TEST(EnergyResolvedProperty, Constructor2){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::Real,
				indexTree,
				-10,
				10,
				2
			);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for EnergyType::FermionicMatsubara with even index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
				indexTree,
				-10,
				9,
				2
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
				indexTree,
				-9,
				10,
				2
			);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::FermionicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		2
	);
	EXPECT_EQ(
		energyResolvedProperty0.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara
	);
	EXPECT_EQ(
		energyResolvedProperty0.getLowerMatsubaraEnergyIndex(),
		-9
	);
	EXPECT_EQ(
		energyResolvedProperty0.getUpperMatsubaraEnergyIndex(),
		9
	);
	EXPECT_EQ(
		energyResolvedProperty0.getNumMatsubaraEnergies(),
		10
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty0.getFundamentalMatsubaraEnergy(),
		2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty0.getLowerMatsubaraEnergy(),
		-9*2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty0.getUpperMatsubaraEnergy(),
		9*2
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 10;
			c++
		){
			EXPECT_EQ(energyResolvedProperty0({n}, c), 0);
		}
	}

	//Fail for EnergyType::BosonicMatsubara with odd index.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
				indexTree,
				-9,
				10,
				2
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
				indexTree,
				-10,
				9,
				2
			);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		2
	);
	EXPECT_EQ(
		energyResolvedProperty1.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara
	);
	EXPECT_EQ(
		energyResolvedProperty1.getLowerMatsubaraEnergyIndex(),
		-10
	);
	EXPECT_EQ(
		energyResolvedProperty1.getUpperMatsubaraEnergyIndex(),
		10
	);
	EXPECT_EQ(
		energyResolvedProperty1.getNumMatsubaraEnergies(),
		11
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty1.getFundamentalMatsubaraEnergy(),
		2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty1.getLowerMatsubaraEnergy(),
		-10*2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty1.getUpperMatsubaraEnergy(),
		10*2
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 11;
			c++
		){
			EXPECT_EQ(energyResolvedProperty1({n}, c), 0);
		}
	}
}

TEST(EnergyResolvedProperty, Constructor3){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::Real,
				indexTree,
				-10,
				10,
				2
			);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for EnergyType::FermionicMatsubara with even index.
	int data0[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		data0[n] = n;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
				indexTree,
				-10,
				9,
				2,
				data0
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
				indexTree,
				-9,
				10,
				2,
				data0
			);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::FermionicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		2,
		data0
	);
	EXPECT_EQ(
		energyResolvedProperty0.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara
	);
	EXPECT_EQ(
		energyResolvedProperty0.getLowerMatsubaraEnergyIndex(),
		-9
	);
	EXPECT_EQ(
		energyResolvedProperty0.getUpperMatsubaraEnergyIndex(),
		9
	);
	EXPECT_EQ(
		energyResolvedProperty0.getNumMatsubaraEnergies(),
		10
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty0.getFundamentalMatsubaraEnergy(),
		2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty0.getLowerMatsubaraEnergy(),
		-9*2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty0.getUpperMatsubaraEnergy(),
		9*2
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 10;
			c++
		){
			EXPECT_EQ(energyResolvedProperty0({n}, c), 10*n + c);
		}
	}

	//Fail for EnergyType::BosonicMatsubara with odd index.
	int data1[3*11];
	for(unsigned int n = 0; n < 3*11; n++)
		data1[n] = n;
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
				indexTree,
				-9,
				10,
				2,
				data1
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			EnergyResolvedProperty<int> energyResolvedProperty(
				EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
				indexTree,
				-10,
				9,
				2,
				data1
			);
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		2,
		data1
	);
	EXPECT_EQ(
		energyResolvedProperty1.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara
	);
	EXPECT_EQ(
		energyResolvedProperty1.getLowerMatsubaraEnergyIndex(),
		-10
	);
	EXPECT_EQ(
		energyResolvedProperty1.getUpperMatsubaraEnergyIndex(),
		10
	);
	EXPECT_EQ(
		energyResolvedProperty1.getNumMatsubaraEnergies(),
		11
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty1.getFundamentalMatsubaraEnergy(),
		2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty1.getLowerMatsubaraEnergy(),
		-10*2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty1.getUpperMatsubaraEnergy(),
		10*2
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 11;
			c++
		){
			EXPECT_EQ(energyResolvedProperty1({n}, c), 11*n + c);
		}
	}
}

TEST(EnergyResolvedProperty, getEnergyType){
	//Already tested throguh
	//EnergyResolvedProperty::Constructor0
	//EnergyResolvedProperty::Constructor1
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
}

TEST(EnergyResolvedProperty, getLowerBound){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//EnergyType::Real.
	//Already tested through
	//EnergyResolvedProperty::Constructor0
	//EnergyResolvedProperty::Constructor1

	//Fail for EnergyType::FermionicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty0.getLowerBound();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty1.getLowerBound();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(EnergyResolvedProperty, getUpperBound){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//EnergyType::Real.
	//Already tested through
	//EnergyResolvedProperty::Constructor0
	//EnergyResolvedProperty::Constructor1

	//Fail for EnergyType::FermionicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty0.getUpperBound();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty1.getUpperBound();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(EnergyResolvedProperty, getResolution){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//EnergyType::Real.
	//Already tested through
	//EnergyResolvedProperty::Constructor0
	//EnergyResolvedProperty::Constructor1

	//Fail for EnergyType::FermionicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty0.getResolution();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty1.getResolution();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(EnergyResolvedProperty, getEnergy){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000
	);
	for(unsigned int n = 0; n < 1000; n++){
		EXPECT_DOUBLE_EQ(
			energyResolvedProperty.getEnergy(n),
			-10 + n*20/999.
		);
	}

	//Fail for EnergyType::FermionicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty0.getResolution();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty1.getResolution();
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(EnergyResolvedProperty, getLowerMatsubaraEnergyIndex){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty.getLowerMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);


	//EnergyType::FermionicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
}

TEST(EnergyResolvedProperty, getUpperMatsubaraEnergyIndex){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty.getUpperMatsubaraEnergyIndex();
		},
		::testing::ExitedWithCode(1),
		""
	);


	//EnergyType::FermionicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
}

TEST(EnergyResolvedProperty, getNumMatsubaraEnergies){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty.getNumMatsubaraEnergies();
		},
		::testing::ExitedWithCode(1),
		""
	);


	//EnergyType::FermionicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
}

TEST(EnergyResolvedProperty, getFundamentalMatsubaraEnergy){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty.getNumMatsubaraEnergies();
		},
		::testing::ExitedWithCode(1),
		""
	);


	//EnergyType::FermionicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
}

TEST(EnergyResolvedProperty, getLowerMatsubaraEnergy){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty.getLowerMatsubaraEnergy();
		},
		::testing::ExitedWithCode(1),
		""
	);


	//EnergyType::FermionicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
}

TEST(EnergyResolvedProperty, getUpperMatsubaraEnergy){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty(
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty.getLowerMatsubaraEnergy();
		},
		::testing::ExitedWithCode(1),
		""
	);


	//EnergyType::FermionicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
}

TEST(EnergyResolvedProperty, getMatsubaraEnergy){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//Fail for EnergyType::Real.
	EXPECT_EXIT(
		{
			EnergyResolvedProperty<int> energyResolvedProperty(
				indexTree,
				-10,
				10,
				1000
			);
			Streams::setStdMuteErr();
			energyResolvedProperty.getLowerMatsubaraEnergy();
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::FermionicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		2.2
	);
	for(
		unsigned int n = 0;
		n < energyResolvedProperty0.getNumMatsubaraEnergies();
		n++
	){
		EXPECT_DOUBLE_EQ(
			real(energyResolvedProperty0.getMatsubaraEnergy(n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			imag(energyResolvedProperty0.getMatsubaraEnergy(n)),
			(-9 + 2*n)*2.2
		);
	}

	//EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		2.2
	);
	for(
		unsigned int n = 0;
		n < energyResolvedProperty1.getNumMatsubaraEnergies();
		n++
	){
		EXPECT_DOUBLE_EQ(
			real(energyResolvedProperty1.getMatsubaraEnergy(n)),
			0
		);
		EXPECT_DOUBLE_EQ(
			imag(energyResolvedProperty1.getMatsubaraEnergy(n)),
			(-10 + 2*n)*2.2
		);
	}
}

};	//End of namespace Property
};	//End of namespace TBTK
