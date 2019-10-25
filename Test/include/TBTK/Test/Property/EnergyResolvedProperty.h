#include "TBTK/Property/EnergyResolvedProperty.h"
#include "TBTK/IndexException.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

//Makes protected members public for testing.
template<typename DataType>
class PublicEnergyResolvedProperty : public EnergyResolvedProperty<DataType>{
public:
	using EnergyResolvedProperty<DataType>::EnergyResolvedProperty;

	PublicEnergyResolvedProperty& operator+=(
		const PublicEnergyResolvedProperty &rhs
	){
		EnergyResolvedProperty<DataType>::operator+=(rhs);

		return *this;
	}
	PublicEnergyResolvedProperty& operator-=(
		const PublicEnergyResolvedProperty &rhs
	){
		EnergyResolvedProperty<DataType>::operator-=(rhs);

		return *this;
	}
	PublicEnergyResolvedProperty& operator*=(
		const DataType &rhs
	){
		EnergyResolvedProperty<DataType>::operator*=(rhs);

		return *this;
	}
	PublicEnergyResolvedProperty& operator/=(
		const DataType &rhs
	){
		EnergyResolvedProperty<DataType>::operator/=(rhs);

		return *this;
	}
};

TEST(EnergyResolvedProperty, Constructor0){
	//Just verify that this compiles.
	EnergyResolvedProperty<int> energyResolvedProperty;
}

TEST(EnergyResolvedProperty, Constructor1){
	EnergyResolvedProperty<int> energyResolvedProperty(
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

	for(
		unsigned int n = 0;
		n < 1000;
		n++
	){
		EXPECT_EQ(energyResolvedProperty(n), 0);
	}
}

TEST(EnergyResolvedProperty, Constructor2){
	int data[1000];
	for(unsigned int n = 0; n < 1000; n++)
		data[n] = n;
	EnergyResolvedProperty<int> energyResolvedProperty(
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

	for(
		unsigned int n = 0;
		n < 1000;
		n++
	){
		EXPECT_EQ(energyResolvedProperty(n), n);
	}
}

TEST(EnergyResolvedProperty, Constructor3){
	EnergyResolvedProperty<int> energyResolvedProperty(
		{2, 3, 4},
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

	for(
		unsigned int n = 0;
		n < 2*3*4*1000;
		n++
	){
		EXPECT_EQ(energyResolvedProperty(n), 0);
	}
}

TEST(EnergyResolvedProperty, Constructor4){
	int data[2*3*4*1000];
	for(unsigned int n = 0; n < 2*3*4*1000; n++)
		data[n] = n;
	EnergyResolvedProperty<int> energyResolvedProperty(
		{2, 3, 4},
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

	for(
		unsigned int n = 0;
		n < 2*3*4*1000;
		n++
	){
		EXPECT_EQ(energyResolvedProperty(n), n);
	}
}

TEST(EnergyResolvedProperty, Constructor5){
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

TEST(EnergyResolvedProperty, Constructor6){
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

TEST(EnergyResolvedProperty, Constructor7){
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

TEST(EnergyResolvedProperty, Constructor8){
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

TEST(EnergyResolvedProerty, SerializeToJSON){
	//EnergyType::Real.
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	int data[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data[n] = n;
	EnergyResolvedProperty<int> energyResolvedProperty0(
		indexTree,
		-10,
		10,
		1000,
		data
	);
	EnergyResolvedProperty<int> energyResolvedProperty1(
		energyResolvedProperty0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(
		energyResolvedProperty1.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::Real
	);
	EXPECT_EQ(
		energyResolvedProperty1.getLowerBound(),
		-10
	);
	EXPECT_EQ(
		energyResolvedProperty1.getUpperBound(),
		10
	);
	EXPECT_EQ(
		energyResolvedProperty1.getResolution(),
		1000
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 1000;
			c++
		){
			EXPECT_EQ(energyResolvedProperty1({n}, c), 1000*n + c);
		}
	}

	//EnergyType::FermionicMatsubara.
	int data2[3*10];
	for(unsigned int n = 0; n < 3*10; n++)
		data2[n] = n;
	EnergyResolvedProperty<int> energyResolvedProperty2(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		2,
		data2
	);
	EnergyResolvedProperty<int> energyResolvedProperty3(
		energyResolvedProperty2.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(
		energyResolvedProperty3.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara
	);
	EXPECT_EQ(
		energyResolvedProperty3.getLowerMatsubaraEnergyIndex(),
		-9
	);
	EXPECT_EQ(
		energyResolvedProperty3.getUpperMatsubaraEnergyIndex(),
		9
	);
	EXPECT_EQ(
		energyResolvedProperty3.getNumMatsubaraEnergies(),
		10
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty3.getFundamentalMatsubaraEnergy(),
		2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty3.getLowerMatsubaraEnergy(),
		-9*2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty3.getUpperMatsubaraEnergy(),
		9*2
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 10;
			c++
		){
			EXPECT_EQ(energyResolvedProperty3({n}, c), 10*n + c);
		}
	}

	//EnergyType::BosonicMatsubara.
	int data4[3*11];
	for(unsigned int n = 0; n < 3*11; n++)
		data4[n] = n;
	EnergyResolvedProperty<int> energyResolvedProperty4(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		2,
		data4
	);
	EnergyResolvedProperty<int> energyResolvedProperty5(
		energyResolvedProperty4.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(
		energyResolvedProperty5.getEnergyType(),
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara
	);
	EXPECT_EQ(
		energyResolvedProperty5.getLowerMatsubaraEnergyIndex(),
		-10
	);
	EXPECT_EQ(
		energyResolvedProperty5.getUpperMatsubaraEnergyIndex(),
		10
	);
	EXPECT_EQ(
		energyResolvedProperty5.getNumMatsubaraEnergies(),
		11
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty5.getFundamentalMatsubaraEnergy(),
		2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty5.getLowerMatsubaraEnergy(),
		-10*2
	);
	EXPECT_DOUBLE_EQ(
		energyResolvedProperty5.getUpperMatsubaraEnergy(),
		10*2
	);

	for(int n = 0; n < 3; n++){
		for(
			unsigned int c = 0;
			c < 11;
			c++
		){
			EXPECT_EQ(energyResolvedProperty5({n}, c), 11*n + c);
		}
	}
}

TEST(EnergyResolvedProperty, getEnergyType){
	//Already tested throguh
	//EnergyResolvedProperty::Constructor1
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
	//EnergyResolvedProperty::Constructor4
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6
}

TEST(EnergyResolvedProperty, getLowerBound){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//EnergyType::Real.
	//Already tested through
	//EnergyResolvedProperty::Constructor1
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
	//EnergyResolvedProperty::Constructor4

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
	//EnergyResolvedProperty::Constructor1
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
	//EnergyResolvedProperty::Constructor4

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
	//EnergyResolvedProperty::Constructor1
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
	//EnergyResolvedProperty::Constructor4

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

TEST(EnergyResolvedProperty, getDeltaE){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	//EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty(
		-10,
		10,
		1000
	);
	EXPECT_FLOAT_EQ(energyResolvedProperty.getDeltaE(), 20/999.);
	//Already tested through
	//EnergyResolvedProperty::Constructor1
	//EnergyResolvedProperty::Constructor2
	//EnergyResolvedProperty::Constructor3
	//EnergyResolvedProperty::Constructor4

	//Fail for EnergyType::FermionicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty1(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
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

	//Fail for EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty2(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		1000
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedProperty2.getResolution();
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
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6
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
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6
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
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6
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
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6
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
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6
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
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6

	//EnergyType::BosonicMatsubara.
	//Already tested through
	//EnergyResolvedProperty::Constructor5
	//EnergyResolvedProperty::Constructor6
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
			(-9 + 2*(int)n)*2.2
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
			(-10 + 2*(int)n)*2.2
		);
	}
}

TEST(EnergyResolvedProperty, getNumEnergies){
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
	EXPECT_EQ(
		energyResolvedProperty.getNumEnergies(),
		energyResolvedProperty.getResolution()
	);

	//EnergyType::FermionicMatsubara.
	energyResolvedProperty = EnergyResolvedProperty<int>(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		1
	);
	EXPECT_EQ(
		energyResolvedProperty.getNumEnergies(),
		energyResolvedProperty.getNumMatsubaraEnergies()
	);

	//EnergyType::BosonicMatsubara.
	energyResolvedProperty = EnergyResolvedProperty<int>(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		1
	);
	EXPECT_EQ(
		energyResolvedProperty.getNumEnergies(),
		energyResolvedProperty.getNumMatsubaraEnergies()
	);
}

TEST(EnergyResolvedProperty, energyWindowsAreEqual){
	//EnergyType::Real.
	EnergyResolvedProperty<int> energyResolvedProperty0(
		0,
		1,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty1(
		-0.5,
		1,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty2(
		0,
		1.5,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty3(
		0,
		1,
		11
	);
	EnergyResolvedProperty<int> energyResolvedProperty4(
		0,
		1.01,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty5(
		-0.01,
		1,
		10
	);

	EXPECT_FALSE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty1
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty2
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty3
		)
	);
	EXPECT_TRUE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty4
		)
	);
	EXPECT_TRUE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty5
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty4,
			0.5e-1
		)
	);
	EXPECT_TRUE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty4,
			1e-1
		)
	);

	//EnergyType::FermionicMatsubara.
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.generateLinearMap();

	EnergyResolvedProperty<int> energyResolvedProperty6(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty7(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-11,
		9,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty8(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		11,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty9(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		11
	);
	EnergyResolvedProperty<int> energyResolvedProperty10(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-9,
		9,
		10
	);

	EXPECT_FALSE(
		energyResolvedProperty6.energyWindowsAreEqual(
			energyResolvedProperty7
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty6.energyWindowsAreEqual(
			energyResolvedProperty8
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty6.energyWindowsAreEqual(
			energyResolvedProperty9
		)
	);
	EXPECT_TRUE(
		energyResolvedProperty6.energyWindowsAreEqual(
			energyResolvedProperty10
		)
	);

	//EnergyType::BosonicMatsubara.
	EnergyResolvedProperty<int> energyResolvedProperty11(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty12(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-12,
		10,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty13(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		12,
		10
	);
	EnergyResolvedProperty<int> energyResolvedProperty14(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		11
	);
	EnergyResolvedProperty<int> energyResolvedProperty15(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		-10,
		10,
		10
	);

	EXPECT_FALSE(
		energyResolvedProperty11.energyWindowsAreEqual(
			energyResolvedProperty12
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty11.energyWindowsAreEqual(
			energyResolvedProperty13
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty11.energyWindowsAreEqual(
			energyResolvedProperty14
		)
	);
	EXPECT_TRUE(
		energyResolvedProperty11.energyWindowsAreEqual(
			energyResolvedProperty15
		)
	);

	//Return false for different energy types.
	EXPECT_FALSE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty6
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty0.energyWindowsAreEqual(
			energyResolvedProperty11
		)
	);
	EXPECT_FALSE(
		energyResolvedProperty6.energyWindowsAreEqual(
			energyResolvedProperty11
		)
	);
}

TEST(EnergyResolvedProperty, serialize){
	//Already tested through EnergyResolvedProperty::SerializatToJSON().
}

TEST(EnergyResolvedProperty, additionAssignmentOperator){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	int data0[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data0[n] = n;
	int data1[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data1[n] = 2*n;
	int data2[300];
	for(unsigned int n = 0; n < 300; n++)
		data2[n] = 3*n;
	int data3[2997];
	for(unsigned int n = 0; n < 2997; n++)
		data3[n] = 3*n;

	//EnergyType::Real.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal0(
		indexTree,
		-10,
		10,
		1000,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal1(
		indexTree,
		-10,
		10,
		1000,
		data1
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal2(
		indexTree,
		-9,
		10,
		1000,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal3(
		indexTree,
		-10,
		9,
		1000,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal4(
		indexTree,
		-9,
		10,
		100,
		data2
	);

	energyResolvedPropertyReal0 += energyResolvedPropertyReal1;
	const std::vector<int> &dataReal0
		= energyResolvedPropertyReal0.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataReal0[n], 3*n);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				+= energyResolvedPropertyReal2;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				+= energyResolvedPropertyReal3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				+= energyResolvedPropertyReal4;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::FermionicMatsubara.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		999,
		1.1,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara1(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		999,
		1.1,
		data1
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara2(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-997,
		1001,
		1.1,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara3(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		997,
		1.1,
		data3
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara4(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		999,
		1.2,
		data0
	);

	energyResolvedPropertyFermionicMatsubara0
		+= energyResolvedPropertyFermionicMatsubara1;
	const std::vector<int> &dataFermionicMatsubara0
		= energyResolvedPropertyFermionicMatsubara0.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataFermionicMatsubara0[n], 3*n);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyFermionicMatsubara0
				+= energyResolvedPropertyFermionicMatsubara2;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyFermionicMatsubara0
				+= energyResolvedPropertyFermionicMatsubara3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyFermionicMatsubara0
				+= energyResolvedPropertyFermionicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::BosonicMatsubara.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara0(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1998,
		1.1,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1998,
		1.1,
		data1
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara2(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		2,
		2000,
		1.1,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara3(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1996,
		1.1,
		data3
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara4(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1998,
		1.2,
		data0
	);

	energyResolvedPropertyBosonicMatsubara0
		+= energyResolvedPropertyBosonicMatsubara1;
	const std::vector<int> &dataBosonicMatsubara0
		= energyResolvedPropertyBosonicMatsubara0.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataBosonicMatsubara0[n], 3*n);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyBosonicMatsubara0
				+= energyResolvedPropertyBosonicMatsubara2;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyBosonicMatsubara0
				+= energyResolvedPropertyBosonicMatsubara3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyBosonicMatsubara0
				+= energyResolvedPropertyBosonicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for different energy types.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				+= energyResolvedPropertyFermionicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				+= energyResolvedPropertyBosonicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyFermionicMatsubara0
				+= energyResolvedPropertyBosonicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(EnergyResolvedProperty, subtractionAssignmentOperator){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	int data0[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data0[n] = n;
	int data1[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data1[n] = 2*n;
	int data2[300];
	for(unsigned int n = 0; n < 300; n++)
		data2[n] = 3*n;
	int data3[2997];
	for(unsigned int n = 0; n < 2997; n++)
		data3[n] = 3*n;

	//EnergyType::Real.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal0(
		indexTree,
		-10,
		10,
		1000,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal1(
		indexTree,
		-10,
		10,
		1000,
		data1
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal2(
		indexTree,
		-9,
		10,
		1000,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal3(
		indexTree,
		-10,
		9,
		1000,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal4(
		indexTree,
		-9,
		10,
		100,
		data2
	);

	energyResolvedPropertyReal0 -= energyResolvedPropertyReal1;
	const std::vector<int> &dataReal0
		= energyResolvedPropertyReal0.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataReal0[n], -n);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				-= energyResolvedPropertyReal2;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				-= energyResolvedPropertyReal3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				-= energyResolvedPropertyReal4;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::FermionicMatsubara.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara0(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		999,
		1.1,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara1(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		999,
		1.1,
		data1
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara2(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-997,
		1001,
		1.1,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara3(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		997,
		1.1,
		data3
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara4(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		999,
		1.2,
		data0
	);

	energyResolvedPropertyFermionicMatsubara0
		-= energyResolvedPropertyFermionicMatsubara1;
	const std::vector<int> &dataFermionicMatsubara0
		= energyResolvedPropertyFermionicMatsubara0.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataFermionicMatsubara0[n], -n);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyFermionicMatsubara0
				-= energyResolvedPropertyFermionicMatsubara2;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyFermionicMatsubara0
				-= energyResolvedPropertyFermionicMatsubara3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyFermionicMatsubara0
				-= energyResolvedPropertyFermionicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//EnergyType::BosonicMatsubara.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara0(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1998,
		1.1,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara1(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1998,
		1.1,
		data1
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara2(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		2,
		2000,
		1.1,
		data0
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara3(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1996,
		1.1,
		data3
	);
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara4(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1998,
		1.2,
		data0
	);

	energyResolvedPropertyBosonicMatsubara0
		-= energyResolvedPropertyBosonicMatsubara1;
	const std::vector<int> &dataBosonicMatsubara0
		= energyResolvedPropertyBosonicMatsubara0.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataBosonicMatsubara0[n], -n);

	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyBosonicMatsubara0
				-= energyResolvedPropertyBosonicMatsubara2;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyBosonicMatsubara0
				-= energyResolvedPropertyBosonicMatsubara3;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyBosonicMatsubara0
				-= energyResolvedPropertyBosonicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);

	//Fail for different energy types.
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				-= energyResolvedPropertyFermionicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyReal0
				-= energyResolvedPropertyBosonicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			energyResolvedPropertyFermionicMatsubara0
				-= energyResolvedPropertyBosonicMatsubara4;
		},
		::testing::ExitedWithCode(1),
		""
	);
}

TEST(EnergyResolvedProperty, multiplicationAssignmentOperator){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	int data[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data[n] = n;

	//EnergyType::Real.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal(
		indexTree,
		-10,
		10,
		1000,
		data
	);

	energyResolvedPropertyReal *= 2;
	const std::vector<int> &dataReal
		= energyResolvedPropertyReal.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataReal[n], 2*n);

	//EnergyType::FermionicMatsubara.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		999,
		1.1,
		data
	);

	energyResolvedPropertyFermionicMatsubara *= 3;
	const std::vector<int> &dataFermionicMatsubara
		= energyResolvedPropertyFermionicMatsubara.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataFermionicMatsubara[n], 3*n);

	//EnergyType::BosonicMatsubara.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1998,
		1.1,
		data
	);

	energyResolvedPropertyBosonicMatsubara *= 4;
	const std::vector<int> &dataBosonicMatsubara
		= energyResolvedPropertyBosonicMatsubara.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataBosonicMatsubara[n], 4*n);
}

TEST(EnergyResolvedProperty, divisionAssignmentOperator){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();

	int data[3000];
	for(unsigned int n = 0; n < 3000; n++)
		data[n] = n;

	//EnergyType::Real.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyReal(
		indexTree,
		-10,
		10,
		1000,
		data
	);

	energyResolvedPropertyReal /= 2;
	const std::vector<int> &dataReal
		= energyResolvedPropertyReal.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataReal[n], n/2);

	//EnergyType::FermionicMatsubara.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyFermionicMatsubara(
		EnergyResolvedProperty<int>::EnergyType::FermionicMatsubara,
		indexTree,
		-999,
		999,
		1.1,
		data
	);

	energyResolvedPropertyFermionicMatsubara /= 3;
	const std::vector<int> &dataFermionicMatsubara
		= energyResolvedPropertyFermionicMatsubara.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataFermionicMatsubara[n], n/3);

	//EnergyType::BosonicMatsubara.
	PublicEnergyResolvedProperty<int> energyResolvedPropertyBosonicMatsubara(
		EnergyResolvedProperty<int>::EnergyType::BosonicMatsubara,
		indexTree,
		0,
		1998,
		1.1,
		data
	);

	energyResolvedPropertyBosonicMatsubara /= 4;
	const std::vector<int> &dataBosonicMatsubara
		= energyResolvedPropertyBosonicMatsubara.getData();
	for(unsigned int n = 0; n < 3000; n++)
		EXPECT_EQ(dataBosonicMatsubara[n], n/4);
}

};	//End of namespace Property
};	//End of namespace TBTK
