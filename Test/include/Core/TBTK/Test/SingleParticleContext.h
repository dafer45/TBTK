#include "TBTK/SingleParticleContext.h"

#include "TBTK/BasicState.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(SingleParticleContext, Constructor){
	//Check default values.
	SingleParticleContext singleParticleContext;
	EXPECT_EQ(singleParticleContext.getStatistics(), Statistics::FermiDirac);
}

TEST(SingleParticleContext, ConstructorCapacity){
	//Not testable on its own.
}

TEST(SingleParticleContext, CopyConstructor){
	//Not testable on its own.
}

TEST(SingleParticleContext, MoveConstructor){
	//Not testable on its own.
}

//TODO
//...
TEST(SingleParticleContext, SerializeToJSON){
}

TEST(SingleParticleContext, Destructor){
	//Not testable on its own.
}

TEST(SingleParticleContext, operatorAsignment){
	//Not testable on its own.
}

TEST(SingleParticleContext, operatorMoveAsignment){
	//Not testable on its own.
}

TEST(SingleParticleContext, setStatistics){
	//Tested through SingleParticleContext::getStatistics().
}

TEST(SingleParticleContext, getStatistics){
	SingleParticleContext singleParticleContext;

	//Check that it is possible to set and get Statistics::BoseEinstein.
	singleParticleContext.setStatistics(Statistics::BoseEinstein);
	EXPECT_EQ(singleParticleContext.getStatistics(), Statistics::BoseEinstein);

	//Check that it is possible to set and get Statistics::FermiDirac.
	singleParticleContext.setStatistics(Statistics::FermiDirac);
	EXPECT_EQ(singleParticleContext.getStatistics(), Statistics::FermiDirac);
}

//Helper class for testing
//SingleParticleContext::generateHoppingAmplitudeSet().
class HoppingAmplitudeCallback : public HoppingAmplitude::AmplitudeCallback{
public:
	HoppingAmplitudeCallback(
		BasisStateSet &basisStateSet
	) : basisStateSet(basisStateSet){
	}

	virtual std::complex<double> getHoppingAmplitude(
		const Index &to,
		const Index &from
	) const{
		return basisStateSet.get(to).getMatrixElement(
			basisStateSet.get(from)
		);
	}
private:
	BasisStateSet &basisStateSet;
};

TEST(SingleParticleContext, generateHoppingAmplitudeSet){
	SingleParticleContext singleParticleContext;
	BasisStateSet &basisStateSet
		= singleParticleContext.getBasisStateSet();

	BasicState state0({1, 2});
	BasicState state1({2, 3});

	state0.addMatrixElement(1, {1, 2});
	state0.addMatrixElement(2, {2, 3});

	state1.addMatrixElement(2, {1, 2});
	state1.addMatrixElement(3, {2, 3});

	basisStateSet.add(state0);
	basisStateSet.add(state1);

	HoppingAmplitudeCallback hoppingAmplitudeCallback(basisStateSet);
	singleParticleContext.generateHoppingAmplitudeSet(
		hoppingAmplitudeCallback
	);

	HoppingAmplitudeSet &hoppingAmplitudeSet
		= singleParticleContext.getHoppingAmplitudeSet();
	hoppingAmplitudeSet.construct();
	HoppingAmplitudeSet::ConstIterator iterator
		= hoppingAmplitudeSet.cbegin();

	EXPECT_TRUE(iterator != hoppingAmplitudeSet.cend());
	EXPECT_TRUE((*iterator).getToIndex().equals({1, 2}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({1, 2}));
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag((*iterator).getAmplitude()), 0);

	++iterator;
	EXPECT_TRUE(iterator != hoppingAmplitudeSet.cend());
	EXPECT_TRUE((*iterator).getToIndex().equals({2, 3}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({1, 2}));
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_DOUBLE_EQ(imag((*iterator).getAmplitude()), 0);

	++iterator;
	EXPECT_TRUE(iterator != hoppingAmplitudeSet.cend());
	EXPECT_TRUE((*iterator).getToIndex().equals({1, 2}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({2, 3}));
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_DOUBLE_EQ(imag((*iterator).getAmplitude()), 0);

	++iterator;
	EXPECT_TRUE(iterator != hoppingAmplitudeSet.cend());
	EXPECT_TRUE((*iterator).getToIndex().equals({2, 3}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({2, 3}));
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 3);
	EXPECT_DOUBLE_EQ(imag((*iterator).getAmplitude()), 0);

	++iterator;
	EXPECT_TRUE(iterator == hoppingAmplitudeSet.cend());
}

//Helper class for testing
//SingleParticleContext::generateOverlapAmplitudeSet().
class OverlapAmplitudeCallback : public OverlapAmplitude::AmplitudeCallback{
public:
	OverlapAmplitudeCallback(
		BasisStateSet &basisStateSet
	) : basisStateSet(basisStateSet){
	}

	virtual std::complex<double> getOverlapAmplitude(
		const Index &to,
		const Index &from
	) const{
		return basisStateSet.get(to).getOverlap(
			basisStateSet.get(from)
		);
	}
private:
	BasisStateSet &basisStateSet;
};

TEST(SingleParticleContext, generateOveralpAmplitudeSet){
	SingleParticleContext singleParticleContext;
	BasisStateSet &basisStateSet
		= singleParticleContext.getBasisStateSet();

	BasicState state0({1, 2});
	BasicState state1({2, 3});

	state0.addOverlap(1, {1, 2});
	state0.addOverlap(2, {2, 3});

	state1.addOverlap(2, {1, 2});
	state1.addOverlap(3, {2, 3});

	basisStateSet.add(state0);
	basisStateSet.add(state1);

	OverlapAmplitudeCallback overlapAmplitudeCallback(basisStateSet);
	singleParticleContext.generateOverlapAmplitudeSet(
		overlapAmplitudeCallback
	);

	OverlapAmplitudeSet &overlapAmplitudeSet
		= singleParticleContext.getOverlapAmplitudeSet();
	OverlapAmplitudeSet::ConstIterator iterator
		= overlapAmplitudeSet.cbegin();

	EXPECT_TRUE(iterator != overlapAmplitudeSet.cend());
	EXPECT_TRUE((*iterator).getBraIndex().equals({1, 2}));
	EXPECT_TRUE((*iterator).getKetIndex().equals({1, 2}));
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag((*iterator).getAmplitude()), 0);

	++iterator;
	EXPECT_TRUE(iterator != overlapAmplitudeSet.cend());
	EXPECT_TRUE((*iterator).getBraIndex().equals({1, 2}));
	EXPECT_TRUE((*iterator).getKetIndex().equals({2, 3}));
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_DOUBLE_EQ(imag((*iterator).getAmplitude()), 0);

	++iterator;
	EXPECT_TRUE(iterator != overlapAmplitudeSet.cend());
	EXPECT_TRUE((*iterator).getBraIndex().equals({2, 3}));
	EXPECT_TRUE((*iterator).getKetIndex().equals({1, 2}));
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_DOUBLE_EQ(imag((*iterator).getAmplitude()), 0);

	++iterator;
	EXPECT_TRUE(iterator != overlapAmplitudeSet.cend());
	EXPECT_TRUE((*iterator).getBraIndex().equals({2, 3}));
	EXPECT_TRUE((*iterator).getKetIndex().equals({2, 3}));
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 3);
	EXPECT_DOUBLE_EQ(imag((*iterator).getAmplitude()), 0);

	++iterator;
	EXPECT_TRUE(iterator == overlapAmplitudeSet.cend());
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, constructCOO){
}

TEST(SingleParticleContext, getBasisStateSet){
	SingleParticleContext singleParticleContext0;
	BasisStateSet &basisStateSet0
		= singleParticleContext0.getBasisStateSet();
	//Dummy expression to supress warning about unused variable.
	if(sizeof(basisStateSet0) == 0);

	const SingleParticleContext singleParticleContext1;
	const BasisStateSet &basisStateSet1
		= singleParticleContext1.getBasisStateSet();
	//Dummy expression to supress warning about unused variable.
	if(sizeof(basisStateSet1) == 0);
}

TEST(SingleParticleContext, getHoppingAmplitudeSet){
	SingleParticleContext singleParticleContext0;
	HoppingAmplitudeSet &hoppingAmplitudeSet0
		= singleParticleContext0.getHoppingAmplitudeSet();
	//Dummy call to supress warning about unused variable.
	hoppingAmplitudeSet0.getBasisSize();

	const SingleParticleContext singleParticleContext1;
	const HoppingAmplitudeSet &hoppingAmplitudeSet1
		= singleParticleContext1.getHoppingAmplitudeSet();
	//Dummy call to supress warning about unused variable.
	hoppingAmplitudeSet1.getBasisSize();
}

TEST(SingleParticleContext, getSourceAmplitudeSet){
	SingleParticleContext singleParticleContext0;
	SourceAmplitudeSet &sourceAmplitudeSet0
		= singleParticleContext0.getSourceAmplitudeSet();
	//Dummy call to supress warning about unused variable.
	sourceAmplitudeSet0.getSizeInBytes();

	const SingleParticleContext singleParticleContext1;
	const SourceAmplitudeSet &sourceAmplitudeSet1
		= singleParticleContext1.getSourceAmplitudeSet();
	//Dummy call to supress warning about unused variable.
	sourceAmplitudeSet1.getSizeInBytes();
}

TEST(SingleParticleContext, getOverlapAmplitudeSet){
	SingleParticleContext singleParticleContext0;
	OverlapAmplitudeSet &overlapAmplitudeSet0
		= singleParticleContext0.getOverlapAmplitudeSet();
	//Dummy call to supress warning about unused variable.
	overlapAmplitudeSet0.getSizeInBytes();

	const SingleParticleContext singleParticleContext1;
	const OverlapAmplitudeSet &overlapAmplitudeSet1
		= singleParticleContext1.getOverlapAmplitudeSet();
	//Dummy call to supress warning about unused variable.
	overlapAmplitudeSet1.getSizeInBytes();
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//Geometry.
TEST(SingleParticleContext, createGeometry){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//Geometry.
TEST(SingleParticleContext, getGeometry){
}

TEST(SingleParticleContext, serialize){
	//Already tested through serializeToJSON
}

};
