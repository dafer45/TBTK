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
