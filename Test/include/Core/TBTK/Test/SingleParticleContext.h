#include "TBTK/SingleParticleContext.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(SingleParticleContext, Constructor){
	//Not testable on its own.
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

	//Check that default initialization is Statistics::FermiDirac.
	EXPECT_EQ(singleParticleContext.getStatistics(), Statistics::FermiDirac);

	//Check that default initialization is Statistics::FermiDirac.
	singleParticleContext.setStatistics(Statistics::BoseEinstein);
	EXPECT_EQ(singleParticleContext.getStatistics(), Statistics::BoseEinstein);

	//Check that default initialization is Statistics::FermiDirac.
	singleParticleContext.setStatistics(Statistics::FermiDirac);
	EXPECT_EQ(singleParticleContext.getStatistics(), Statistics::FermiDirac);
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, addHoppingAmplitude){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, addHoppingAmplitudeAndHermitianConjugate){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, getBasisIndex){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, getBasisSize){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, construct){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, constructCOO){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, destructCOO){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, reconstructCOO){
}

//TODO
//This function should possibly be removed from the SingleParticleContext
//itself by makin the SingleParticleContext inherit from the
//HoppingAmplitudeSet.
TEST(SingleParticleContext, getHoppingAmplitudeSet){
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
