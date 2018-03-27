#include "TBTK/SourceAmplitude.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(SourceAmplitude, ConstructorAmplitude){
	SourceAmplitude sourceAmplitude(std::complex<double>(1, 2), {1, 2, 3});
	EXPECT_EQ(sourceAmplitude.getAmplitude(), std::complex<double>(1, 2));
	EXPECT_TRUE(sourceAmplitude.getIndex().equals({1, 2, 3}));
}

std::complex<double> amplitudeCallback(const Index &index){
	if(index.equals({1, 2, 3}))
		return std::complex<double>(3, 4);
	else
		return -1;
}

TEST(SourceAmplitude, ConstructorAmplitudeCallback){
	SourceAmplitude sourceAmplitude(amplitudeCallback, {1, 2, 3});
	EXPECT_EQ(sourceAmplitude.getAmplitude(), std::complex<double>(3, 4));
	EXPECT_TRUE(sourceAmplitude.getIndex().equals({1, 2, 3}));
}

TEST(SourceAmplitude, SerializeToJSON){
	SourceAmplitude sourceAmplitude0(std::complex<double>(1, 2), {1, 2, 3});
	SourceAmplitude sourceAmplitude1(
		sourceAmplitude0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(sourceAmplitude0.getAmplitude(), sourceAmplitude1.getAmplitude());
	EXPECT_TRUE(sourceAmplitude0.getIndex().equals(sourceAmplitude1.getIndex()));
}

TEST(SourceAmplitude, getAmplitude){
	//Extensively tested through other tests.
}

TEST(SourceAmplitude, getIndex){
	//Extensively tested through other tests.
}

TEST(SourceAmplitude, toString){
	//Not tested due to insuficient control of number formating.
}

TEST(SourceAmplitude, getSizeInBytes){
	EXPECT_TRUE(SourceAmplitude(std::complex<double>(1, 2), {1, 2, 3}).getSizeInBytes() > 0);
}

};
