#include "TBTK/OverlapAmplitude.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(OverlapAmplitude, ConstructorAmplitude){
	std::string errorMessage = "Constructor failed.";

	OverlapAmplitude overlapAmplitude(
		std::complex<double>(1, 2),
		{1, 2, 3},
		{4, 5}
	);
	EXPECT_EQ(
		overlapAmplitude.getAmplitude(),
		std::complex<double>(1, 2)
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude.getBraIndex().equals({1, 2, 3})
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude.getKetIndex().equals({4, 5})
	) << errorMessage;
}

class AmplitudeCallback : public OverlapAmplitude::AmplitudeCallback{
	std::complex<double> getOverlapAmplitude(
		const Index &bra,
		const Index &ket
	) const{
		if(bra.equals({1, 2}))
			return std::complex<double>(3, 4);
		else if(ket.equals({1, 2}))
			return std::complex<double>(3, -4);
		else
			return -1;
	}
} amplitudeCallback;

TEST(OverlapAmplitude, ConstructorAmplitudeCallback){
	std::string errorMessage = "Callback constructor failed.";

	OverlapAmplitude overlapAmplitude(
		amplitudeCallback,
		{1, 2},
		{3, 4, 5}
	);
	EXPECT_EQ(
		overlapAmplitude.getAmplitude(),
		std::complex<double>(3, 4)
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude.getBraIndex().equals({1, 2})
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude.getKetIndex().equals({3, 4, 5})
	) << errorMessage;
}

TEST(OverlapAmplitude, CopyConstructor){
	std::string errorMessage = "Copy constructor failed.";

	OverlapAmplitude overlapAmplitude0(
		std::complex<double>(1, 2),
		{1, 2, 3},
		{4, 5}
	);
	OverlapAmplitude overlapAmplitude1(
		amplitudeCallback,
		{1, 2},
		{3, 4, 5}
	);

	OverlapAmplitude overlapAmplitude2 = overlapAmplitude0;
	EXPECT_EQ(
		overlapAmplitude2.getAmplitude(),
		std::complex<double>(1, 2)
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude2.getBraIndex().equals({1, 2, 3})
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude2.getKetIndex().equals({4, 5})
	) << errorMessage;

	OverlapAmplitude overlapAmplitude3 = overlapAmplitude1;
	EXPECT_EQ(
		overlapAmplitude3.getAmplitude(),
		std::complex<double>(3, 4)
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude3.getBraIndex().equals({1, 2})
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude3.getKetIndex().equals({3, 4, 5})
	) << errorMessage;
}

TEST(OverlapAmplitude, SerializeToJSON){
	std::string errorMessage = "JSON serialization failed.";

	OverlapAmplitude overlapAmplitude0(
		std::complex<double>(1, 2),
		{1, 2, 3},
		{4, 5}
	);
	OverlapAmplitude overlapAmplitude1(
		overlapAmplitude0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_EQ(
		overlapAmplitude0.getAmplitude(),
		overlapAmplitude1.getAmplitude()
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude0.getBraIndex().equals(
			overlapAmplitude1.getBraIndex()
		)
	) << errorMessage;
	EXPECT_TRUE(
		overlapAmplitude0.getKetIndex().equals(
			overlapAmplitude1.getKetIndex()
		)
	) << errorMessage;
}

TEST(OverlapAmplitude, getAmplitude){
	//Extensively tested through other tests.
}

TEST(OverlapAmplitude, getBraIndex){
	//Extensively tested through other tests.
}

TEST(OverlapAmplitude, getFromIndex){
	//Extensively tested through other tests.
}

TEST(OverlapAmplitude, getIsCallbackDependent){
	OverlapAmplitude overlapAmplitude0(1, {0}, {0});
	EXPECT_FALSE(overlapAmplitude0.getIsCallbackDependent());

	OverlapAmplitude overlapAmplitude1(amplitudeCallback, {0}, {0});
	EXPECT_TRUE(overlapAmplitude1.getIsCallbackDependent());
}

TEST(OverlapAmplitude, getAmplitudeCallback){
	OverlapAmplitude overlapAmplitude0(1, {0}, {0});
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			overlapAmplitude0.getAmplitudeCallback();
		},
		::testing::ExitedWithCode(1),
		""
	);

	OverlapAmplitude overlapAmplitude1(amplitudeCallback, {0}, {0});
	EXPECT_EQ(&overlapAmplitude1.getAmplitudeCallback(), &amplitudeCallback);
}

TEST(OverlapAmplitude, toString){
	//Not tested due to insuficient control of number formating.
}

TEST(OverlapAmplitude, getSizeInBytes){
	EXPECT_TRUE(
		OverlapAmplitude(
			std::complex<double>(1, 2), {1, 2, 3}, {4, 5}
		).getSizeInBytes() > 0
	) << "getSizeInBytes() failed.";
}

};
