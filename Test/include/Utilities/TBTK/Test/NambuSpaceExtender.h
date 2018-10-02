#include "TBTK/NambuSpaceExtender.h"

#include "gtest/gtest.h"

namespace TBTK{

//Callback used to by the test for NambuSpaceExtender::extend().
std::complex<double> amplitudeCallback(
	const Index &toIndex,
	const Index &fromIndex
){
	if(toIndex[1] == 0){
		if(toIndex[0] == 4)
			return std::complex<double>(0, 4);
		else
			return std::complex<double>(0, -4);
	}
	else{
		if(toIndex[0] == 4)
			return -std::complex<double>(0, -4);
		else
			return -std::complex<double>(0, 4);
	}
}

TEST(NambuSpaceExtender, extend){
	//Normal model.
	Model model0;
	model0.setVerbose(false);
	model0 << HoppingAmplitude(1, {0}, {0});
	model0 << HoppingAmplitude(2, {1}, {1});
	model0 << HoppingAmplitude(
		std::complex<double>(0, 3),
		{2},
		{3}
	) + HC;
	model0 << HoppingAmplitude(amplitudeCallback, {4}, {5}) + HC;
	model0.construct();

	//Nambu space reference model.
	Model model1;
	model1.setVerbose(false);
	model1 << HoppingAmplitude(1, {0, 0}, {0, 0});
	model1 << HoppingAmplitude(2, {1, 0}, {1, 0});
	model1 << HoppingAmplitude(
		std::complex<double>(0, 3),
		{2, 0},
		{3, 0}
	) + HC;
	model1 << HoppingAmplitude(amplitudeCallback, {4, 0}, {5, 0}) + HC;
	model1 << HoppingAmplitude(-1, {0, 1}, {0, 1});
	model1 << HoppingAmplitude(-2, {1, 1}, {1, 1});
	model1 << HoppingAmplitude(
		-std::complex<double>(0, 3),
		{3, 1},
		{2, 1}
	) + HC;
	model1 << HoppingAmplitude(amplitudeCallback, {4, 1}, {5, 1}) + HC;
	model1.construct();

	//Nambu space model created using the NambuSpaceExtender.
	Model model2 = NambuSpaceExtender::extend(model0);
	model2.setVerbose(false);
	model2.construct();

	//Compare the model created with the NambuSpaceExtender with the
	//reference model.
	EXPECT_EQ(model1.getBasisSize(), model2.getBasisSize());
	HoppingAmplitudeSet::ConstIterator iterator1
		= model1.getHoppingAmplitudeSet().cbegin();
	HoppingAmplitudeSet::ConstIterator iterator2
		= model2.getHoppingAmplitudeSet().cbegin();
	while(iterator1 != model1.getHoppingAmplitudeSet().cend()){
		EXPECT_EQ(
			abs(
				(*iterator1).getAmplitude()
				- (*iterator2).getAmplitude()
			),
			0.
		);
		EXPECT_TRUE(
			(*iterator1).getToIndex().equals(
				(*iterator2).getToIndex()
			)
		);

		++iterator1;
		++iterator2;
	}
	EXPECT_TRUE(iterator2 == model2.getHoppingAmplitudeSet().cend());
}

};
