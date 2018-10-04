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
	model0.setChemicalPotential(0);
	model0.setTemperature(100);
	model0.setStatistics(Statistics::FermiDirac);

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

	//Check additional Model parameters.
	EXPECT_EQ(model2.getChemicalPotential(), 0);
	EXPECT_EQ(model2.getTemperature(), 100);
	EXPECT_EQ(model2.getStatistics(), Statistics::FermiDirac);

	////////////////////////////////////////////
	// Test with non-zero chemical potential. //
	////////////////////////////////////////////

	//Normal model.
	Model model3;
	model3.setVerbose(false);
	model3 << HoppingAmplitude(1, {0}, {1}) + HC;
	model3.construct();
	model3.setChemicalPotential(2);

	//Reference model.
	Model model4;
	model4.setVerbose(false);
	model4 << HoppingAmplitude(1, {0, 0}, {1, 0}) + HC;
	model4 << HoppingAmplitude(-2, {0, 0}, {0, 0});
	model4 << HoppingAmplitude(-2, {1, 0}, {1, 0});

	model4 << HoppingAmplitude(-1, {1, 1}, {0, 1}) + HC;
	model4 << HoppingAmplitude(2, {0, 1}, {0, 1});
	model4 << HoppingAmplitude(2, {1, 1}, {1, 1});
	model4.construct();

	Model model5 = NambuSpaceExtender::extend(model3);
	model5.setVerbose(false);
	model5.construct();

	//Compare the model created with the NambuSpaceExtender with the
	//reference model.
	EXPECT_EQ(model4.getBasisSize(), model5.getBasisSize());
	HoppingAmplitudeSet::ConstIterator iterator4
		= model4.getHoppingAmplitudeSet().cbegin();
	HoppingAmplitudeSet::ConstIterator iterator5
		= model5.getHoppingAmplitudeSet().cbegin();
	while(iterator4 != model4.getHoppingAmplitudeSet().cend()){
		//These tests are vulnerable since they rely on the order in
		//which HoppinHamplitudes are returned for HoppingAmplitudes
		//with the same from-Index. This test may therefore break
		//without signaling a real problem. This test should optimally
		//be rewritten to remove dependence on specific knowledge of
		//how the HoppingAmplitudeTree::Iterator iterates through
		//HoppingAmplitudes with the same from-Index. In particular,
		//this test will break if the diagonal entries are added to
		//model4 before the off-diagonal, or if NambuSpaceExtender is
		//modified to add the diagonal terms before the off-diagonal,
		//without there actually being any problem.
		EXPECT_EQ(
			abs(
				(*iterator4).getAmplitude()
				- (*iterator5).getAmplitude()
			),
			0.
		);
		EXPECT_TRUE(
			(*iterator4).getToIndex().equals(
				(*iterator5).getToIndex()
			)
		);

		++iterator4;
		++iterator5;
	}
	EXPECT_TRUE(iterator5 == model5.getHoppingAmplitudeSet().cend());
}

};
