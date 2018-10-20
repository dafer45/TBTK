#include "TBTK/Property/SpinPolarizedLDOS.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(SpinPolarizedLDOS, Constructor0){
	SpinPolarizedLDOS spinPolarizedLDOS({2, 3, 4}, -10, 10, 1000);
	ASSERT_EQ(spinPolarizedLDOS.getDimensions(), 3);
	EXPECT_EQ(spinPolarizedLDOS.getRanges()[0], 2);
	EXPECT_EQ(spinPolarizedLDOS.getRanges()[1], 3);
	EXPECT_EQ(spinPolarizedLDOS.getRanges()[2], 4);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS.getUpperBound(), 10);
	ASSERT_EQ(spinPolarizedLDOS.getResolution(), 1000);
	ASSERT_EQ(spinPolarizedLDOS.getSize(), 2*3*4*1000);
	const std::vector<SpinMatrix> &data = spinPolarizedLDOS.getData();
	for(unsigned int n = 0; n < data.size(); n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				EXPECT_DOUBLE_EQ(real(data[n].at(r, c)), 0);
				EXPECT_DOUBLE_EQ(imag(data[n].at(r, c)), 0);
			}
		}
	}
}

TEST(SpinPolarizedLDOS, Constructor1){
	SpinMatrix *dataInput = new SpinMatrix[2*3*4*1000];
	for(unsigned int n = 0; n < 2*3*4*1000; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				dataInput[n].at(r, c) = 4*n + 2*r + c;
			}
		}
	}
	SpinPolarizedLDOS spinPolarizedLDOS(
		{2, 3, 4},
		-10,
		10,
		1000,
		dataInput
	);
	delete [] dataInput;
	ASSERT_EQ(spinPolarizedLDOS.getDimensions(), 3);
	EXPECT_EQ(spinPolarizedLDOS.getRanges()[0], 2);
	EXPECT_EQ(spinPolarizedLDOS.getRanges()[1], 3);
	EXPECT_EQ(spinPolarizedLDOS.getRanges()[2], 4);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS.getUpperBound(), 10);
	ASSERT_EQ(spinPolarizedLDOS.getResolution(), 1000);
	ASSERT_EQ(spinPolarizedLDOS.getSize(), 2*3*4*1000);
	const std::vector<SpinMatrix> &data = spinPolarizedLDOS.getData();
	for(unsigned int n = 0; n < data.size(); n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				EXPECT_DOUBLE_EQ(
					real(data[n].at(r, c)),
					4*n + 2*r + c
				);
				EXPECT_DOUBLE_EQ(
					imag(data[n].at(r, c)),
					0
				);
			}
		}
	}
}

TEST(SpinPolarizedLDOS, Constructor2){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	SpinPolarizedLDOS spinPolarizedLDOS(indexTree, -10, 10, 1000);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS.getUpperBound(), 10);
	ASSERT_EQ(spinPolarizedLDOS.getResolution(), 1000);
	ASSERT_EQ(spinPolarizedLDOS.getSize(), 3*1000);
	for(int n = 0; n < 3; n++){
		for(unsigned int m = 0; m < spinPolarizedLDOS.getResolution(); m++){
			for(unsigned int r = 0; r < 2; r++){
				for(unsigned int c = 0; c < 2; c++){
					EXPECT_DOUBLE_EQ(
						real(
							spinPolarizedLDOS(
								{n},
								m
							).at(r, c)
						),
						0
					);
					EXPECT_DOUBLE_EQ(
						imag(
							spinPolarizedLDOS(
								{n},
								m
							).at(r, c)
						),
						0
					);
				}
			}
		}
	}
}

TEST(SpinPolarizedLDOS, Constructor3){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	SpinMatrix dataInput[3*1000];
	for(unsigned int n = 0; n < 3; n++){
		for(unsigned int m = 0; m < 1000; m++){
			for(unsigned int r = 0; r < 2; r++){
				for(unsigned int c = 0; c < 2; c++){
					dataInput[1000*n + m].at(r, c)
						= 4*1000*n + 4*m + 2*r + c;
				}
			}
		}
	}
	SpinPolarizedLDOS spinPolarizedLDOS(
		indexTree,
		-10,
		10,
		1000,
		dataInput
	);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS.getUpperBound(), 10);
	ASSERT_EQ(spinPolarizedLDOS.getResolution(), 1000);
	ASSERT_EQ(spinPolarizedLDOS.getSize(), 3*1000);
	for(int n = 0; n < 3; n++){
		for(
			unsigned int m = 0;
			m < spinPolarizedLDOS.getResolution();
			m++
		){
			for(unsigned int r = 0; r < 2; r++){
				for(unsigned int c = 0; c < 2; c++){
					EXPECT_DOUBLE_EQ(
						real(
							spinPolarizedLDOS(
								{n},
								m
							).at(r, c)
						),
						4*1000*n + 4*m + 2*r + c
					);
					EXPECT_DOUBLE_EQ(
						imag(
							spinPolarizedLDOS(
								{n},
								m
							).at(r, c)
						),
						0
					);
				}
			}
		}
	}
}

//TODO
//The test below should work once AbstractProperty<SpinMatrix, false, false>
//has been implemented.
TEST(SpinPolarizedLDOS, SerializeToJSON){
	//IndexDescriptor::Format::Ranges.
/*	int ranges[3] = {2, 3, 4};
	SpinMatrix *dataInput0 = new SpinMatrix[2*3*4*1000];
	for(unsigned int n = 0; n < 2*3*4*1000; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				dataInput0[n].at(r, c) = 4*n + 2*r + c;
			}
		}
	}
	SpinPolarizedLDOS spinPolarizedLDOS0(
		3,
		ranges,
		-10,
		10,
		1000,
		dataInput0
	);
	delete [] dataInput0;
	SpinPolarizedLDOS spinPolarizedLDOS1(
		spinPolarizedLDOS0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	ASSERT_EQ(spinPolarizedLDOS1.getDimensions(), 3);
	EXPECT_EQ(spinPolarizedLDOS1.getRanges()[0], 2);
	EXPECT_EQ(spinPolarizedLDOS1.getRanges()[1], 3);
	EXPECT_EQ(spinPolarizedLDOS1.getRanges()[2], 4);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS1.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS1.getUpperBound(), 10);
	ASSERT_EQ(spinPolarizedLDOS1.getResolution(), 1000);
	ASSERT_EQ(spinPolarizedLDOS1.getSize(), 2*3*4*1000);
	const SpinMatrix *data = spinPolarizedLDOS1.getData();
	for(unsigned int n = 0; n < 2*3*4*1000; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				EXPECT_DOUBLE_EQ(
					real(data[n].at(r, c)),
					4*n + 2*r + c
				);
				EXPECT_DOUBLE_EQ(
					imag(data[n].at(r, c)),
					0
				);
			}
		}
	}

	//IndexDescriptor::Format::Custom.
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	SpinMatrix dataInput1[3*1000];
	for(unsigned int n = 0; n < 3; n++){
		for(unsigned int m = 0; m < 1000; m++){
			for(unsigned int r = 0; r < 2; r++){
				for(unsigned int c = 0; c < 2; c++){
					dataInput1[1000*n + m].at(r, c)
						= 4*1000*n + 4*m + 2*r + c;
				}
			}
		}
	}
	SpinPolarizedLDOS spinPolarizedLDOS2(
		indexTree,
		-10,
		10,
		1000,
		dataInput1
	);
	SpinPolarizedLDOS spinPolarizedLDOS3(
		spinPolarizedLDOS2.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS3.getLowerBound(), -10);
	EXPECT_DOUBLE_EQ(spinPolarizedLDOS3.getUpperBound(), 10);
	ASSERT_EQ(spinPolarizedLDOS3.getResolution(), 1000);
	ASSERT_EQ(spinPolarizedLDOS3.getSize(), 3*1000);
	for(int n = 0; n < 3; n++){
		for(unsigned int m = 0; m < 1000; m++){
			for(unsigned int r = 0; r < 2; r++){
				for(unsigned int c = 0; c < 2; c++){
					EXPECT_DOUBLE_EQ(
						real(
							spinPolarizedLDOS3(
								{n},
								m
							).at(r, c)
						),
						4*1000*n + 4*m + 2*r + c
					);
					EXPECT_DOUBLE_EQ(
						imag(
							spinPolarizedLDOS3(
								{n},
								m
							).at(r, c)
						),
						0
					);
				}
			}
		}
	}*/
}

TEST(LDOS, serialize){
	//Already tested through
	//Magnetization::SerializeToJSON
}

};	//End of namespace Property
};	//End of namespace TBTK
