#include "TBTK/Property/Magnetization.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Property{

TEST(Magnetization, Constructor0){
	Magnetization magnetization({2, 3, 4});
	ASSERT_EQ(magnetization.getDimensions(), 3);
	EXPECT_EQ(magnetization.getRanges()[0], 2);
	EXPECT_EQ(magnetization.getRanges()[1], 3);
	EXPECT_EQ(magnetization.getRanges()[2], 4);
	ASSERT_EQ(magnetization.getSize(), 2*3*4);
	const std::vector<SpinMatrix> &data = magnetization.getData();
	for(unsigned int n = 0; n < data.size(); n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				EXPECT_DOUBLE_EQ(real(data[n].at(r, c)), 0);
				EXPECT_DOUBLE_EQ(imag(data[n].at(r, c)), 0);
			}
		}
	}
}

TEST(Magnetization, Constructor1){
	CArray<SpinMatrix> dataInput(2*3*4);
	for(unsigned int n = 0; n < 2*3*4; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				dataInput[n].at(r, c) = 4*n + 2*r + c;
			}
		}
	}
	Magnetization magnetization({2, 3, 4}, dataInput);
	ASSERT_EQ(magnetization.getDimensions(), 3);
	EXPECT_EQ(magnetization.getRanges()[0], 2);
	EXPECT_EQ(magnetization.getRanges()[1], 3);
	EXPECT_EQ(magnetization.getRanges()[2], 4);
	ASSERT_EQ(magnetization.getSize(), 2*3*4);
	const std::vector<SpinMatrix> &data = magnetization.getData();
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

TEST(Magnetization, Constructor2){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	Magnetization magnetization(indexTree);
	ASSERT_EQ(magnetization.getSize(), 3);
	for(unsigned int n = 0; n < magnetization.getSize(); n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				EXPECT_DOUBLE_EQ(
					real(magnetization({n}).at(r, c)),
					0
				);
				EXPECT_DOUBLE_EQ(
					imag(magnetization({n}).at(r, c)),
					0
				);
			}
		}
	}
}

TEST(Magnetization, Constructor3){
	IndexTree indexTree;
	indexTree.add({0});
	indexTree.add({1});
	indexTree.add({2});
	indexTree.generateLinearMap();
	CArray<SpinMatrix> dataInput(3);
	for(unsigned int n = 0; n < 3; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				dataInput[n].at(r, c) = 4*n + 2*r + c;
			}
		}
	}
	Magnetization magnetization(indexTree, dataInput);
	ASSERT_EQ(magnetization.getSize(), 3);
	for(unsigned int n = 0; n < magnetization.getSize(); n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				EXPECT_DOUBLE_EQ(
					real(magnetization({n}).at(r, c)),
					4*n + 2*r + c
				);
				EXPECT_DOUBLE_EQ(
					imag(magnetization({n}).at(r, c)),
					0
				);
			}
		}
	}
}

//TODO
//The test below should work once AbstractProperty<SpinMatrix, false, false>
//has been implemented.
TEST(LDOS, SerializeToJSON){
	//IndexDescriptor::Format::Ranges.
/*	int ranges[3] = {2, 3, 4};
	SpinMatrix dataInput0[2*3*4];
	for(unsigned int n = 0; n < 2*3*4; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				dataInput0[n].at(r, c) = 4*n + 2*r + c;
			}
		}
	}
	Magnetization magnetization0(3, ranges, dataInput0);
	Magnetization magnetization1(
		magnetization0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	ASSERT_EQ(magnetization1.getDimensions(), 3);
	EXPECT_EQ(magnetization1.getRanges()[0], 2);
	EXPECT_EQ(magnetization1.getRanges()[1], 3);
	EXPECT_EQ(magnetization1.getRanges()[2], 4);
	ASSERT_EQ(magnetization1.getSize(), 2*3*4);
	const SpinMatrix *data1 = magnetization1.getData();
	for(unsigned int n = 0; n < 2*3*4; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				EXPECT_DOUBLE_EQ(
					real(data1[n].at(r, c)),
					4*n + 2*r + c
				);
				EXPECT_DOUBLE_EQ(
					imag(data1[n].at(r, c)),
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
	SpinMatrix dataInput2[3];
	for(unsigned int n = 0; n < 3; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				dataInput2[n].at(r, c) = 4*n + 2*r + c;
			}
		}
	}
	Magnetization magnetization2(indexTree, dataInput2);
	Magnetization magnetization3(
		magnetization2.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);
	ASSERT_EQ(magnetization3.getSize(), 3);
	for(unsigned int n = 0; n < 3; n++){
		for(unsigned int r = 0; r < 2; r++){
			for(unsigned int c = 0; c < 2; c++){
				EXPECT_DOUBLE_EQ(
					real(magnetization3({n}).at(r, c)),
					4*n + 2*r + c
				);
				EXPECT_DOUBLE_EQ(
					imag(magnetization3({n}).at(r, c)),
					0
				);
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
