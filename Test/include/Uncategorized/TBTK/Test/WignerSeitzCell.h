#include "TBTK/WignerSeitzCell.h"

#include "gtest/gtest.h"

namespace TBTK{

class WignerSeitzCellTest : public ::testing::Test{
protected:
	WignerSeitzCell wignerSeitzCell[6];

	//Small parameter used to set the displacement away from the corners.
	double d = 1e-5;

	//The 2D Wigner-Seitz cell used in the test is a hexagon with the
	//following corners.
	std::vector<std::vector<double>> corners2D;
	//Displacements from the corners that take the points onto the first,
	//second, second, and third Wigner-Seitz cell, respectively. These
	//points are used to check that the division into Wigner-Seitz cell is
	//properly done around the corners.
	std::vector<std::vector<std::vector<double>>> displacements2D;
	std::vector<Index> expectedWignerSeitzCells2D;

	//The 3D Wigner-Seitz cell used in the test is a hexagon with the
	//following corners.
	std::vector<std::vector<double>> corners3D;
	//Displacements from the corners that take the points onto the first,
	//second, second, and third Wigner-Seitz cell, respectively. These
	//points are used to check that the division into Wigner-Seitz cell is
	//properly done around the corners.
	std::vector<std::vector<std::vector<double>>> displacements3D;
	std::vector<Index> expectedWignerSeitzCells3D;

	void SetUp() override{
		//Set up 1D, 2D, and 3D Wigner-Seitz cell with Nodal and
		//Interior mesh types.
		wignerSeitzCell[0] = WignerSeitzCell(
			{{2}},
			SpacePartition::MeshType::Nodal
		);
		wignerSeitzCell[1] = WignerSeitzCell(
			{{2}},
			SpacePartition::MeshType::Interior
		);
		wignerSeitzCell[2] = WignerSeitzCell(
			{{2, 0}, {-1, sqrt(3)}},
			SpacePartition::MeshType::Nodal
		);
		wignerSeitzCell[3] = WignerSeitzCell(
			{{2, 0}, {-1, sqrt(3)}},
			SpacePartition::MeshType::Interior
		);
		wignerSeitzCell[4] = WignerSeitzCell(
			{
				{2, 0, 0},
				{-1, sqrt(3), 0},
				{0, 0, 2}
			},
			SpacePartition::MeshType::Nodal
		);
		wignerSeitzCell[5] = WignerSeitzCell(
			{
				{2, 0, 0},
				{-1, sqrt(3), 0},
				{0, 0, 2}
			},
			SpacePartition::MeshType::Interior
		);

		//Points that lie at the intersection between different
		//Wigner-Seitz cell for the 2D lattice.
		corners2D = {
			{1, 1/sqrt(3)},
			{0, 2/sqrt(3)},
			{-1, 1/sqrt(3)},
			{-1, -1/sqrt(3)},
			{0, -2/sqrt(3)},
			{1, -1/sqrt(3)}
		};
		//Displacements away from the points above and into the
		//interior of the 0th, 1st, 1st, and 2nd Wigner-Seitz cell,
		//respectively. Each group of four corresponds to one point in
		//corners2D. The last four groups are reflections of the first
		//two groups throught the x- and y-axes.
		displacements2D = {
			{
				{-d, 0},
				{d, -2*d},
				{-d, 2*d},
				{d, d}
			},
			{
				{0, -d},
				{d, 0},
				{-d, 0},
				{0, d}
			},
			{
				{d, 0},
				{-d, -2*d},
				{d, 2*d},
				{-d, d}
			},
			{
				{d, 0},
				{-d, 2*d},
				{d, -2*d},
				{-d, -d}
			},
			{
				{0, d},
				{d, 0},
				{-d, 0},
				{0, -d}
			},
			{
				{-d, 0},
				{d, 2*d},
				{-d, -2*d},
				{d, -d}
			}
		};
		expectedWignerSeitzCells2D = {{0}, {1}, {1}, {2}};

		//Points that lie at the intersection between different
		//Wigner-Seitz cell for the 3D lattice.
		corners3D = {
			{1, 1/sqrt(3), 1},
			{0, 2/sqrt(3), 1},
			{-1, 1/sqrt(3), 1},
			{-1, -1/sqrt(3), 1},
			{0, -2/sqrt(3), 1},
			{1, -1/sqrt(3), 1},
			{1, 1/sqrt(3), -1},
			{0, 2/sqrt(3), -1},
			{-1, 1/sqrt(3), -1},
			{-1, -1/sqrt(3), -1},
			{0, -2/sqrt(3), -1},
			{1, -1/sqrt(3), -1}
		};
		//Displacements away from the points above and into the
		//interior of the 0th, 1st, 2nd, 1st, 2nd, 3rd, 1st, 2nd, 3rd,
		//2nd, and 5th Wigner-Seitz cell, respectively. Each group of
		//eleven corresponds to one point in corners3D. The second to
		//sixth group is a rotation of the first group by pi/3, 2pi/3,
		//... 5pi/3. The last six are reflections of the first six
		//through the x/y-plane.
		displacements3D = {
			{
				{-d, 0, -d},
				{-d, 0, d/2},
				{-d, 0, d},
				{d, -2*d, -d},
				{d, -2*d, -d/2},
				{d, -2*d, d},
				{-d, 2*d, -2*d},
				{-d, 2*d, -d},
				{-d, 2*d, d},
				{d, d, -2*d},
				{d, d, d}
			},
			{
				{-d/2, -d*sqrt(3)/2, -d},
				{-d/2, -d*sqrt(3)/2, d/2},
				{-d/2, -d*sqrt(3)/2, d},
				{d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), -d},
				{d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), -d/2},
				{d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), d},
				{-d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), -2*d},
				{-d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), -d},
				{-d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), d},
				{d*(1 - sqrt(3))/2, d*(1 + sqrt(3))/2, -2*d},
				{d*(1 - sqrt(3))/2, d*(1 + sqrt(3))/2, d}
			},
			{
				{d, 0, -d},
				{d, 0, d/2},
				{d, 0, d},
				{-d, -2*d, -d},
				{-d, -2*d, -d/2},
				{-d, -2*d, d},
				{d, 2*d, -2*d},
				{d, 2*d, -d},
				{d, 2*d, d},
				{-d, d, -2*d},
				{-d, d, d}
			},
			{
				{d, 0, -d},
				{d, 0, d/2},
				{d, 0, d},
				{-d, 2*d, -d},
				{-d, 2*d, -d/2},
				{-d, 2*d, d},
				{d, -2*d, -2*d},
				{d, -2*d, -d},
				{d, -2*d, d},
				{-d, -d, -2*d},
				{-d, -d, d}
			},
			{
				{-d/2, d*sqrt(3)/2, -d},
				{-d/2, d*sqrt(3)/2, d/2},
				{-d/2, d*sqrt(3)/2, d},
				{d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), -d},
				{d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), -d/2},
				{d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), d},
				{-d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), -2*d},
				{-d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), -d},
				{-d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), d},
				{d*(1 - sqrt(3))/2, -d*(1 + sqrt(3))/2, -2*d},
				{d*(1 - sqrt(3))/2, -d*(1 + sqrt(3))/2, d}
			},
			{
				{-d, 0, -d},
				{-d, 0, d/2},
				{-d, 0, d},
				{d, 2*d, -d},
				{d, 2*d, -d/2},
				{d, 2*d, d},
				{-d, -2*d, -2*d},
				{-d, -2*d, -d},
				{-d, -2*d, d},
				{d, -d, -2*d},
				{d, -d, d}
			},
			{
				{-d, 0, d},
				{-d, 0, -d/2},
				{-d, 0, -d},
				{d, -2*d, d},
				{d, -2*d, d/2},
				{d, -2*d, -d},
				{-d, 2*d, 2*d},
				{-d, 2*d, d},
				{-d, 2*d, -d},
				{d, d, 2*d},
				{d, d, -d}
			},
			{
				{-d/2, -d*sqrt(3)/2, d},
				{-d/2, -d*sqrt(3)/2, -d/2},
				{-d/2, -d*sqrt(3)/2, -d},
				{d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), d},
				{d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), d/2},
				{d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), -d},
				{-d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), 2*d},
				{-d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), d},
				{-d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), -d},
				{d*(1 - sqrt(3))/2, d*(1 + sqrt(3))/2, 2*d},
				{d*(1 - sqrt(3))/2, d*(1 + sqrt(3))/2, -d}
			},
			{
				{d, 0, d},
				{d, 0, -d/2},
				{d, 0, -d},
				{-d, -2*d, d},
				{-d, -2*d, d/2},
				{-d, -2*d, -d},
				{d, 2*d, 2*d},
				{d, 2*d, d},
				{d, 2*d, -d},
				{-d, d, 2*d},
				{-d, d, -d}
			},
			{
				{d, 0, d},
				{d, 0, -d/2},
				{d, 0, -d},
				{-d, 2*d, d},
				{-d, 2*d, d/2},
				{-d, 2*d, -d},
				{d, -2*d, 2*d},
				{d, -2*d, d},
				{d, -2*d, -d},
				{-d, -d, 2*d},
				{-d, -d, -d}
			},
			{
				{-d/2, d*sqrt(3)/2, d},
				{-d/2, d*sqrt(3)/2, -d/2},
				{-d/2, d*sqrt(3)/2, -d},
				{d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), d},
				{d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), d/2},
				{d*(1/2. + sqrt(3)), -d*(sqrt(3)/2 - 1), -d},
				{-d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), 2*d},
				{-d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), d},
				{-d*(1/2. + sqrt(3)), d*(sqrt(3)/2 - 1), -d},
				{d*(1 - sqrt(3))/2, -d*(1 + sqrt(3))/2, 2*d},
				{d*(1 - sqrt(3))/2, -d*(1 + sqrt(3))/2, -d}
			},
			{
				{-d, 0, d},
				{-d, 0, -d/2},
				{-d, 0, -d},
				{d, 2*d, d},
				{d, 2*d, d/2},
				{d, 2*d, -d},
				{-d, -2*d, 2*d},
				{-d, -2*d, d},
				{-d, -2*d, -d},
				{d, -d, 2*d},
				{d, -d, -d}
			}
		};
		expectedWignerSeitzCells3D = {
			{0}, {1}, {2}, {1}, {2}, {3}, {1}, {2}, {3}, {2}, {5}
		};
	}
};

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.1 2019-11-21
TEST_F(WignerSeitzCellTest, getMajorCellIndex1){
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({0.5}).equals({0}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({-0.5}).equals({0}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({1}).equals({0}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({-1}).equals({0}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({1.01}).equals({1}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({-1.01}).equals({1}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({2}).equals({1}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({-2}).equals({1}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({2.01}).equals({2}));
	EXPECT_TRUE(wignerSeitzCell[0].getMajorCellIndex({-2.01}).equals({2}));
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.2 2019-11-21
TEST_F(WignerSeitzCellTest, getMajorCellIndex2){
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({0.5}).equals({0}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({-0.5}).equals({0}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({1}).equals({0}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({-1}).equals({0}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({1.01}).equals({1}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({-1.01}).equals({1}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({2}).equals({1}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({-2}).equals({1}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({2.01}).equals({2}));
	EXPECT_TRUE(wignerSeitzCell[1].getMajorCellIndex({-2.01}).equals({2}));
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.3 2019-11-21
TEST_F(WignerSeitzCellTest, getMajorCellIndex3){
	for(unsigned int n = 0; n < corners2D.size(); n++){
		for(unsigned int d = 0; d < 4; d++){
			std::vector<double> coordinate = {
				corners2D[n][0] + displacements2D[n][d][0],
				corners2D[n][1] + displacements2D[n][d][1]
			};
			EXPECT_TRUE(
				wignerSeitzCell[2].getMajorCellIndex(
					coordinate
				).equals(expectedWignerSeitzCells2D[d])
			);
		}
	}
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.4 2019-11-21
TEST_F(WignerSeitzCellTest, getMajorCellIndex4){
	for(unsigned int n = 0; n < corners2D.size(); n++){
		for(unsigned int d = 0; d < 4; d++){
			std::vector<double> coordinate = {
				corners2D[n][0] + displacements2D[n][d][0],
				corners2D[n][1] + displacements2D[n][d][1]
			};
			EXPECT_TRUE(
				wignerSeitzCell[3].getMajorCellIndex(
					coordinate
				).equals(expectedWignerSeitzCells2D[d])
			);
		}
	}
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.5 2019-11-21
TEST_F(WignerSeitzCellTest, getMajorCellIndex5){
	for(unsigned int n = 0; n < corners3D.size(); n++){
		for(unsigned int d = 0; d < 11; d++){
			std::vector<double> coordinate = {
				corners3D[n][0] + displacements3D[n][d][0],
				corners3D[n][1] + displacements3D[n][d][1],
				corners3D[n][2] + displacements3D[n][d][2]
			};
			EXPECT_TRUE(
				wignerSeitzCell[4].getMajorCellIndex(
					coordinate
				).equals(expectedWignerSeitzCells3D[d])
			);
		}
	}
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.6 2019-11-21
TEST_F(WignerSeitzCellTest, getMajorCellIndex6){
	for(unsigned int n = 0; n < corners3D.size(); n++){
		for(unsigned int d = 0; d < 11; d++){
			std::vector<double> coordinate = {
				corners3D[n][0] + displacements3D[n][d][0],
				corners3D[n][1] + displacements3D[n][d][1],
				corners3D[n][2] + displacements3D[n][d][2]
			};
			EXPECT_TRUE(
				wignerSeitzCell[5].getMajorCellIndex(
					coordinate
				).equals(expectedWignerSeitzCells3D[d])
			);
		}
	}
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMajorCellIndex.7 2019-11-21
TEST_F(WignerSeitzCellTest, getMajorCellIndex7){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			wignerSeitzCell[0].getMajorCellIndex({1, 2});
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMinorCellIndex.1 2019-11-21
TEST_F(WignerSeitzCellTest, getMinorCellIndex1){
	//TODO:
	//This test fails for negative r, check whether this is a bug or
	//expected behavior.
/*	for(int r = -2; r < 3; r++){
		double R = 2*r;
		std::vector<std::vector<double>> points = {
			{R - 1 + 2/3. - d},
			{R - 1 + 2/3. + d},
			{R - 1 + 2*2/3. - d},
			{R - 1 + 2*2/3. + d}
		};
		std::vector<Index> expectedMinorCells = {{2}, {0}, {0}, {1}};
		for(unsigned int n = 0; n < points.size(); n++){
			Streams::out << r << "\t" << n << "\t" << wignerSeitzCell[0].getMinorCellIndex(points[n], {3}) << "\t" << expectedMinorCells[n] << "\n";
			EXPECT_TRUE(
				wignerSeitzCell[0].getMinorCellIndex(
					points[n],
					{3}
				).equals(expectedMinorCells[n])
			);
		}
	}*/
}

//TBTKFeature Uncategorized.WignerSeitzCell.getMinorCellIndex.2 2019-11-21
TEST_F(WignerSeitzCellTest, getMinorCellIndex2){
	//TODO:
	//Extend this test to also check outside the first Wigner-Seitz cell.
	//I.e. add similar outer loop as in the test getMinorCellIndex1.
	std::vector<std::vector<double>> points = {
		{-1 + 1/3. - d},
		{-1 + 1/3. + d},
		{-1 + 3*1/3. - d},
		{-1 + 3*1/3. + d},
		{-1 + 5*1/3. - d},
		{-1 + 5*1/3. + d}
	};
	std::vector<Index> expectedMinorCells = {{1}, {2}, {2}, {0}, {0}, {1}};
	for(unsigned int n = 0; n < points.size(); n++){
		EXPECT_TRUE(
			wignerSeitzCell[1].getMinorCellIndex(
				points[n],
				{3}
			).equals(expectedMinorCells[n])
		);
	}
}

// TODO:
// - Tests for getMinorCellIndex for 2D and 3d.
// - Tests for getMajorMesh.
// - Tests for getMinorMesh.
// - Tests for getMinorMeshPoints.

};
