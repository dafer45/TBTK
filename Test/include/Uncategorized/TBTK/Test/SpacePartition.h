#include "TBTK/SpacePartition.h"

#include "gtest/gtest.h"

namespace TBTK{

class PublicSpacePartition : public SpacePartition{
public:
	PublicSpacePartition(){}

	PublicSpacePartition(
		const std::vector<std::vector<double>> &basisVectors,
		MeshType meshType
	) : SpacePartition(basisVectors, meshType){}

	virtual Index getMajorCellIndex(
		const std::vector<double> &coordinate
	) const{
		return Index();
	}

	virtual Index getMinorCellIndex(
		const std::vector<double> &coordinate,
		const std::vector<unsigned int> &numMeshPoints
	) const{
		return Index();
	}

	virtual std::vector<std::vector<double>> getMajorMesh(
		const std::vector<unsigned int> &numMeshPoints
	) const{
		return std::vector<std::vector<double>>();
	}

	virtual std::vector<std::vector<double>> getMinorMesh(
		const std::vector<unsigned int> &numMeshPoints
	) const{
		return std::vector<std::vector<double>>();
	}

	virtual std::vector<double> getMinorMeshPoint(
		const std::vector<unsigned int> &meshPoint,
		const std::vector<unsigned int> &numMeshPoints
	) const{
		return std::vector<double>();
	}

	const std::vector<Vector3d>& getBasisVectors() const{
		return SpacePartition::getBasisVectors();
	}

	MeshType getMeshType() const{
		return SpacePartition::getMeshType();
	}
};

class SpacePartitionTest : public ::testing::Test{
protected:
	const std::vector<std::vector<double>> basisVectors;
	PublicSpacePartition spacePartition[3];

	void SetUp() override{
		spacePartition[0] = PublicSpacePartition(
			{{2}},
			SpacePartition::MeshType::Nodal
		);
		spacePartition[1] = PublicSpacePartition(
			{
				{2, 0},
				{0, 2}
			},
			SpacePartition::MeshType::Interior
		);
		spacePartition[2] = PublicSpacePartition(
			{
				{2, 0, 0},
				{0, 2, 0},
				{0, 0, 2}
			},
			SpacePartition::MeshType::Nodal
		);
	}
};

//TBTKFeature Uncategorized.SpacePartition.construction.1 2019-11-20
TEST(SpacePartition, construction1){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PublicSpacePartition spacePartition(
				{{}},
				SpacePartition::MeshType::Nodal
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Uncategorized.SpacePartition.construction.2 2019-11-20
TEST(SpacePartition, construction2){
	EXPECT_EXIT(
		{
			Streams::setStdMuteErr();
			PublicSpacePartition spacePartition(
				{
					{1, 0, 0, 0},
					{0, 1, 0, 0},
					{0, 0, 1, 0},
					{0, 0, 0, 1}
				},
				SpacePartition::MeshType::Nodal
			);
		},
		::testing::ExitedWithCode(1),
		""
	);
}

//TBTKFeature Uncategorized.SpacePartition.getNumDimensions.1 2019-11-20
TEST_F(SpacePartitionTest, getNumDimensions1){
	EXPECT_EQ(spacePartition[0].getNumDimensions(), 1);
	EXPECT_EQ(spacePartition[1].getNumDimensions(), 2);
	EXPECT_EQ(spacePartition[2].getNumDimensions(), 3);
}

//TBTKFeature Uncategorized.SpacePartition.getBasisVectors.1 2019-11-20
TEST_F(SpacePartitionTest, getBasisVectors1){
	const std::vector<Vector3d> &basisVectors
		= spacePartition[0].getBasisVectors();

	EXPECT_EQ(basisVectors[0].x, 2);
	EXPECT_EQ(basisVectors[0].y, 0);
	EXPECT_EQ(basisVectors[0].z, 0);

	EXPECT_EQ(basisVectors[1].x, 0);
	EXPECT_EQ(basisVectors[1].y, 1);
	EXPECT_EQ(basisVectors[1].z, 0);

	EXPECT_EQ(basisVectors[2].x, 0);
	EXPECT_EQ(basisVectors[2].y, 0);
	EXPECT_EQ(basisVectors[2].z, 1);
}

//TBTKFeature Uncategorized.SpacePartition.getBasisVectors.2 2019-11-20
TEST_F(SpacePartitionTest, getBasisVectors2){
	const std::vector<Vector3d> &basisVectors
		= spacePartition[1].getBasisVectors();

	EXPECT_EQ(basisVectors[0].x, 2);
	EXPECT_EQ(basisVectors[0].y, 0);
	EXPECT_EQ(basisVectors[0].z, 0);

	EXPECT_EQ(basisVectors[1].x, 0);
	EXPECT_EQ(basisVectors[1].y, 2);
	EXPECT_EQ(basisVectors[1].z, 0);

	EXPECT_EQ(basisVectors[2].x, 0);
	EXPECT_EQ(basisVectors[2].y, 0);
	EXPECT_EQ(basisVectors[2].z, 1);
}

//TBTKFeature Uncategorized.SpacePartition.getBasisVectors.3 2019-11-20
TEST_F(SpacePartitionTest, getBasisVectors3){
	const std::vector<Vector3d> &basisVectors
		= spacePartition[2].getBasisVectors();

	EXPECT_EQ(basisVectors[0].x, 2);
	EXPECT_EQ(basisVectors[0].y, 0);
	EXPECT_EQ(basisVectors[0].z, 0);

	EXPECT_EQ(basisVectors[1].x, 0);
	EXPECT_EQ(basisVectors[1].y, 2);
	EXPECT_EQ(basisVectors[1].z, 0);

	EXPECT_EQ(basisVectors[2].x, 0);
	EXPECT_EQ(basisVectors[2].y, 0);
	EXPECT_EQ(basisVectors[2].z, 2);
}

//TBTKFeature Uncategorized.SpacePartition.getMeshType.1 2019-11-20
TEST_F(SpacePartitionTest, getMeshType1){
	EXPECT_EQ(
		spacePartition[0].getMeshType(),
		SpacePartition::MeshType::Nodal
	);
	EXPECT_EQ(
		spacePartition[1].getMeshType(),
		SpacePartition::MeshType::Interior
	);
	EXPECT_EQ(
		spacePartition[2].getMeshType(),
		SpacePartition::MeshType::Nodal
	);
}

};
