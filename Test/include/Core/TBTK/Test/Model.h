#include "TBTK/Model.h"

#include "gtest/gtest.h"

namespace TBTK{

TEST(Model, Constructor){
	//Check default values.
	Model model;
	EXPECT_DOUBLE_EQ(model.getTemperature(), 0);
	EXPECT_DOUBLE_EQ(model.getChemicalPotential(), 0);
}

TEST(Model, ConstructorCapacity){
	//Check default values.
	Model model;
	EXPECT_DOUBLE_EQ(model.getTemperature(), 0);
	EXPECT_DOUBLE_EQ(model.getChemicalPotential(), 0);
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext, ManyBodyContext, and the Geometry.
TEST(Model, CopyConstructor){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext, ManyBodyContext, and the Geometry.
TEST(Model, MoveConstructor){
}

//TODO
//...
TEST(Model, SerializeToJSON){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext, ManyBodyContext, and the Geometry.
TEST(Model, Destructor){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext, ManyBodyContext, and the Geometry.
TEST(Model, operatorAssignment){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext, ManyBodyContext, and the Geometry.
TEST(Model, operatorMoveAssignment){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, add){
}

TEST(Model, addModel){
	Model model0;
	model0 << HoppingAmplitude(0, {0}, {0});
	model0 << HoppingAmplitude(1, {1}, {1});

	Model model1;
	model1 << HoppingAmplitude(2, {0}, {0});
	model1 << HoppingAmplitude(3, {1}, {1});

	Model model;
	model.setVerbose(false);
	model.addModel(model0, {0});
	model.addModel(model1, {1});
	model.construct();

	EXPECT_EQ(model.getBasisSize(), 4);

	HoppingAmplitudeSet::Iterator iterator = model.getHoppingAmplitudeSet().getIterator();
	EXPECT_EQ(real(iterator.getHA()->getAmplitude()), 0);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({0, 0}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({0, 0}));

	iterator.searchNextHA();
	EXPECT_EQ(real(iterator.getHA()->getAmplitude()), 1);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({0, 1}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({0, 1}));

	iterator.searchNextHA();
	EXPECT_EQ(real(iterator.getHA()->getAmplitude()), 2);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({1, 0}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({1, 0}));

	iterator.searchNextHA();
	EXPECT_EQ(real(iterator.getHA()->getAmplitude()), 3);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({1, 1}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({1, 1}));
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, getBasisIndex){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, getBasisSize){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, construct){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, getIsConstructed){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, sortHoppingAmplitudes){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, constructCOO){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, destructCOO){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, reconstructCOO){
}

TEST(Model, setTemperature){
	Model model;
	model.setTemperature(100);
	EXPECT_DOUBLE_EQ(model.getTemperature(), 100);
	model.setTemperature(200);
	EXPECT_DOUBLE_EQ(model.getTemperature(), 200);
}

TEST(Model, getTemperature){
	//Already tested through Model::setTemperature().
}

TEST(Model, setChemicalPotential){
	Model model;
	model.setChemicalPotential(100);
	EXPECT_DOUBLE_EQ(model.getChemicalPotential(), 100);
	model.setChemicalPotential(200);
	EXPECT_DOUBLE_EQ(model.getChemicalPotential(), 200);
}

TEST(Model, getChemicalPotentialTemperature){
	//Already tested through Model::setChemicalPotential().
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, setStatistics){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, getStatistics){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//SingleParticleContext.
TEST(Model, getHoppingAmplitudeSet){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//Geometry.
TEST(Model, createGeometry){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//Geometry.
TEST(Model, getGeometry){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//ManyBodyContext.
TEST(Model, createManyBodyContext){
}

//TODO
//Should possibly be removed completely by making the Model inherit from the
//ManyBodyContext.
TEST(Model, getManyBodyContext){
}

class HoppingAmplitudeFilter : public AbstractHoppingAmplitudeFilter{
public:
	HoppingAmplitudeFilter* clone() const{
		return new HoppingAmplitudeFilter();
	}

	bool isIncluded(const HoppingAmplitude &hoppingAmplitude) const{
		if(
			hoppingAmplitude.getToIndex()[1] == 1
			|| hoppingAmplitude.getFromIndex()[1] == 1
		){
			return false;
		}
		else{
			return true;
		}
	}
};

class IndexFilter : public AbstractIndexFilter{
public:
	IndexFilter* clone() const{
		return new IndexFilter();
	}

	bool isIncluded(const Index &index) const{
		if(index[1] == 1)
			return false;
		else
			return true;
	}
};

TEST(Model, setFilter){
	//HoppingAmplitudeFilter.
	Model model0;
	model0.setVerbose(false);
	model0.setFilter(HoppingAmplitudeFilter());
	model0 << HoppingAmplitude(0, {0, 0, 0}, {0, 0, 0});
	model0 << HoppingAmplitude(0, {0, 1, 0}, {0, 0, 0});
	model0 << HoppingAmplitude(0, {0, 0, 0}, {0, 1, 0});
	model0 << HoppingAmplitude(0, {0, 2, 0}, {0, 0, 0});
	model0 << HoppingAmplitude(0, {0, 0, 0}, {0, 2, 0});
	model0 << HoppingAmplitude(0, {1, 2, 1}, {1, 0, 1});
	model0 << HoppingAmplitude(0, {1, 0, 1}, {1, 2, 1});
	model0.construct();
	EXPECT_EQ(model0.getBasisSize(), 4);

	//IndexFilter.
	Model model1;
	model1.setVerbose(false);
	model1.setFilter(IndexFilter());
	model1 << HoppingAmplitude(0, {0, 0, 0}, {0, 0, 0});
	model1 << HoppingAmplitude(0, {0, 1, 0}, {0, 0, 0});
	model1 << HoppingAmplitude(0, {0, 0, 0}, {0, 1, 0});
	model1 << HoppingAmplitude(0, {0, 2, 0}, {0, 0, 0});
	model1 << HoppingAmplitude(0, {0, 0, 0}, {0, 2, 0});
	model1 << HoppingAmplitude(0, {1, 2, 1}, {1, 0, 1});
	model1 << HoppingAmplitude(0, {1, 0, 1}, {1, 2, 1});
	model1.construct();
	EXPECT_EQ(model1.getBasisSize(), 4);
}

TEST(Model, operatorInsertion){
	Model model;
	model.setVerbose(false);
	//Normal
	model << HoppingAmplitude(0, {0}, {0});
	//Tuple
	model << HoppingAmplitude(1, {1}, {0}) + HC;
	//HoppingAMplitudeList.
	HoppingAmplitudeList hoppingAmplitudeList;
	hoppingAmplitudeList.pushBack(HoppingAmplitude(2, {2}, {0}));
	hoppingAmplitudeList.pushBack(HoppingAmplitude(2, {0}, {2}));
	model << hoppingAmplitudeList;
	model.construct();

	EXPECT_EQ(model.getBasisSize(), 3);

	HoppingAmplitudeSet::Iterator iterator = model.getHoppingAmplitudeSet().getIterator();
	EXPECT_DOUBLE_EQ(real(iterator.getHA()->getAmplitude()), 0);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({0}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({0}));

	iterator.searchNextHA();
	EXPECT_DOUBLE_EQ(real(iterator.getHA()->getAmplitude()), 1);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({1}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({0}));

	iterator.searchNextHA();
	EXPECT_DOUBLE_EQ(real(iterator.getHA()->getAmplitude()), 2);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({2}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({0}));

	iterator.searchNextHA();
	EXPECT_DOUBLE_EQ(real(iterator.getHA()->getAmplitude()), 1);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({0}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({1}));

	iterator.searchNextHA();
	EXPECT_DOUBLE_EQ(real(iterator.getHA()->getAmplitude()), 2);
	EXPECT_TRUE(iterator.getHA()->getToIndex().equals({0}));
	EXPECT_TRUE(iterator.getHA()->getFromIndex().equals({2}));
}

TEST(Model, serialize){
	//Tested through SerializeToJSON.
}

}; //End of namespace TBTK
