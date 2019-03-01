#include "TBTK/ElementNotFoundException.h"
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
	Model model0;
	model0.setVerbose(false);

	//Add HoppingAmplitude.
	model0 << HoppingAmplitude(1, {0, 1}, {0, 2}) + HC;
	model0 << HoppingAmplitude(2, {1}, {1});
	model0.construct();

	//Add SourceAmplitude.
	model0 << SourceAmplitude(3, {0, 1});
	model0 << SourceAmplitude(4, {0, 2});
	model0 << SourceAmplitude(5, {1});

	//Add OverlapAmplitude.
	model0 << OverlapAmplitude(6, {0, 1}, {0, 1});
	model0 << OverlapAmplitude(7, {0, 1}, {0, 2});
	model0 << OverlapAmplitude(7, {0, 2}, {0, 1});
	model0 << OverlapAmplitude(8, {0, 2}, {0, 2});
	model0 << OverlapAmplitude(9, {0, 1}, {1});
	model0 << OverlapAmplitude(9, {1}, {0, 1});

	//Set chemical potential, temperature, and statistics.
	model0.setChemicalPotential(-1);
	model0.setTemperature(300);
	model0.setStatistics(Statistics::BoseEinstein);

	//Add Geometric data.
	Geometry &geometry0 = model0.getGeometry();
	geometry0.setCoordinate({0, 1}, {0, 0, 0});
	geometry0.setCoordinate({0, 2}, {0, 1, 2});
	geometry0.setCoordinate({1}, {0, 1, 3});

	//Serialize and deserialize.
	Model model1(
		model0.serialize(Serializable::Mode::JSON),
		Serializable::Mode::JSON
	);

	//Check HoppingAmplitudes.
	EXPECT_EQ(model1.getBasisSize(), 3);
	HoppingAmplitudeSet::ConstIterator iteratorHA
		= model1.getHoppingAmplitudeSet().cbegin();

	EXPECT_DOUBLE_EQ(real((*iteratorHA).getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag((*iteratorHA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorHA).getFromIndex().equals({0, 1}));
	EXPECT_TRUE((*iteratorHA).getToIndex().equals({0, 2}));
	++iteratorHA;

	EXPECT_DOUBLE_EQ(real((*iteratorHA).getAmplitude()), 1);
	EXPECT_DOUBLE_EQ(imag((*iteratorHA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorHA).getFromIndex().equals({0, 2}));
	EXPECT_TRUE((*iteratorHA).getToIndex().equals({0, 1}));
	++iteratorHA;

	EXPECT_DOUBLE_EQ(real((*iteratorHA).getAmplitude()), 2);
	EXPECT_DOUBLE_EQ(imag((*iteratorHA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorHA).getFromIndex().equals({1}));
	EXPECT_TRUE((*iteratorHA).getToIndex().equals({1}));
	++iteratorHA;
	EXPECT_TRUE(iteratorHA == model1.getHoppingAmplitudeSet().cend());

	//Check SourceAmplitudes.
	SourceAmplitudeSet::ConstIterator iteratorSA
		= model1.getSourceAmplitudeSet().cbegin();

	EXPECT_DOUBLE_EQ(real((*iteratorSA).getAmplitude()), 3);
	EXPECT_DOUBLE_EQ(imag((*iteratorSA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorSA).getIndex().equals({0, 1}));
	++iteratorSA;

	EXPECT_DOUBLE_EQ(real((*iteratorSA).getAmplitude()), 4);
	EXPECT_DOUBLE_EQ(imag((*iteratorSA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorSA).getIndex().equals({0, 2}));
	++iteratorSA;

	EXPECT_DOUBLE_EQ(real((*iteratorSA).getAmplitude()), 5);
	EXPECT_DOUBLE_EQ(imag((*iteratorSA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorSA).getIndex().equals({1}));
	++iteratorSA;
	EXPECT_TRUE(iteratorSA == model1.getSourceAmplitudeSet().cend());

	//Check OverlapAmplitudes.
	OverlapAmplitudeSet::ConstIterator iteratorOA
		= model1.getOverlapAmplitudeSet().cbegin();

	EXPECT_DOUBLE_EQ(real((*iteratorOA).getAmplitude()), 6);
	EXPECT_DOUBLE_EQ(imag((*iteratorOA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorOA).getBraIndex().equals({0, 1}));
	EXPECT_TRUE((*iteratorOA).getKetIndex().equals({0, 1}));
	++iteratorOA;

	EXPECT_DOUBLE_EQ(real((*iteratorOA).getAmplitude()), 7);
	EXPECT_DOUBLE_EQ(imag((*iteratorOA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorOA).getBraIndex().equals({0, 1}));
	EXPECT_TRUE((*iteratorOA).getKetIndex().equals({0, 2}));
	++iteratorOA;

	EXPECT_DOUBLE_EQ(real((*iteratorOA).getAmplitude()), 9);
	EXPECT_DOUBLE_EQ(imag((*iteratorOA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorOA).getBraIndex().equals({0, 1}));
	EXPECT_TRUE((*iteratorOA).getKetIndex().equals({1}));
	++iteratorOA;

	EXPECT_DOUBLE_EQ(real((*iteratorOA).getAmplitude()), 7);
	EXPECT_DOUBLE_EQ(imag((*iteratorOA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorOA).getBraIndex().equals({0, 2}));
	EXPECT_TRUE((*iteratorOA).getKetIndex().equals({0, 1}));
	++iteratorOA;

	EXPECT_DOUBLE_EQ(real((*iteratorOA).getAmplitude()), 8);
	EXPECT_DOUBLE_EQ(imag((*iteratorOA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorOA).getBraIndex().equals({0, 2}));
	EXPECT_TRUE((*iteratorOA).getKetIndex().equals({0, 2}));
	++iteratorOA;

	EXPECT_DOUBLE_EQ(real((*iteratorOA).getAmplitude()), 9);
	EXPECT_DOUBLE_EQ(imag((*iteratorOA).getAmplitude()), 0);
	EXPECT_TRUE((*iteratorOA).getBraIndex().equals({1}));
	EXPECT_TRUE((*iteratorOA).getKetIndex().equals({0, 1}));
	++iteratorOA;
	EXPECT_TRUE(iteratorOA == model1.getOverlapAmplitudeSet().cend());

	//Check chemical potentail, temperature, and statistics.
	EXPECT_DOUBLE_EQ(model1.getChemicalPotential(), -1);
	EXPECT_DOUBLE_EQ(model1.getTemperature(), 300);
	EXPECT_TRUE(model1.getStatistics() == Statistics::BoseEinstein);

	//Check the Geometry.
	const Geometry &geometry = model1.getGeometry();
	EXPECT_EQ(geometry.getDimensions(), 3);

	std::vector<double> coordinate = geometry.getCoordinate({0, 1});
	EXPECT_EQ(coordinate[0], 0);
	EXPECT_EQ(coordinate[1], 0);
	EXPECT_EQ(coordinate[2], 0);

	coordinate = geometry.getCoordinate({0, 2});
	EXPECT_EQ(coordinate[0], 0);
	EXPECT_EQ(coordinate[1], 1);
	EXPECT_EQ(coordinate[2], 2);

	coordinate = geometry.getCoordinate({1});
	EXPECT_EQ(coordinate[0], 0);
	EXPECT_EQ(coordinate[1], 1);
	EXPECT_EQ(coordinate[2], 3);
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

	HoppingAmplitudeSet::ConstIterator iterator
		= model.getHoppingAmplitudeSet().cbegin();
	EXPECT_EQ(real((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getToIndex().equals({0, 0}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({0, 0}));

	++iterator;
	EXPECT_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_TRUE((*iterator).getToIndex().equals({0, 1}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({0, 1}));

	++iterator;
	EXPECT_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_TRUE((*iterator).getToIndex().equals({1, 0}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({1, 0}));

	++iterator;
	EXPECT_EQ(real((*iterator).getAmplitude()), 3);
	EXPECT_TRUE((*iterator).getToIndex().equals({1, 1}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({1, 1}));
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

	model1 << SourceAmplitude(0, {0, 0, 0});
	model1 << SourceAmplitude(0, {0, 1, 0});
	model1 << SourceAmplitude(0, {0, 2, 0});
	model1 << SourceAmplitude(0, {1, 0, 1});
	model1 << SourceAmplitude(0, {1, 2, 1});
	EXPECT_EQ(model1.getSourceAmplitudeSet().get({0, 0, 0}).size(), 1);
	EXPECT_THROW(
		model1.getSourceAmplitudeSet().get({0, 1, 0}),
		ElementNotFoundException
	);
	EXPECT_EQ(model1.getSourceAmplitudeSet().get({0, 2, 0}).size(), 1);
	EXPECT_EQ(model1.getSourceAmplitudeSet().get({1, 0, 1}).size(), 1);
	EXPECT_EQ(model1.getSourceAmplitudeSet().get({1, 2, 1}).size(), 1);

	model1 << OverlapAmplitude(0, {0, 0, 0}, {0, 0, 0});
	model1 << OverlapAmplitude(0, {0, 1, 0}, {0, 0, 0});
	model1 << OverlapAmplitude(0, {0, 0, 0}, {0, 1, 0});
	model1 << OverlapAmplitude(0, {0, 2, 0}, {0, 0, 0});
	model1 << OverlapAmplitude(0, {0, 0, 0}, {0, 2, 0});
	model1 << OverlapAmplitude(0, {1, 2, 1}, {1, 0, 1});
	model1 << OverlapAmplitude(0, {1, 0, 1}, {1, 2, 1});
	model1.getOverlapAmplitudeSet().get({0, 0, 0}, {0, 0, 0});
	EXPECT_THROW(
		model1.getOverlapAmplitudeSet().get({0, 1, 0}, {0, 0, 0}),
		ElementNotFoundException
	);
	EXPECT_THROW(
		model1.getOverlapAmplitudeSet().get({0, 0, 0}, {0, 1, 0}),
		ElementNotFoundException
	);
	model1.getOverlapAmplitudeSet().get({0, 2, 0}, {0, 0, 0});
	model1.getOverlapAmplitudeSet().get({0, 0, 0}, {0, 2, 0});
	model1.getOverlapAmplitudeSet().get({1, 2, 1}, {1, 0, 1});
	model1.getOverlapAmplitudeSet().get({1, 0, 1}, {1, 2, 1});
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
	hoppingAmplitudeList.add(HoppingAmplitude(2, {2}, {0}));
	hoppingAmplitudeList.add(HoppingAmplitude(2, {0}, {2}));
	model << hoppingAmplitudeList;
	model.construct();

	EXPECT_EQ(model.getBasisSize(), 3);

	HoppingAmplitudeSet::ConstIterator iterator
		= model.getHoppingAmplitudeSet().cbegin();
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 0);
	EXPECT_TRUE((*iterator).getToIndex().equals({0}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({0}));

	++iterator;
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_TRUE((*iterator).getToIndex().equals({1}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({0}));

	++iterator;
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_TRUE((*iterator).getToIndex().equals({2}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({0}));

	++iterator;
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 1);
	EXPECT_TRUE((*iterator).getToIndex().equals({0}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({1}));

	++iterator;
	EXPECT_DOUBLE_EQ(real((*iterator).getAmplitude()), 2);
	EXPECT_TRUE((*iterator).getToIndex().equals({0}));
	EXPECT_TRUE((*iterator).getFromIndex().equals({2}));
}

TEST(Model, serialize){
	//Tested through SerializeToJSON.
}

}; //End of namespace TBTK
