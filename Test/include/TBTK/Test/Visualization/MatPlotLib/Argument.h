#include "TBTK/Visualization/MatPlotLib/Argument.h"

#include "gtest/gtest.h"

namespace TBTK{
namespace Visualization{
namespace MatPlotLib{

class ArgumentTest : public ::testing::Test{
protected:
	Argument arguments[2];

	void SetUp() override{
		arguments[0] = Argument("Argument");
		arguments[1] = Argument({
			{"Key0", "Value0"},
			{"Key1", "Value1"},
		});
	}

	std::string implicitConversion(const Argument &argument){
		std::string result;
		if(argument.getArgumentMap().size() != 0){
			for(auto element : argument.getArgumentMap())
				result += element.first + element.second;
		}
		else{
			result = argument.getArgumentString();
		}

		return result;
	}
};

//TBTKFeature Visualization.MatPlotLib.Argument.getArgumentString.1 2019-11-27
TEST_F(ArgumentTest, getArgumentString1){
	EXPECT_EQ(arguments[0].getArgumentString().compare("Argument"), 0);
	EXPECT_EQ(arguments[1].getArgumentString().compare(""), 0);
}

//TBTKFeature Visualization.MatPlotLib.Argument.getArgumentMap.1 2019-11-27
TEST_F(ArgumentTest, getArgumentMap1){
	EXPECT_EQ(arguments[0].getArgumentMap().size(), 0);

	const std::map<std::string, std::string> argumentMap
		= arguments[1].getArgumentMap();
	EXPECT_EQ(argumentMap.size(), 2);
	EXPECT_EQ(argumentMap.at("Key0").compare("Value0"), 0);
	EXPECT_EQ(argumentMap.at("Key1").compare("Value1"), 0);
}

//TBTKFeature Visualization.MatPlotLib.Argument.implicitConversion.1 2019-11-27
TEST_F(ArgumentTest, implicitConversion1){
	EXPECT_EQ(implicitConversion("MyArgument").compare("MyArgument"), 0);
}

//TBTKFeature Visualization.MatPlotLib.Argument.implicitConversion.2 2019-11-27
TEST_F(ArgumentTest, implicitConversion2){
	EXPECT_EQ(
		implicitConversion({
			{"MyKey0", "MyValue0"},
			{"MyKey1", "MyValue1"}
		}).compare("MyKey0MyValue0MyKey1MyValue1"), 0);
}

};	//End of namespace MatPlotLib
};	//End of namespace Visualization
};	//End of namespace TBTK
